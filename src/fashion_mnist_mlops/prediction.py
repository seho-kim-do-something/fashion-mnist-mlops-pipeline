"""Select, download, and use a trained Fashion-MNIST model."""

from dataclasses import dataclass
from io import BytesIO
import logging
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
import warnings
from zipfile import BadZipFile

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from minio.error import MinioException, S3Error
import numpy as np
from PIL import Image, UnidentifiedImageError
from requests.exceptions import RequestException
import tensorflow as tf
from urllib3.exceptions import HTTPError

from fashion_mnist_mlops.data import preprocess_images
from fashion_mnist_mlops.storage import StorageError, create_client


logger = logging.getLogger(__name__)
CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


class ModelLoadError(RuntimeError):
    """Raised when a trained model cannot be selected or loaded."""


class InvalidImageError(ValueError):
    """Raised when an upload is not a readable image."""


class UnsupportedImageError(ValueError):
    """Raised when an image format is not supported."""


@dataclass(frozen=True)
class ModelReference:
    run_id: str
    validation_accuracy: float
    bucket: str
    object_key: str
    end_time: int


def _select_best_model(client: MlflowClient, experiment_name: str) -> ModelReference:
    """Choose the highest final validation accuracy; if equal, choose the run that finished most recently."""
    best: ModelReference | None = None
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ModelLoadError(f"MLflow experiment '{experiment_name}' was not found.")

        page_token = None
        while True:
            # Read every page so selection does not depend on search limits or run IDs.
            runs = client.search_runs(
                [experiment.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                max_results=1000,
                page_token=page_token,
            )
            for run in runs:
                accuracy = run.data.metrics.get("validation_accuracy")
                uri = run.data.tags.get("minio_model_uri", "")
                if (
                    run.info.status != "FINISHED"
                    or run.info.end_time is None
                    or accuracy is None
                    or not math.isfinite(accuracy)
                    or not 0 <= accuracy <= 1
                    or not uri
                ):
                    continue
                try:
                    location = urlsplit(uri)
                except ValueError:
                    logger.warning("Skipping run %s: invalid minio_model_uri.", run.info.run_id)
                    continue
                object_key = location.path.lstrip("/")
                if (
                    location.scheme != "s3"
                    or not location.netloc
                    or not object_key.endswith(".keras")
                    or location.query
                    or location.fragment
                ):
                    logger.warning("Skipping run %s: invalid minio_model_uri.", run.info.run_id)
                    continue
                candidate = ModelReference(
                    run.info.run_id, accuracy, location.netloc, object_key, run.info.end_time,
                )
                if best is None or (
                    candidate.validation_accuracy, candidate.end_time
                ) > (best.validation_accuracy, best.end_time):
                    best = candidate
            page_token = runs.token
            if not page_token:
                break
    except (MlflowException, RequestException, OSError, ValueError) as error:
        raise ModelLoadError(f"Could not read model runs from MLflow: {error}") from error

    if best is None:
        raise ModelLoadError(
            f"No completed runs with validation_accuracy and a MinIO model reference "
            f"were found in experiment '{experiment_name}'."
        )
    return best


def load_best_model() -> tuple[tf.keras.Model, ModelReference]:
    """Download the selected model once and load it into memory for prediction."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
    except (MlflowException, ValueError) as error:
        raise ModelLoadError(f"Could not configure MLflow at {tracking_uri}: {error}") from error
    reference = _select_best_model(
        client, os.environ.get("MLFLOW_EXPERIMENT_NAME", "fashion-mnist"),
    )
    location = f"s3://{reference.bucket}/{reference.object_key}"
    logger.info(
        "Selected MLflow run %s: validation_accuracy=%s, model=%s",
        reference.run_id, reference.validation_accuracy, location,
    )

    with TemporaryDirectory(prefix="fashion-mnist-model-") as directory:
        model_path = Path(directory) / "model.keras"
        try:
            minio = create_client()
            minio.fget_object(reference.bucket, reference.object_key, str(model_path))
        except S3Error as error:
            raise ModelLoadError(
                f"Could not download selected model {location}: {error.code} - {error.message}"
            ) from error
        except (StorageError, MinioException, HTTPError, OSError, ValueError) as error:
            raise ModelLoadError(f"Could not retrieve selected model {location}: {error}") from error

        try:
            # Prediction needs the saved weights, not the training optimizer state.
            model = tf.keras.models.load_model(model_path, compile=False)
        except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile, tf.errors.OpError) as error:
            raise ModelLoadError(f"Could not load Keras model {location}: {error}") from error
    if model.input_shape != (None, 28, 28, 1) or model.output_shape != (None, 10):
        raise ModelLoadError(f"Model {location} does not have the expected Fashion-MNIST shapes.")
    logger.info("Model from MLflow run %s is ready for prediction.", reference.run_id)
    return model, reference


def _prepare_image(content: bytes) -> np.ndarray:
    """Convert one uploaded image to a normalized (1, 28, 28, 1) batch."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                if image.format not in {"PNG", "JPEG", "BMP"} or getattr(image, "n_frames", 1) != 1:
                    raise UnsupportedImageError("Upload a single-frame PNG, JPEG, or BMP image.")
                image.verify()
            with Image.open(BytesIO(content)) as image:
                grayscale = image.convert("L").resize((28, 28), Image.Resampling.BILINEAR)
                pixels = np.asarray(grayscale)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise InvalidImageError("Image dimensions are too large.") from error
    except UnsupportedImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise InvalidImageError("The upload is not a valid, readable image.") from error
    # Reuse the training normalization rather than scaling pixels a second way.
    return preprocess_images(pixels[np.newaxis, ...])


def predict_class(model: tf.keras.Model, content: bytes) -> str:
    """Return the Fashion-MNIST class name for a single image."""
    batch = _prepare_image(content)
    probabilities = np.asarray(model(batch, training=False))
    if probabilities.shape != (1, 10) or not np.isfinite(probabilities).all():
        raise RuntimeError("The model returned invalid prediction scores.")
    return CLASS_NAMES[int(np.argmax(probabilities[0]))]
