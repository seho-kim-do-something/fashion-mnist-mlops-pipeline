from io import BytesIO
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException
from minio.error import S3Error
import numpy as np
from PIL import Image
import pytest
import tensorflow as tf

import fashion_mnist_mlops.api as api_module
import fashion_mnist_mlops.prediction as prediction
from fashion_mnist_mlops.data import preprocess_images
from fashion_mnist_mlops.storage import StorageError


class RunPage(list):
    """Represent a page of MLflow runs without contacting a server."""

    def __init__(self, runs, token=None):
        super().__init__(runs)
        self.token = token


def make_run(run_id, accuracy=0.9, end_time=100, status="FINISHED", uri=None):
    """Build run metadata using the same metric and tag names as training."""
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, end_time=end_time, status=status),
        data=SimpleNamespace(
            metrics={"validation_accuracy": accuracy},
            tags={"minio_model_uri": uri or f"s3://fashion-minst-bucket/model/{run_id}/model.keras"},
        ),
    )


def tracking_client(*pages):
    """Return a fake MLflow client with the requested search pages."""
    client = Mock()
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="1")
    client.search_runs.side_effect = pages
    return client


def image_bytes(image_format="PNG", mode="RGB", size=(40, 40), color=255):
    """Create a small in-memory image so tests need no downloaded files."""
    stream = BytesIO()
    Image.new(mode, size, color=color).save(stream, format=image_format)
    return stream.getvalue()


def test_selection_uses_highest_final_accuracy_across_pages():
    """Search two pages; expect the highest final validation accuracy, not the newest run."""
    client = tracking_client(
        RunPage([make_run("newer", accuracy=0.88, end_time=300)], token="next"),
        RunPage([make_run("older", accuracy=0.92, end_time=100)]),
    )
    selected = prediction._select_best_model(client, "fashion-mnist")
    assert selected.run_id == "older"
    assert selected.validation_accuracy == 0.92
    assert selected.bucket == "fashion-minst-bucket"
    assert selected.object_key == "model/older/model.keras"
    assert client.search_runs.call_args_list[1].kwargs["page_token"] == "next"


def test_selection_breaks_ties_by_completion_time():
    """Compare equal accuracies; expect the latest completion, regardless of run ID order."""
    client = tracking_client(RunPage([
        make_run("zzz", end_time=100),
        make_run("aaa", end_time=200),
    ]))
    assert prediction._select_best_model(client, "fashion-mnist").run_id == "aaa"


@pytest.mark.parametrize(
    "invalid_run",
    [
        make_run("failed", status="FAILED", accuracy=0.99),
        make_run("running", status="RUNNING", accuracy=0.99),
        make_run("no-end-time", end_time=None),
        make_run("nan", accuracy=float("nan")),
        make_run("infinite", accuracy=float("inf")),
        make_run("out-of-range", accuracy=1.1),
        make_run("wrong-scheme", uri="http://bucket/model.keras"),
        make_run("wrong-extension", uri="s3://bucket/model.txt"),
        make_run("invalid-uri", uri="s3://[invalid/model.keras"),
    ],
)
def test_selection_skips_unsuitable_runs(invalid_run):
    """Include an invalid run; expect only the completed run with valid metadata to qualify."""
    client = tracking_client(RunPage([invalid_run, make_run("valid", accuracy=0.8)]))
    assert prediction._select_best_model(client, "fashion-mnist").run_id == "valid"


def test_selection_requires_metric_and_minio_reference():
    """Remove required metadata; expect a clear error when no suitable runs remain."""
    no_metric = make_run("no-metric")
    no_metric.data.metrics = {"val_accuracy": 0.99}
    no_reference = make_run("no-reference")
    no_reference.data.tags = {}
    client = tracking_client(RunPage([no_metric, no_reference]))
    with pytest.raises(prediction.ModelLoadError, match="No completed runs"):
        prediction._select_best_model(client, "fashion-mnist")


def test_selection_reports_missing_experiment():
    """Look up an absent experiment; expect its name in the error message."""
    client = Mock()
    client.get_experiment_by_name.return_value = None
    with pytest.raises(prediction.ModelLoadError, match="experiment 'missing' was not found"):
        prediction._select_best_model(client, "missing")


def test_selection_reports_mlflow_failure():
    """Make MLflow search fail; expect a clear model-loading error."""
    client = tracking_client()
    client.search_runs.side_effect = MlflowException("server unavailable")
    with pytest.raises(prediction.ModelLoadError, match="Could not read model runs from MLflow"):
        prediction._select_best_model(client, "fashion-mnist")


@pytest.fixture
def selected_model(monkeypatch):
    """Use a selected run and fake MinIO client without contacting either server."""
    reference = prediction.ModelReference("abc123", 0.91, "fashion-minst-bucket", "model/abc123/model.keras", 100)
    client = Mock()
    monkeypatch.setattr(prediction, "MlflowClient", Mock())
    monkeypatch.setattr(prediction, "_select_best_model", Mock(return_value=reference))
    monkeypatch.setattr(prediction, "create_client", Mock(return_value=client))
    return reference, client


def test_load_best_model_downloads_and_loads_keras(selected_model, monkeypatch, caplog):
    """Download a saved Keras model; expect its run details logged and the model loaded."""
    reference, client = selected_model
    original_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    def download(bucket, object_key, file_path):
        original_model.save(file_path)

    client.fget_object.side_effect = download
    caplog.set_level(logging.INFO, logger=prediction.__name__)
    loaded, selected = prediction.load_best_model()
    assert selected == reference
    args = client.fget_object.call_args.args
    assert args[:2] == (reference.bucket, reference.object_key)
    assert not Path(args[2]).exists()
    sample = np.zeros((1, 28, 28, 1), dtype=np.float32)
    np.testing.assert_allclose(loaded(sample), original_model(sample))
    assert reference.run_id in caplog.text
    assert "0.91" in caplog.text
    assert f"s3://{reference.bucket}/{reference.object_key}" in caplog.text


def test_load_best_model_reports_missing_object(selected_model):
    """Make the selected object missing; expect an error, not a fallback to another run."""
    _, client = selected_model
    client.fget_object.side_effect = S3Error(
        response=None, code="NoSuchKey", message="Model does not exist",
        resource="/model.keras", request_id="1", host_id="1",
    )
    with pytest.raises(prediction.ModelLoadError, match="NoSuchKey"):
        prediction.load_best_model()
    assert client.fget_object.call_count == 1


def test_load_best_model_reports_minio_connection_failure(selected_model):
    """Refuse the MinIO connection; expect a clear retrieval error."""
    _, client = selected_model
    client.fget_object.side_effect = ConnectionRefusedError("connection refused")
    with pytest.raises(prediction.ModelLoadError, match="Could not retrieve selected model"):
        prediction.load_best_model()


def test_load_best_model_reports_missing_credentials(selected_model, monkeypatch):
    """Omit MinIO credentials; expect the configuration error to explain loading failure."""
    monkeypatch.setattr(
        prediction, "create_client", Mock(side_effect=StorageError("Credentials are missing")),
    )
    with pytest.raises(prediction.ModelLoadError, match="Credentials are missing"):
        prediction.load_best_model()


def test_load_best_model_reports_corrupt_model(selected_model, monkeypatch):
    """Reject a corrupt Keras archive; expect a clear model-loading error."""
    monkeypatch.setattr(
        prediction.tf.keras.models, "load_model", Mock(side_effect=ValueError("Invalid model")),
    )
    with pytest.raises(prediction.ModelLoadError, match="Could not load Keras model"):
        prediction.load_best_model()


def test_load_best_model_rejects_wrong_shapes(selected_model, monkeypatch):
    """Load an incompatible model; expect rejection before it can serve predictions."""
    model = Mock(input_shape=(None, 32, 32, 3), output_shape=(None, 10))
    monkeypatch.setattr(prediction.tf.keras.models, "load_model", Mock(return_value=model))
    with pytest.raises(prediction.ModelLoadError, match="expected Fashion-MNIST shapes"):
        prediction.load_best_model()


@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "BMP"])
def test_image_preparation_matches_training(image_format):
    """Prepare a white image; expect float32 (1, 28, 28, 1) values normalized to 1."""
    batch = prediction._prepare_image(image_bytes(image_format, mode="L"))
    expected = preprocess_images(np.full((1, 28, 28), 255, dtype=np.uint8))
    assert batch.shape == (1, 28, 28, 1)
    assert batch.dtype == np.float32
    np.testing.assert_array_equal(batch, expected)


def test_image_preparation_converts_rgb_to_grayscale():
    """Prepare a red RGB image; expect the same grayscale pixel values as Pillow."""
    batch = prediction._prepare_image(image_bytes(color=(255, 0, 0)))
    grayscale_value = Image.new("RGB", (1, 1), (255, 0, 0)).convert("L").getpixel((0, 0))
    np.testing.assert_allclose(batch, grayscale_value / 255.0)


@pytest.fixture
def api_client(monkeypatch):
    """Start the API with a fake loaded model and no network access."""
    scores = np.zeros((1, 10), dtype=np.float32)
    scores[0, 7] = 1
    model = Mock(return_value=scores)
    loader = Mock(return_value=(model, None))
    monkeypatch.setattr(api_module, "load_dotenv", Mock())
    monkeypatch.setattr(api_module, "load_best_model", loader)
    with TestClient(api_module.app) as client:
        yield client, model, loader


def test_health_and_prediction(api_client):
    """Send an image; expect health OK, class Sneaker, and exactly one startup load."""
    client, model, loader = api_client
    assert client.get("/health").json() == {"status": "ok"}
    for _ in range(2):
        response = client.post("/predict", files={"file": ("shoe.png", image_bytes(), "image/png")})
        assert response.status_code == 200
        assert response.json() == {"class_name": "Sneaker"}
    assert loader.call_count == 1
    assert model.call_args.args[0].shape == (1, 28, 28, 1)
    assert model.call_args.kwargs == {"training": False}


@pytest.mark.parametrize(
    ("filename", "content", "expected_status"),
    [
        ("empty.png", b"", 400),
        ("broken.png", b"not an image", 400),
        ("unsupported.gif", image_bytes("GIF", mode="L"), 415),
        ("large.png", b"x" * (api_module.MAX_UPLOAD_BYTES + 1), 413),
    ],
)
def test_prediction_rejects_invalid_uploads(api_client, filename, content, expected_status):
    """Send an invalid upload; expect the matching HTTP error without invoking the model."""
    client, model, _ = api_client
    response = client.post("/predict", files={"file": (filename, content)})
    assert response.status_code == expected_status
    model.assert_not_called()


def test_prediction_requires_file(api_client):
    """Omit the image field; expect FastAPI to return HTTP 422."""
    client, _, _ = api_client
    assert client.post("/predict").status_code == 422


def test_prediction_reports_inference_failure(api_client):
    """Make inference fail; expect HTTP 500 while health remains OK."""
    client, model, _ = api_client
    model.side_effect = RuntimeError("inference failed")
    response = client.post("/predict", files={"file": ("image.png", image_bytes())})
    assert response.status_code == 500
    assert response.json()["detail"] == "Model prediction failed; check API logs."
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize(
    "message",
    [
        "No completed runs with validation_accuracy and a MinIO model reference",
        "Could not read model runs from MLflow",
        "Could not retrieve selected model: NoSuchKey",
    ],
)
def test_api_stays_available_without_model(monkeypatch, message):
    """Fail model loading; expect health 200 and prediction 503 with a useful error."""
    loader = Mock(side_effect=prediction.ModelLoadError(message))
    monkeypatch.setattr(api_module, "load_dotenv", Mock())
    monkeypatch.setattr(api_module, "load_best_model", loader)
    with TestClient(api_module.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post("/predict", files={"file": ("image.png", image_bytes())})
        assert response.status_code == 503
        assert message in response.json()["detail"]
        assert loader.call_count == 1
