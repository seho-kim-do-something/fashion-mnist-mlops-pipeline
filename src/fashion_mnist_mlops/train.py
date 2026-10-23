"""Train and save a Fashion-MNIST classifier."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
from uuid import uuid4

from dotenv import load_dotenv
import mlflow
import numpy as np
import tensorflow as tf

from fashion_mnist_mlops.config import TrainingConfig, load_config
from fashion_mnist_mlops.data import load_training_data, split_data
from fashion_mnist_mlops.model import build_model
from fashion_mnist_mlops.storage import StorageError, upload_model
from fashion_mnist_mlops.webhook import WebhookError, send_training_complete


logger = logging.getLogger(__name__)


def train(config: TrainingConfig) -> Path:
    """Train, track in MLflow, store the model, then notify; return the local run path."""
    # Connect to MLflow before training so connection errors appear early.
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "fashion-mnist"))

    # The run is marked FAILED in MLflow if any error occurs inside this block.
    with mlflow.start_run():
        run_directory, payload = _train_and_log(config)

    # Reached only after training, MLflow logging, and the model upload all succeed.
    send_training_complete(payload)
    return run_directory


def _train_and_log(config: TrainingConfig) -> tuple[Path, dict]:
    """Train, save local files, log to MLflow, and upload the model.

    Returns the local run directory and the training-complete webhook payload.
    """
    # Clear Keras state left by previously created models.
    tf.keras.backend.clear_session()
    # Set Python, NumPy, and TensorFlow seeds using random_seed from train.toml.
    tf.keras.utils.set_random_seed(config.random_seed)
    # Make TensorFlow operations repeatable on the same software and hardware.
    tf.config.experimental.enable_op_determinism()

    images, labels = load_training_data(config.data_directory)
    x_train, y_train, x_validation, y_validation = split_data(
        images, labels, config.validation_split, config.random_seed,
    )
    model = build_model(config)

    # A unique directory prevents runs from overwriting earlier results.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_directory = config.output_directory / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    recorded_config = asdict(config)
    recorded_config["data_directory"] = str(config.data_directory)
    recorded_config["output_directory"] = str(config.output_directory)
    # Record library versions and resolved optimizer defaults alongside user settings.
    recorded_config["runtime"] = {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "numpy": np.__version__,
        "backend": tf.keras.backend.backend(),
        "deterministic_operations": True,
        "training_samples": len(y_train),
        "validation_samples": len(y_validation),
        "optimizer": tf.keras.optimizers.serialize(model.optimizer),
        "loss_function": tf.keras.losses.serialize(model.loss),
    }
    (run_directory / "config.json").write_text(
        json.dumps(recorded_config, indent=2) + "\n", encoding="utf-8",
    )

    # Log the training settings from train.toml and the dataset split sizes.
    mlflow.log_params({
        **asdict(config),
        "training_samples": len(y_train),
        "validation_samples": len(y_validation),
    })
    mlflow.set_tags({
        "model_architecture": model.name,
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "local_run_directory": str(run_directory),
    })

    # Shuffle only training data; the seed makes epoch orderings repeatable across runs.
    training_data = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(y_train), seed=config.random_seed, reshuffle_each_iteration=True)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    validation_data = (
        tf.data.Dataset.from_tensor_slices((x_validation, y_validation))
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    # Log loss and accuracy at the end of each epoch, e.g. loss and val_loss at step 0, 1, 2...
    log_epoch_metrics = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: mlflow.log_metrics(logs, step=epoch),
    )
    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=config.epochs,
        shuffle=False,  # The training dataset handles shuffling.
        verbose=2,
        callbacks=[log_epoch_metrics],
    )

    model.save(run_directory / "model.keras")
    (run_directory / "history.json").write_text(
        json.dumps(history.history, indent=2) + "\n", encoding="utf-8",
    )
    metrics = {
        "training_loss": history.history["loss"][-1],
        "validation_loss": history.history["val_loss"][-1],
        "validation_accuracy": history.history["val_accuracy"][-1],
    }
    (run_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8",
    )

    # Log final metrics and upload the model, configuration, and history files.
    mlflow.log_metrics(metrics)
    mlflow.log_artifacts(str(run_directory))

    # Store the model in MinIO under the MLflow run ID so earlier models are never overwritten.
    mlflow_run_id = mlflow.active_run().info.run_id
    bucket, object_key = upload_model(run_directory / "model.keras", mlflow_run_id)
    model_uri = f"s3://{bucket}/{object_key}"
    mlflow.set_tag("minio_model_uri", model_uri)

    print(f"MLflow run ID: {mlflow_run_id}")
    print(f"Model uploaded to: {model_uri}")
    print(f"Training loss: {metrics['training_loss']:.4f}")
    print(f"Validation loss: {metrics['validation_loss']:.4f}")
    print(f"Validation accuracy: {metrics['validation_accuracy']:.2%}")
    print(f"Run saved to: {run_directory}")
    payload = {
        "status": "completed",
        "mlflow_run_id": mlflow_run_id,
        "minio_bucket": bucket,
        "minio_object_key": object_key,
        "validation_accuracy": metrics["validation_accuracy"],
    }
    return run_directory, payload


def main() -> None:
    """Read the configuration path from the command line and start training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.toml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    # Load settings from .env in the current directory; existing environment variables take priority.
    load_dotenv(".env")
    try:
        train(load_config(args.config))
    except StorageError as error:
        logger.exception("Model upload failed: %s", error)
        raise SystemExit(1) from None
    except WebhookError as error:
        # The run and model are already saved; only the notification failed.
        logger.exception("Training completed, but the webhook failed: %s", error)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
