from dataclasses import replace
import json
import os
from pathlib import Path
from unittest.mock import Mock

from minio.error import S3Error
from mlflow.exceptions import MlflowException
import httpx
import numpy as np
import mlflow
import pytest
import tensorflow as tf

from fashion_mnist_mlops.config import load_config
from fashion_mnist_mlops.data import preprocess_images, split_data
from fashion_mnist_mlops.model import build_model
from fashion_mnist_mlops.storage import StorageError, create_client, upload_model
from fashion_mnist_mlops.train import main, train
from fashion_mnist_mlops.webhook import WebhookError, send_training_complete


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "train.toml"


@pytest.fixture(autouse=True)
def fake_model_upload(monkeypatch):
    """Replace the MinIO upload during training so tests need no MinIO server."""
    uploads = []

    def record_upload(model_path, run_id):
        uploads.append((model_path, run_id))
        return "fashion-minst-bucket", f"model/{run_id}/model.keras"

    monkeypatch.setattr("fashion_mnist_mlops.train.upload_model", record_upload)
    return uploads


@pytest.fixture(autouse=True)
def fake_webhook(monkeypatch):
    """Replace the webhook during training so tests send no HTTP requests."""
    payloads = []

    def record_webhook(payload):
        # The notification must wait until MLflow has marked the run FINISHED.
        run = mlflow.tracking.MlflowClient().get_run(payload["mlflow_run_id"])
        assert run.info.status == "FINISHED"
        payloads.append(payload)
        return 200

    monkeypatch.setattr("fashion_mnist_mlops.train.send_training_complete", record_webhook)
    return payloads


def test_load_config():
    """Load configs/train.toml and check its values.

    Expected: the values match the file (10 epochs, adam, cross_entropy,
    0.2 validation split), and the data and output paths become absolute paths.
    """
    config = load_config(CONFIG_PATH)
    assert config.epochs == 10
    assert config.optimizer == "adam"
    assert config.loss_function == "cross_entropy"
    assert config.validation_split == 0.2
    assert config.data_directory.is_absolute()
    assert config.output_directory.is_absolute()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("epochs = 10", "epochs = 0"),
        ("batch_size = 64", "batch_size = true"),
        ("learning_rate = 0.001", "learning_rate = -0.1"),
        ("learning_rate = 0.001", "learning_rate = nan"),
        ('optimizer = "adam"', 'optimizer = "unknown"'),
        ('loss_function = "cross_entropy"', 'loss_function = "unknown"'),
        ("random_seed = 42", "random_seed = -1"),
        ("random_seed = 42", "random_seed = 4294967296"),
        ("validation_split = 0.2", "validation_split = 1.0"),
        ('directory = "data"', 'directory = ""'),
        ("epochs = 10", "epoch = 10"),
    ],
)
def test_invalid_config(tmp_path, old, new):
    """Replace one valid setting with a bad value and try to load the file.

    Bad values include 0 epochs, a boolean batch size, a negative or NaN
    learning rate, unknown optimizer or loss names, an out-of-range seed,
    a validation split of 1.0, an empty data directory, and a misspelled key.
    Expected: load_config raises ValueError for every case.
    """
    path = tmp_path / "train.toml"
    path.write_text(CONFIG_PATH.read_text().replace(old, new))
    with pytest.raises(ValueError):
        load_config(path)


def test_preprocessing():
    """Preprocess one all-black image and one all-white image.

    Expected: shape changes from (2, 28, 28) to (2, 28, 28, 1), the type is
    float32, and pixel values are scaled from 0-255 to 0.0-1.0.
    """
    images = np.stack(
        [np.zeros((28, 28), dtype=np.uint8), np.full((28, 28), 255, dtype=np.uint8)]
    )
    processed = preprocess_images(images)
    assert processed.shape == (2, 28, 28, 1)
    assert processed.dtype == np.float32
    np.testing.assert_array_equal(processed[0], 0.0)
    np.testing.assert_array_equal(processed[1], 1.0)


def test_preprocessing_rejects_wrong_shape():
    """Preprocess 32x32 images instead of 28x28.

    Expected: ValueError with a message about the shape.
    """
    with pytest.raises(ValueError, match="shape"):
        preprocess_images(np.zeros((2, 32, 32), dtype=np.uint8))


def test_split_is_reproducible_and_disjoint():
    """Split 20 samples with a 0.2 validation split.

    Expected: the same seed gives the same split; 16 samples go to training and
    4 to validation; no sample is in both; together they cover all samples;
    a different seed gives a different split.
    """
    images = np.zeros((20, 28, 28), dtype=np.uint8)
    labels = np.arange(20)
    first = split_data(images, labels, 0.2, 42)
    second = split_data(images, labels, 0.2, 42)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    assert len(first[1]) == 16
    assert len(first[3]) == 4
    assert set(first[1]).isdisjoint(first[3])
    assert set(first[1]) | set(first[3]) == set(labels)
    different = split_data(images, labels, 0.2, 43)
    assert not np.array_equal(first[3], different[3])


def test_split_rejects_empty_validation():
    """Split only 2 samples with a 0.2 validation split.

    2 x 0.2 = 0.4, which rounds down to 0 validation samples.
    Expected: ValueError saying each subset needs at least one sample.
    """
    with pytest.raises(ValueError, match="at least one sample"):
        split_data(np.zeros((2, 28, 28), dtype=np.uint8), np.zeros(2), 0.2, 42)


@pytest.mark.parametrize("optimizer", ["adam", "sgd", "rmsprop"])
@pytest.mark.parametrize("loss_function", ["cross_entropy", "sparse_categorical_crossentropy"])
def test_model_shape_and_configuration(optimizer, loss_function):
    """Build the model for every optimizer and loss name combination (6 cases).

    Expected: Keras uses the TensorFlow backend; input shape is (28, 28, 1);
    output has 10 classes; the chosen optimizer and learning rate are used;
    the loss is sparse categorical cross-entropy; each output row sums to 1.
    """
    config = replace(load_config(CONFIG_PATH), optimizer=optimizer, loss_function=loss_function)
    model = build_model(config)
    assert tf.keras.backend.backend() == "tensorflow"
    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 10)
    assert type(model.optimizer).__name__.lower() == optimizer
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(config.learning_rate)
    assert isinstance(model.loss, tf.keras.losses.SparseCategoricalCrossentropy)
    probabilities = model(np.zeros((2, 28, 28, 1), dtype=np.float32)).numpy()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_training_saves_reproducible_runs(tmp_path, monkeypatch):
    """Train twice on 40 fake images with the same seed for 1 epoch.

    Expected: each run gets its own directory with config.json, history.json,
    metrics.json, and model.keras; saved settings and sample counts (32 train,
    8 validation) are correct; final metrics match the last history values;
    both saved models give the same predictions and weights.
    """
    images = np.random.default_rng(7).integers(0, 256, (40, 28, 28), dtype=np.uint8)
    labels = np.tile(np.arange(10, dtype=np.uint8), 4)
    # Replace downloads with synthetic data to keep the test small and offline.
    monkeypatch.setattr(
        "fashion_mnist_mlops.train.load_training_data", lambda directory: (images, labels),
    )
    # Use a temporary local MLflow store instead of the MLflow server.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    config = replace(
        load_config(CONFIG_PATH), epochs=1, batch_size=8, output_directory=tmp_path / "runs",
    )
    first = train(config)
    second = train(config)
    assert first != second
    assert {path.name for path in first.iterdir()} == {
        "config.json", "history.json", "metrics.json", "model.keras",
    }
    saved_config = json.loads((first / "config.json").read_text())
    assert saved_config["random_seed"] == config.random_seed
    assert saved_config["epochs"] == 1
    assert saved_config["runtime"]["training_samples"] == 32
    assert saved_config["runtime"]["validation_samples"] == 8
    assert saved_config["runtime"]["backend"] == "tensorflow"
    history = json.loads((first / "history.json").read_text())
    assert {"loss", "val_loss", "val_accuracy"} <= history.keys()
    metrics = json.loads((first / "metrics.json").read_text())
    assert metrics["training_loss"] == history["loss"][-1]
    assert metrics["validation_loss"] == history["val_loss"][-1]
    assert metrics["validation_accuracy"] == history["val_accuracy"][-1]
    assert 0 <= metrics["validation_accuracy"] <= 1
    first_model = tf.keras.models.load_model(first / "model.keras")
    second_model = tf.keras.models.load_model(second / "model.keras")
    sample = preprocess_images(images[:2])
    np.testing.assert_allclose(first_model(sample), second_model(sample), rtol=1e-6, atol=1e-6)
    for left, right in zip(first_model.get_weights(), second_model.get_weights()):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)


def test_training_logs_to_mlflow(tmp_path, monkeypatch, fake_model_upload, fake_webhook):
    """Train for 2 epochs on fake data and read the run back from a local MLflow store.

    Expected: one FINISHED run in the test experiment; all training settings
    and sample counts are logged as parameters; loss, val_loss, accuracy, and
    val_accuracy each have values for steps 0 and 1; final metrics match
    metrics.json; the local run directory tag is set; the four run files are
    uploaded as artifacts; the model is uploaded once using the MLflow run ID,
    and its MinIO location is saved in the minio_model_uri tag; one webhook is
    sent with status, run ID, bucket, object key, and validation accuracy.
    """
    images = np.random.default_rng(7).integers(0, 256, (40, 28, 28), dtype=np.uint8)
    labels = np.tile(np.arange(10, dtype=np.uint8), 4)
    monkeypatch.setattr(
        "fashion_mnist_mlops.train.load_training_data", lambda directory: (images, labels),
    )
    tracking_uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-experiment")
    config = replace(
        load_config(CONFIG_PATH), epochs=2, batch_size=8, output_directory=tmp_path / "runs",
    )
    run_directory = train(config)

    client = mlflow.tracking.MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name("test-experiment")
    [run] = client.search_runs([experiment.experiment_id])
    assert run.info.status == "FINISHED"

    params = run.data.params
    for key in ("epochs", "batch_size", "learning_rate", "optimizer",
                "loss_function", "random_seed", "validation_split"):
        assert params[key] == str(getattr(config, key))
    assert params["training_samples"] == "32"
    assert params["validation_samples"] == "8"

    # Two epochs produce two values for each per-epoch metric.
    for key in ("loss", "val_loss", "accuracy", "val_accuracy"):
        history = client.get_metric_history(run.info.run_id, key)
        assert [metric.step for metric in history] == [0, 1]

    saved_metrics = json.loads((run_directory / "metrics.json").read_text())
    for key, value in saved_metrics.items():
        assert run.data.metrics[key] == pytest.approx(value)

    assert run.data.tags["local_run_directory"] == str(run_directory)
    artifacts = {item.path for item in client.list_artifacts(run.info.run_id)}
    assert artifacts == {"config.json", "history.json", "metrics.json", "model.keras"}

    assert fake_model_upload == [(run_directory / "model.keras", run.info.run_id)]
    assert run.data.tags["minio_model_uri"] == (
        f"s3://fashion-minst-bucket/model/{run.info.run_id}/model.keras"
    )

    assert fake_webhook == [{
        "status": "completed",
        "mlflow_run_id": run.info.run_id,
        "minio_bucket": "fashion-minst-bucket",
        "minio_object_key": f"model/{run.info.run_id}/model.keras",
        "validation_accuracy": saved_metrics["validation_accuracy"],
    }]


def test_failed_training_marks_mlflow_run_failed(tmp_path, monkeypatch, fake_webhook):
    """Make data loading fail during training.

    Expected: the original error is raised to the caller, the MLflow run
    is saved with status FAILED, and no webhook is sent.
    """
    def fail_loading(directory):
        raise RuntimeError("download failed")

    monkeypatch.setattr("fashion_mnist_mlops.train.load_training_data", fail_loading)
    tracking_uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-experiment")
    config = replace(load_config(CONFIG_PATH), output_directory=tmp_path / "runs")
    with pytest.raises(RuntimeError, match="download failed"):
        train(config)

    client = mlflow.tracking.MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name("test-experiment")
    [run] = client.search_runs([experiment.experiment_id])
    assert run.info.status == "FAILED"
    assert fake_webhook == []


def test_failed_upload_sends_no_webhook(tmp_path, monkeypatch, fake_webhook):
    """Make the MinIO upload fail after training.

    Expected: StorageError is raised, the MLflow run is FAILED, and no webhook is sent.
    """
    images = np.random.default_rng(7).integers(0, 256, (40, 28, 28), dtype=np.uint8)
    labels = np.tile(np.arange(10, dtype=np.uint8), 4)
    monkeypatch.setattr(
        "fashion_mnist_mlops.train.load_training_data", lambda directory: (images, labels),
    )

    def fail_upload(model_path, run_id):
        raise StorageError("MinIO is down")

    monkeypatch.setattr("fashion_mnist_mlops.train.upload_model", fail_upload)
    tracking_uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-experiment")
    config = replace(
        load_config(CONFIG_PATH), epochs=1, batch_size=8, output_directory=tmp_path / "runs",
    )
    with pytest.raises(StorageError, match="MinIO is down"):
        train(config)

    client = mlflow.tracking.MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name("test-experiment")
    [run] = client.search_runs([experiment.experiment_id])
    assert run.info.status == "FAILED"
    assert fake_webhook == []


class FakeMinio:
    """Stand-in for the MinIO client that records calls instead of using the network."""

    def __init__(self, *args, bucket_exists=False, error=None, **kwargs):
        self.exists = bucket_exists
        self.error = error
        self.created_buckets = []
        self.uploads = []

    def bucket_exists(self, bucket):
        if self.error:
            raise self.error
        return self.exists

    def make_bucket(self, bucket):
        self.created_buckets.append(bucket)

    def fput_object(self, bucket, object_key, file_path):
        self.uploads.append((bucket, object_key, file_path))

    def stat_object(self, bucket, object_key):
        return object()


@pytest.fixture
def minio_settings(monkeypatch, tmp_path):
    """Set MinIO settings and create a small model file to upload."""
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test-user")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test-password")
    monkeypatch.setenv("MINIO_BUCKET", "fashion-minst-bucket")
    monkeypatch.setenv("MINIO_MODEL_PREFIX", "model/")
    model_path = tmp_path / "model.keras"
    model_path.write_bytes(b"model")
    return model_path


def use_fake_client(monkeypatch, client):
    """Make storage.create_client return the given fake client."""
    monkeypatch.setattr("fashion_mnist_mlops.storage.Minio", lambda *args, **kwargs: client)


def test_upload_model_creates_bucket_and_uses_run_id(monkeypatch, minio_settings):
    """Upload a model when the bucket does not exist yet.

    Expected: the bucket fashion-minst-bucket is created; the file is uploaded
    as model/<run-id>/model.keras; the bucket and object key are returned.
    """
    client = FakeMinio(bucket_exists=False)
    use_fake_client(monkeypatch, client)
    location = upload_model(minio_settings, "abc123")
    assert client.created_buckets == ["fashion-minst-bucket"]
    assert client.uploads == [
        ("fashion-minst-bucket", "model/abc123/model.keras", str(minio_settings)),
    ]
    assert location == ("fashion-minst-bucket", "model/abc123/model.keras")


def test_upload_model_reuses_existing_bucket(monkeypatch, minio_settings):
    """Upload a model when the bucket already exists.

    Expected: no new bucket is created, and the model is still uploaded.
    """
    client = FakeMinio(bucket_exists=True)
    use_fake_client(monkeypatch, client)
    upload_model(minio_settings, "abc123")
    assert client.created_buckets == []
    assert len(client.uploads) == 1


def test_upload_model_requires_credentials(monkeypatch, minio_settings):
    """Upload without MINIO_ACCESS_KEY and MINIO_SECRET_KEY.

    Expected: StorageError saying both values must be set in .env.
    """
    monkeypatch.delenv("MINIO_ACCESS_KEY")
    monkeypatch.delenv("MINIO_SECRET_KEY")
    with pytest.raises(StorageError, match="MINIO_ACCESS_KEY and MINIO_SECRET_KEY"):
        upload_model(minio_settings, "abc123")


def test_upload_model_reports_missing_file(tmp_path, minio_settings):
    """Upload a model file that does not exist.

    Expected: StorageError saying the model file was not found.
    """
    with pytest.raises(StorageError, match="Model file not found"):
        upload_model(tmp_path / "missing.keras", "abc123")


def test_upload_model_reports_connection_failure(monkeypatch, minio_settings):
    """Upload while MinIO cannot be reached (connection refused).

    Expected: StorageError saying MinIO could not be reached.
    """
    use_fake_client(monkeypatch, FakeMinio(error=ConnectionRefusedError("refused")))
    with pytest.raises(StorageError, match="Could not connect to MinIO"):
        upload_model(minio_settings, "abc123")


def test_upload_model_reports_rejected_request(monkeypatch, minio_settings):
    """Upload with credentials that MinIO rejects.

    Expected: StorageError that includes the MinIO error code, e.g. InvalidAccessKeyId.
    """
    error = S3Error(
        response=None, code="InvalidAccessKeyId", message="Unknown access key",
        resource="/fashion-minst-bucket", request_id="1", host_id="1",
    )
    use_fake_client(monkeypatch, FakeMinio(error=error))
    with pytest.raises(StorageError, match="InvalidAccessKeyId"):
        upload_model(minio_settings, "abc123")


WEBHOOK_PAYLOAD = {
    "status": "completed",
    "mlflow_run_id": "abc123",
    "minio_bucket": "fashion-minst-bucket",
    "minio_object_key": "model/abc123/model.keras",
    "validation_accuracy": 0.9,
}


def test_webhook_posts_json_payload(monkeypatch):
    """Send the webhook to a server that answers HTTP 200.

    Expected: one POST to WEBHOOK_URL with the payload as JSON, and status 200 is returned.
    """
    requests = []

    def fake_post(url, json, timeout):
        requests.append((url, json))
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setenv("WEBHOOK_URL", "http://example.test/post")
    monkeypatch.setattr("fashion_mnist_mlops.webhook.httpx.post", fake_post)
    assert send_training_complete(WEBHOOK_PAYLOAD) == 200
    assert requests == [("http://example.test/post", WEBHOOK_PAYLOAD)]


def test_webhook_reports_connection_error(monkeypatch):
    """Send the webhook while the server cannot be reached.

    Expected: WebhookError saying the webhook could not be reached.
    """
    def fake_post(url, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("fashion_mnist_mlops.webhook.httpx.post", fake_post)
    with pytest.raises(WebhookError, match="Could not reach webhook"):
        send_training_complete(WEBHOOK_PAYLOAD)


def test_webhook_reports_error_status(monkeypatch):
    """Send the webhook to a server that answers HTTP 500.

    Expected: WebhookError that includes the status code 500.
    """
    def fake_post(url, json, timeout):
        return httpx.Response(500, text="server error", request=httpx.Request("POST", url))

    monkeypatch.setattr("fashion_mnist_mlops.webhook.httpx.post", fake_post)
    with pytest.raises(WebhookError, match="HTTP 500"):
        send_training_complete(WEBHOOK_PAYLOAD)


@pytest.mark.parametrize("existing_value", [None, "shell-experiment"])
def test_training_loads_dotenv_without_overriding_environment(tmp_path, monkeypatch, existing_value):
    """Start the command; expect .env to load while existing environment values take priority."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("MLFLOW_EXPERIMENT_NAME=file-experiment\n", encoding="utf-8")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    if existing_value is not None:
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", existing_value)
    monkeypatch.setattr("sys.argv", ["fashion-mnist-train", "--config", str(CONFIG_PATH)])
    values = []

    def record_training(config):
        values.append(os.environ["MLFLOW_EXPERIMENT_NAME"])

    monkeypatch.setattr("fashion_mnist_mlops.train.train", record_training)
    main()
    assert values == [existing_value or "file-experiment"]


@pytest.mark.parametrize("operation", ["log_params", "log_artifacts", "set_tag"])
def test_mlflow_logging_failure_sends_no_webhook(tmp_path, monkeypatch, fake_webhook, operation):
    """Fail required MLflow logging; expect a FAILED run and no success notification."""
    images = np.zeros((40, 28, 28), dtype=np.uint8)
    labels = np.tile(np.arange(10, dtype=np.uint8), 4)
    monkeypatch.setattr(
        "fashion_mnist_mlops.train.load_training_data", lambda directory: (images, labels),
    )
    monkeypatch.setattr(
        mlflow, operation, Mock(side_effect=MlflowException("logging unavailable")),
    )
    tracking_uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-experiment")
    config = replace(
        load_config(CONFIG_PATH), epochs=1, batch_size=8, output_directory=tmp_path / "runs",
    )
    with pytest.raises(MlflowException, match="logging unavailable"):
        train(config)
    client = mlflow.tracking.MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name("test-experiment")
    [run] = client.search_runs([experiment.experiment_id])
    assert run.info.status == "FAILED"
    assert fake_webhook == []


def test_upload_checks_that_object_exists(monkeypatch, minio_settings):
    """Fail the check after upload; expect a missing-object error rather than success."""
    client = FakeMinio(bucket_exists=True)
    client.stat_object = Mock(side_effect=S3Error(
        response=None, code="NoSuchKey", message="Object missing",
        resource="/model.keras", request_id="1", host_id="1",
    ))
    use_fake_client(monkeypatch, client)
    with pytest.raises(StorageError, match="was not found in MinIO"):
        upload_model(minio_settings, "abc123")
    assert len(client.uploads) == 1


def test_minio_reports_invalid_endpoint(monkeypatch, minio_settings):
    """Reject malformed MinIO configuration; expect a clear StorageError."""
    monkeypatch.setenv("MINIO_ENDPOINT", "localhost:9000/invalid-path")
    with pytest.raises(StorageError, match="Invalid MinIO configuration"):
        create_client()


def test_webhook_reports_invalid_url(monkeypatch):
    """Reject an invalid webhook URL; expect a clear WebhookError."""
    monkeypatch.setattr(
        "fashion_mnist_mlops.webhook.httpx.post",
        Mock(side_effect=httpx.InvalidURL("Invalid URL")),
    )
    with pytest.raises(WebhookError, match="Could not reach webhook"):
        send_training_complete(WEBHOOK_PAYLOAD)
