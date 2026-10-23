"""Upload trained models to MinIO object storage."""

import os
from pathlib import Path

from minio import Minio
from minio.error import MinioException, S3Error
from urllib3.exceptions import HTTPError


class StorageError(RuntimeError):
    """Raised when a model cannot be stored in MinIO."""


def create_client() -> Minio:
    """Create a MinIO client from the MINIO_* environment variables."""
    access_key = os.environ.get("MINIO_ACCESS_KEY")
    secret_key = os.environ.get("MINIO_SECRET_KEY")
    if not access_key or not secret_key:
        raise StorageError("MINIO_ACCESS_KEY and MINIO_SECRET_KEY must be set in .env.")
    try:
        return Minio(
            os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
            access_key=access_key,
            secret_key=secret_key,
            # MINIO_SECURE=true switches from HTTP to HTTPS.
            secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
        )
    except ValueError as error:
        raise StorageError(
            f"Invalid MinIO configuration. Use MINIO_ENDPOINT in host:port format: {error}"
        ) from error


def upload_model(model_path: Path, run_id: str) -> tuple[str, str]:
    """Upload a model file under <prefix>/<run_id>/ and return (bucket, object_key)."""
    if not model_path.is_file():
        raise StorageError(f"Model file not found: {model_path}")

    bucket = os.environ.get("MINIO_BUCKET", "fashion-minst-bucket")
    prefix = os.environ.get("MINIO_MODEL_PREFIX", "model/").strip("/")
    # The run ID keeps each model separate, e.g. model/<run-id>/model.keras.
    object_key = f"{prefix}/{run_id}/{model_path.name}"
    endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")

    client = create_client()
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
        client.fput_object(bucket, object_key, str(model_path))
        # Confirm the object is stored before reporting success.
        client.stat_object(bucket, object_key)
    except S3Error as error:
        if error.code == "NoSuchKey":
            raise StorageError(
                f"Upload finished, but {bucket}/{object_key} was not found in MinIO."
            ) from error
        raise StorageError(
            f"MinIO rejected the request for {bucket}/{object_key}: {error.code} - {error.message}"
        ) from error
    # Connection refused, timeouts, and unexpected responses end up here.
    except (MinioException, HTTPError, OSError) as error:
        raise StorageError(
            f"Could not connect to MinIO at {endpoint}. Is the service running? ({error})"
        ) from error

    return bucket, object_key
