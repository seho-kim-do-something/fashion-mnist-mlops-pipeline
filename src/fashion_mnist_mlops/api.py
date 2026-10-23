"""FastAPI endpoints for Fashion-MNIST prediction."""

from contextlib import asynccontextmanager
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
import tensorflow as tf

from fashion_mnist_mlops.prediction import (
    InvalidImageError,
    ModelLoadError,
    UnsupportedImageError,
    load_best_model,
    predict_class,
)


logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the best model once, but keep HTTP endpoints available if loading fails."""
    logging.basicConfig(level=logging.INFO)
    load_dotenv(".env")
    app.state.model = None
    app.state.model_error = None
    try:
        app.state.model, _ = load_best_model()
    except (ModelLoadError, OSError) as error:
        app.state.model_error = str(error)
        logger.exception("Prediction model unavailable: %s", error)
    yield
    app.state.model = None


app = FastAPI(title="Fashion-MNIST Prediction", lifespan=lifespan)


class PredictionResponse(BaseModel):
    class_name: str


@app.get("/health")
def health() -> dict[str, str]:
    """Confirm the API is running, independently of model availability."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    """Accept one image file and return its predicted clothing class."""
    if request.app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Prediction model unavailable: {request.app.state.model_error}",
        )
    # Limit the bytes read into memory and validate image contents, not just the filename.
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        logger.warning("Upload rejected: image exceeds the 5 MiB limit.")
        raise HTTPException(status_code=413, detail="Image must be 5 MiB or smaller.")
    if not content:
        logger.warning("Upload rejected: file is empty.")
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    try:
        class_name = predict_class(request.app.state.model, content)
    except UnsupportedImageError as error:
        logger.warning("Upload rejected: %s", error)
        raise HTTPException(status_code=415, detail=str(error)) from error
    except InvalidImageError as error:
        logger.warning("Upload rejected: %s", error)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except (RuntimeError, ValueError, tf.errors.OpError) as error:
        logger.exception("Model prediction failed.")
        raise HTTPException(status_code=500, detail="Model prediction failed; check API logs.") from error
    logger.info("Prediction completed: class_name=%s", class_name)
    return PredictionResponse(class_name=class_name)
