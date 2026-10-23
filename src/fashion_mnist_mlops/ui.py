"""Streamlit interface for the FastAPI prediction service."""

from io import BytesIO
import logging
import os
import warnings

from dotenv import load_dotenv
import httpx
from PIL import Image, UnidentifiedImageError
import streamlit as st


logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class PredictionError(RuntimeError):
    """Raised when the backend cannot return a usable prediction."""


def _validate_image(content: bytes) -> None:
    """Check the image before previewing it or sending it to the backend."""
    if not content:
        raise ValueError("The file is empty. Please upload an image.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Please choose an image that is 5 MiB or smaller.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                if image.format not in {"PNG", "JPEG", "BMP"} or getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Please upload a single-frame PNG, JPEG, or BMP image.")
                image.verify()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("The image dimensions are too large. Please choose a smaller image.") from error
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise ValueError("This file is not a readable image. Please choose another image.") from error


def _request_prediction(content: bytes, filename: str) -> str:
    """Send the original image using FastAPI's multipart file format."""
    base_url = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
    try:
        response = httpx.post(
            f"{base_url}/predict",
            files={"file": (filename, content)},
            timeout=30,
        )
    except httpx.TimeoutException as error:
        raise PredictionError("The prediction request timed out. Please try again.") from error
    except (httpx.RequestError, httpx.InvalidURL) as error:
        raise PredictionError(
            "Cannot connect to the prediction API. Check that FastAPI is running "
            "and API_BASE_URL is correct."
        ) from error

    if response.status_code == 503:
        raise PredictionError(
            "No trained model is available. Complete a training run, check the API logs, "
            "then restart the API to load the model."
        )
    if response.status_code in {400, 413, 415, 422}:
        try:
            body = response.json()
        except ValueError:
            body = None
        detail = body.get("detail") if isinstance(body, dict) else None
        message = detail if isinstance(detail, str) else "Please upload a valid PNG, JPEG, or BMP image."
        raise PredictionError(message)
    if not response.is_success:
        raise PredictionError(
            f"The prediction API returned HTTP {response.status_code}. "
            "Please check the API logs and try again."
        )
    try:
        result = response.json()
    except ValueError as error:
        raise PredictionError("The prediction API returned an unreadable response.") from error
    class_name = result.get("class_name") if isinstance(result, dict) else None
    if not isinstance(class_name, str) or not class_name.strip():
        raise PredictionError("The prediction API response did not contain a class name.")
    return class_name


def main() -> None:
    """Display the upload form and request predictions only when the button is clicked."""
    load_dotenv(".env")
    logging.basicConfig(level=logging.INFO)
    st.set_page_config(page_title="Fashion-MNIST Classifier", layout="centered")
    st.title("Fashion-MNIST Classifier")
    st.write("Upload a clothing image and let the model predict its category.")
    st.caption("For best results, use one light-colored item on a dark background.")

    uploaded = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=False,
        help="PNG, JPEG, or BMP. Maximum size: 5 MiB.",
    )
    if uploaded is None:
        st.info("Upload an image to get started.")
        return
    content = uploaded.getvalue()
    try:
        _validate_image(content)
    except ValueError as error:
        logger.warning("UI upload rejected: %s", error)
        st.error(str(error))
        return

    st.image(content, caption="Uploaded image", width=280)
    # Streamlit reruns on input changes; predict only after an explicit button click.
    if st.button("Predict class", type="primary"):
        try:
            with st.spinner("Predicting..."):
                class_name = _request_prediction(content, uploaded.name)
        except PredictionError as error:
            logger.exception("UI prediction request failed.")
            st.error(str(error))
        else:
            st.success(f"Predicted class: {class_name}")
            logger.info("UI prediction completed: class_name=%s", class_name)


if __name__ == "__main__":
    main()
