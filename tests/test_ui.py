from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, Mock

import httpx
from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

from fashion_mnist_mlops import ui


def image_bytes(image_format="PNG"):
    """Create a small test image without reading files or downloading data."""
    stream = BytesIO()
    Image.new("L", (28, 28), 255).save(stream, format=image_format)
    return stream.getvalue()


@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "BMP"])
def test_ui_accepts_supported_images(image_format):
    """Validate a supported image; expect no error and no API request."""
    ui._validate_image(image_bytes(image_format))


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "empty"),
        (b"not an image", "not a readable image"),
        (b"x" * (ui.MAX_UPLOAD_BYTES + 1), "5 MiB"),
        (image_bytes("GIF"), "single-frame PNG, JPEG, or BMP"),
    ],
)
def test_ui_rejects_invalid_images(content, message):
    """Validate an invalid upload; expect a helpful error before preview or prediction."""
    with pytest.raises(ValueError, match=message):
        ui._validate_image(content)


def test_ui_posts_image_to_configured_api(monkeypatch):
    """Request prediction; expect the original multipart file and returned class name."""
    post = Mock(return_value=httpx.Response(200, json={"class_name": "Bag"}))
    monkeypatch.setattr(ui.httpx, "post", post)
    monkeypatch.setenv("API_BASE_URL", "http://api:8000/")
    content = image_bytes()
    assert ui._request_prediction(content, "bag.png") == "Bag"
    post.assert_called_once_with(
        "http://api:8000/predict", files={"file": ("bag.png", content)}, timeout=30,
    )


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (httpx.ConnectError("refused"), "Cannot connect"),
        (httpx.ReadTimeout("timeout"), "timed out"),
        (httpx.InvalidURL("invalid"), "Cannot connect"),
    ],
)
def test_ui_reports_connection_errors(monkeypatch, error, message):
    """Fail an API request; expect a clear connection or timeout message."""
    monkeypatch.setattr(ui.httpx, "post", Mock(side_effect=error))
    with pytest.raises(ui.PredictionError, match=message):
        ui._request_prediction(image_bytes(), "image.png")


@pytest.mark.parametrize(
    ("status", "detail", "message"),
    [
        (400, "Invalid image", "Invalid image"),
        (413, "Image too large", "Image too large"),
        (415, "Unsupported format", "Unsupported format"),
        (422, [{"msg": "missing file"}], "valid PNG"),
        (503, "Model unavailable", "No trained model"),
        (500, "Inference failed", "HTTP 500"),
    ],
)
def test_ui_reports_api_errors(monkeypatch, status, detail, message):
    """Receive an HTTP error; expect a useful upload, model, or server message."""
    post = Mock(return_value=httpx.Response(status, json={"detail": detail}))
    monkeypatch.setattr(ui.httpx, "post", post)
    with pytest.raises(ui.PredictionError, match=message):
        ui._request_prediction(image_bytes(), "image.png")


@pytest.mark.parametrize("result", [{}, {"class_name": ""}, {"class_name": 3}, []])
def test_ui_rejects_missing_prediction(monkeypatch, result):
    """Receive malformed success JSON; expect an error rather than a false prediction."""
    monkeypatch.setattr(ui.httpx, "post", Mock(return_value=httpx.Response(200, json=result)))
    with pytest.raises(ui.PredictionError, match="did not contain a class name"):
        ui._request_prediction(image_bytes(), "image.png")


def test_ui_reports_unreadable_response(monkeypatch):
    """Receive non-JSON success content; expect a clear response error."""
    monkeypatch.setattr(ui.httpx, "post", Mock(return_value=httpx.Response(200, text="bad response")))
    with pytest.raises(ui.PredictionError, match="unreadable response"):
        ui._request_prediction(image_bytes(), "image.png")


def test_ui_initial_page_needs_no_api(monkeypatch):
    """Open the UI without an upload; expect instructions and no backend request."""
    post = Mock()
    monkeypatch.setattr(ui.httpx, "post", post)
    monkeypatch.setattr(ui, "load_dotenv", Mock())
    app = AppTest.from_file(str(Path(ui.__file__))).run()
    assert not app.exception
    assert app.title[0].value == "Fashion-MNIST Classifier"
    assert app.info[0].value == "Upload an image to get started."
    post.assert_not_called()


@pytest.mark.parametrize("outcome", ["success", "unavailable", "invalid"])
def test_ui_displays_prediction_or_error(monkeypatch, outcome):
    """Click Predict with an upload; expect a visible class or error, without live services."""
    screen = MagicMock()
    uploaded = Mock()
    uploaded.name = "image.png"
    uploaded.getvalue.return_value = b"broken" if outcome == "invalid" else image_bytes()
    screen.file_uploader.return_value = uploaded
    screen.button.return_value = True
    request = Mock(return_value="Bag")
    if outcome == "unavailable":
        request.side_effect = ui.PredictionError("No trained model is available.")
    monkeypatch.setattr(ui, "st", screen)
    monkeypatch.setattr(ui, "load_dotenv", Mock())
    monkeypatch.setattr(ui, "_request_prediction", request)
    ui.main()
    if outcome == "success":
        screen.image.assert_called_once()
        screen.success.assert_called_once_with("Predicted class: Bag")
        screen.error.assert_not_called()
    elif outcome == "unavailable":
        screen.error.assert_called_once_with("No trained model is available.")
        screen.success.assert_not_called()
    else:
        screen.error.assert_called_once()
        screen.image.assert_not_called()
        request.assert_not_called()
