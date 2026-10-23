"""Send a notification when training completes."""

import json
import os

import httpx


class WebhookError(RuntimeError):
    """Raised when the training-complete notification cannot be delivered."""


def send_training_complete(payload: dict) -> int:
    """POST the payload as JSON to WEBHOOK_URL and return the 2xx status code."""
    url = os.environ.get("WEBHOOK_URL", "http://localhost:8080/post")
    try:
        response = httpx.post(url, json=payload, timeout=10)
    # Connection refused, DNS errors, and timeouts end up here.
    except (httpx.RequestError, httpx.InvalidURL) as error:
        raise WebhookError(f"Could not reach webhook at {url}: {error}") from error
    if not response.is_success:
        raise WebhookError(
            f"Webhook at {url} returned HTTP {response.status_code}: {response.text[:200]}"
        )
    print(f"Webhook sent to {url} (HTTP {response.status_code}): {json.dumps(payload)}")
    return response.status_code
