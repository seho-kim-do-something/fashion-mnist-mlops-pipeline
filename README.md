# Fashion-MNIST MLOps Pipeline

Trains a LeNet-5-style classifier on Fashion-MNIST with Keras (TensorFlow backend),
tracks each run in MLflow, stores each trained model in MinIO, and sends a
training-complete webhook. Docker Compose runs MLflow, MinIO, and HTTPBin.

FastAPI serves predictions using the best stored model. Streamlit provides
an image upload and prediction page.

## Architecture and technologies

```text
Local training (train.toml + .env)
  -> TensorFlow/Keras LeNet-5 -> MLflow parameters, metrics, and artifacts
  -> MinIO model/<mlflow-run-id>/model.keras -> HTTPBin webhook

Browser -> Streamlit -> FastAPI -> prediction
                         |
                         +-- startup: select run in MLflow, download model from MinIO
```

Python, NumPy, TensorFlow/Keras, MLflow, MinIO, HTTPX, FastAPI, Pillow,
Streamlit, Docker Compose, and pytest are used. Training runs separately from
the prediction services. The webhook is a notification, not a model-reload trigger.

## Project structure

```text
configs/train.toml                  Training settings
src/fashion_mnist_mlops/config.py   Loads and validates settings
src/fashion_mnist_mlops/data.py     Downloads, prepares, and splits the data
src/fashion_mnist_mlops/model.py    Builds the LeNet-5 model
src/fashion_mnist_mlops/train.py    Trains the model and logs to MLflow
src/fashion_mnist_mlops/storage.py  Uploads the model to MinIO
src/fashion_mnist_mlops/webhook.py  Sends the training-complete webhook
src/fashion_mnist_mlops/prediction.py Selects, loads, and uses the best model
src/fashion_mnist_mlops/api.py      Health and prediction endpoints
src/fashion_mnist_mlops/ui.py       Streamlit image upload and prediction page
tests/test_training.py              Tests
tests/test_prediction.py            Model selection and prediction tests
tests/test_ui.py                    UI validation and API request tests
Dockerfile                          Shared FastAPI and Streamlit container image
docker-compose.yml                  Local services, FastAPI, and Streamlit
.env.example                        Service settings template
```

## Setup

Requires Python 3.11 or 3.12, Docker with Compose, and a running Docker engine.
Internet access is needed for dependencies, images, and the first dataset download.
Run all commands from this project directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp -n .env.example .env
```

In `.env`, set `MINIO_ACCESS_KEY` (3+ characters) and `MINIO_SECRET_KEY`
(8+ characters). Compose will not start without them. `cp -n` preserves an
existing `.env`. Do not commit `.env`.

The project uses the lightweight `mlflow-skinny` tracking client.
The MLflow server runs separately in Docker.

## Start the services

```bash
docker compose up -d mlflow minio httpbin
docker compose ps
```

| Service | Host address | Inside Docker network |
| --- | --- | --- |
| MLflow | http://localhost:5001 | http://mlflow:5000 |
| MinIO API | http://localhost:9000 | http://minio:9000 |
| MinIO console | http://localhost:9001 | http://minio:9001 |
| HTTPBin | http://localhost:8080 | http://httpbin:8080 |
| FastAPI | http://localhost:8000 | http://api:8000 |
| Streamlit | http://localhost:8501 | http://ui:8501 |

Data is kept in the `mlflow_data` and `minio_data` volumes.
MLflow uses host port 5001 because port 5000 is often taken on macOS.
MinIO uses a Bitnami legacy image that no longer receives updates;
use it for local development only.

Check that each service is running (each command should succeed):

```bash
curl --fail http://localhost:5001/health
curl --fail http://localhost:9000/minio/health/ready
curl --fail http://localhost:8080/get
```

Wait for all three checks to succeed, then run training. Only after training
succeeds, start FastAPI and Streamlit:

```bash
fashion-mnist-train --config configs/train.toml
docker compose up -d --build ui
```

After the first build, use `docker compose up -d` to start all existing services.
Use `--build` again when source files or dependencies change.
If the API was already running before training, run `docker compose restart api`
so it selects the best model again.

Stop the services with `docker compose down`.
Add `-v` to also delete all stored data.

## Configuration

Edit `configs/train.toml`:

| Setting | Meaning |
| --- | --- |
| `epochs` | Number of training passes |
| `batch_size` | Images per batch |
| `learning_rate` | Optimizer learning rate |
| `optimizer` | `adam`, `sgd`, or `rmsprop` |
| `loss_function` | `cross_entropy` |
| `random_seed` | Seed for repeatable results |
| `validation_split` | Fraction of data used for validation (e.g. `0.2`) |

These settings belong in the `[training]` section.
`loss_function` also accepts `sparse_categorical_crossentropy`; both names use
sparse categorical cross-entropy. Optimizer-specific options use Keras defaults;
no scheduler or early stopping is configured.
`[data].directory` controls the dataset cache and `[output].directory` the run
folders. Relative paths are resolved from the directory where you run training.
Invalid, missing, or unknown configuration values are rejected.

Service settings belong in `.env`, not `train.toml`:

| Variable | Default / purpose |
| --- | --- |
| `WEBHOOK_URL` | `http://localhost:8080/post` |
| `API_BASE_URL` | `http://localhost:8000` for a locally run UI |
| `MLFLOW_PORT` | `5001` |
| `MINIO_API_PORT`, `MINIO_CONSOLE_PORT` | `9000`, `9001` |
| `HTTPBIN_PORT`, `API_PORT`, `STREAMLIT_PORT` | `8080`, `8000`, `8501` |

MLflow and MinIO connection settings are listed below. If you change a host
port, also update the matching local URL in `.env`. Inside Compose, services
use service names and internal ports; host port changes do not affect those URLs.
`MODEL_OBJECT_KEY` is not needed: the API reads the object key from the selected
MLflow run.

## Train

```bash
fashion-mnist-train --config configs/train.toml
```

Run it from the project directory; settings in `.env` are loaded automatically.
Variables already set in your shell take priority over `.env`.

The first run downloads Fashion-MNIST into `data/fashion-mnist/`.
If macOS shows `CERTIFICATE_VERIFY_FAILED`, run this first:

```bash
export SSL_CERT_FILE="$(python -m certifi)"
```

With the default settings, 48,000 images are used for training and 12,000 for
validation. Each run saves its files to `artifacts/<timestamp>-<id>/`:
`config.json`, `history.json`, `metrics.json`, and `model.keras`.
`config.json` records the actual settings, library versions, dataset sizes,
and optimizer/loss details. Seeds and deterministic TensorFlow operations
support reproducibility; exact results can still vary across environments.
The separate Fashion-MNIST test set is not used. No minimum accuracy is required.

## MLflow tracking

| Variable | Default |
| --- | --- |
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` |
| `MLFLOW_EXPERIMENT_NAME` | `fashion-mnist` |

Each run logs:

- **Parameters:** all training settings and sample counts
- **Metrics per epoch:** `loss`, `accuracy`, `val_loss`, `val_accuracy`
- **Final metrics:** `training_loss`, `validation_loss`, `validation_accuracy`
- **Artifacts:** the four run files listed above
- **Tag `minio_model_uri`:** where the model is stored in MinIO

To view results, open http://localhost:5001, select `fashion-mnist`, and open a run.
Failed runs are marked `FAILED`.

## MinIO model storage

After training, the model is uploaded to MinIO:

```text
fashion-minst-bucket/model/<mlflow-run-id>/model.keras
```

The bucket is created automatically if it does not exist. Using the MLflow
run ID in the path means each run keeps its own model.

| Variable | Default |
| --- | --- |
| `MINIO_ENDPOINT` | `localhost:9000` |
| `MINIO_ACCESS_KEY` | required |
| `MINIO_SECRET_KEY` | required |
| `MINIO_SECURE` | `false` |
| `MINIO_BUCKET` | `fashion-minst-bucket` |
| `MINIO_MODEL_PREFIX` | `model/` |

Keep `MINIO_BUCKET=fashion-minst-bucket` and `MINIO_MODEL_PREFIX=model/` for the
required storage layout. The spelling `minst` is intentional. Object prefixes
are created by uploading an object; no separate folder creation is needed.
MLflow artifacts use its own volume, not the MinIO bucket.

To view the model, sign in to http://localhost:9001 with the MinIO credentials
from `.env`, open **Object Browser**, then `fashion-minst-bucket` → `model`.

If the upload fails (for example, MinIO is not running or the credentials are
wrong), training stops with a `Model upload failed: ...` message and the
MLflow run is marked `FAILED`.

## Training-complete webhook

After training, MLflow logging, and the MinIO upload all succeed, training sends
an HTTP POST to `WEBHOOK_URL` (default `http://localhost:8080/post`, the local
HTTPBin service) with this JSON:

```json
{
  "status": "completed",
  "mlflow_run_id": "<run-id>",
  "minio_bucket": "fashion-minst-bucket",
  "minio_object_key": "model/<run-id>/model.keras",
  "validation_accuracy": 0.87
}
```

No webhook is sent if training, MLflow logging, or the upload fails.
If the webhook cannot be reached or returns a non-2xx status, the command ends
with `Training completed, but the webhook failed: ...`. The MLflow run and the
uploaded model are kept.

Check the training notification:

```bash
docker compose logs --tail=20 httpbin
```

Expected: `200 POST /post`. Training also prints `Webhook sent ... (HTTP 200)`
with the JSON payload. HTTPBin echoes requests but does not keep a browsable
history of their payloads.

For a manual receiver check (not a training notification):

```bash
curl --fail -X POST http://localhost:8080/post \
  -H "Content-Type: application/json" \
  -d '{"status":"test"}'
```

Expected: JSON with the submitted value under `json`.

## Tests

```bash
python -m pytest -q
```

Tests use small fake data, temporary files, and fake service clients,
so no download or running service is needed.
They cover configuration and `.env` precedence, training reproducibility,
MLflow logging, versioned uploads, webhook success/failure ordering, model
selection, API image validation and responses, and UI results and errors.
These tests do not replace the real-service checks below.

## FastAPI prediction

Train at least one model first, then build and start the API:

```bash
docker compose up -d --build api
docker compose logs --tail=50 api
```

The first build installs TensorFlow and the project dependencies, so it can take
several minutes. Compose also starts MLflow and MinIO if needed.
`API_PORT` in `.env` controls the host port (default `8000`).
Inside Docker, the API uses `http://mlflow:5000` and `minio:9000`;
MinIO credentials come from `.env`, not the image.

At startup, the API searches the configured `MLFLOW_EXPERIMENT_NAME` for
`FINISHED` runs with final `validation_accuracy` and a `minio_model_uri` tag.
It selects the highest accuracy; ties use the most recent completion time.
The bucket and object key come from that run's tag, not a fixed object key.
Logs show the selected run ID, accuracy, MinIO location, and whether loading succeeded.

The model is loaded only once. After a new training run or a startup failure,
run `docker compose restart api` to select and load a model again.
If the selected model is missing or cannot be loaded, the API reports the error
instead of silently choosing a different model.

Check that the API is running:

```bash
curl --fail http://localhost:8000/health
```

Expected: `{"status":"ok"}`, even if no model is available.

Send one image (replace the path with your file):

```bash
curl --fail-with-body -X POST http://localhost:8000/predict \
  -F "file=@/absolute/path/to/image.png"
```

Expected: `{"class_name":"Sneaker"}` or another Fashion-MNIST class name.
To test in your browser using Swagger UI (FastAPI's interactive API documentation):

1. Open http://localhost:8000/docs.
2. Expand **POST /predict** and click **Try it out**.
3. Click **Choose File**, select an image, then click **Execute**.
4. Check **Response body** for the predicted class name and **200** for success.

Use a clothing image with a dark background, like the training images.
The API converts it to grayscale, resizes it to 28x28, and uses the training
normalization (float32 values from 0 to 1).

Supported files are single-frame PNG, JPEG, and BMP images up to 5 MiB.
Invalid or empty images return HTTP `400`, unsupported formats `415`,
oversized uploads `413`, and a missing file field `422`.
If no model was loaded, `/predict` returns `503` with an explanation;
`/health` still returns `200`. Check `docker compose logs api` for details.

## Streamlit interface

Start the UI and its backend services:

```bash
docker compose up -d --build ui
```

Open http://localhost:8501:

1. Upload one PNG, JPEG, or BMP image (up to 5 MiB).
2. Check the image preview, then click **Predict class**.
3. The predicted class appears below the button.

The UI sends the image to FastAPI; it does not load or run a model.
The current API returns only `class_name`, so confidence scores are not shown.
Invalid uploads, connection failures, and unavailable models display helpful messages.
After training a model when none was loaded, run `docker compose restart api`.

`STREAMLIT_PORT` in `.env` controls the browser port (default `8501`).
Compose sets `API_BASE_URL=http://api:8000` for container-to-container requests.
For local use outside Docker, the UI reads `API_BASE_URL` from `.env`
(default `http://localhost:8000`). Existing environment variables take priority.

Check logs or stop only the UI:

```bash
docker compose logs --tail=30 ui api
docker compose stop ui
```

Run the UI tests without starting Docker:

```bash
python -m pytest -q tests/test_ui.py
```

Expected: all UI tests pass. To check connection-error messages manually,
keep the UI open, run `docker compose stop api`, and click **Predict class**.
The UI should show a connection error. Restore the backend with
`docker compose start api`.

## Final end-to-end check

With `.venv` active, `.env` configured, and the Docker engine running:

```bash
python -m pytest -q
docker compose up -d mlflow minio httpbin
curl --fail http://localhost:5001/health
curl --fail http://localhost:9000/minio/health/ready
curl --fail http://localhost:8080/get
fashion-mnist-train --config configs/train.toml
docker compose logs --tail=20 httpbin
docker compose up -d --build ui
docker compose restart api
docker compose logs --tail=50 api
curl --fail http://localhost:8000/health
```

Allow services to start before the HTTP checks; retry if they are still starting.
The API restart ensures it selects again if a container from an earlier run
was already running. Wait for `ready for prediction` before uploading an image.

| Acceptance check | Expected result |
| --- | --- |
| Configure and train without source edits | Settings come from `configs/train.toml` and `.env` |
| Record hyperparameters and validation metrics | New run in http://localhost:5001, experiment `fashion-mnist` |
| Upload a versioned model | http://localhost:9001 shows `fashion-minst-bucket/model/<run-id>/model.keras` |
| Notify after successful upload and logging | HTTPBin logs `200 POST /post`; printed JSON matches that run |
| Start prediction services with Compose | API logs the selected run and successful model loading |
| Upload and see a prediction | http://localhost:8501 displays the class after **Predict class** |
| Document setup without embedded secrets | `.env.example` contains blank credentials; `.env` stays local |

Check that the run's `minio_model_uri` matches its MinIO object.
Use the same image in Swagger UI and Streamlit; both should return the same class.
For an invalid-input check with a loaded model:

```bash
curl -i -X POST http://localhost:8000/predict -F "file=@README.md"
```

Expected: HTTP `400`, with a readable image-validation error.

## Troubleshooting and limitations

| Problem | What to check |
| --- | --- |
| Missing MinIO credentials | Set both keys in `.env`; password must have at least 8 characters |
| Connection refused during training | Wait for MLflow, MinIO, and HTTPBin checks to succeed |
| API returns 503 | Read `docker compose logs api`; confirm a FINISHED run has a metric and MinIO tag, then restart `api` |
| UI cannot reach API | In Docker use `http://api:8000`; locally use `http://localhost:8000` |
| Root URL returns 404 | Use `/health`, `/predict`, or `/docs`; there is no `/` page |
| `curl: (26)` | Replace the example image path with a real local file |
| Build downloads packages again | Source/README changes invalidate the install layer; unchanged starts do not need `--build` |
| Inaccurate predictions | Use one Fashion-MNIST-style item on a dark background; ordinary photos may not classify well |

`/health` checks the server, not the model. Docker marks failed health checks
as unhealthy but does not automatically restart the service.
Selection happens only at API startup, with no fallback if the best model is
missing. Restart the API after resolving storage errors or training a new model.
Confidence scores, automatic reloading, and webhook retries are not implemented
and are not required. A failed webhook leaves the trained run FINISHED.

Dependencies use version ranges rather than a lock file. The MinIO legacy image
is suitable only for this isolated local setup, not production. Generated data,
models, virtual environments, and credentials are excluded by `.gitignore`;
Docker build exclusions are in `.dockerignore`. Keep MLflow/MinIO volumes for
the live experiment and model deliverables; `docker compose down -v` deletes them.
