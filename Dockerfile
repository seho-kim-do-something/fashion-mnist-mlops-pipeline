FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KERAS_BACKEND=tensorflow

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "fashion_mnist_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
