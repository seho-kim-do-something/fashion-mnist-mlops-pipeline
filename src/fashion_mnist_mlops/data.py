"""Fashion-MNIST loading and preprocessing."""

import gzip
from pathlib import Path

import numpy as np
import tensorflow as tf


def preprocess_images(images: np.ndarray) -> np.ndarray:
    """Normalize pixel values to 0-1 and reshape images from (28, 28) to (28, 28, 1)."""
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError("Expected images with shape (samples, 28, 28).")
    return images.astype("float32")[..., np.newaxis] / 255.0


def load_training_data(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """Download or reuse cached Fashion-MNIST training images and integer labels."""
    directory.mkdir(parents=True, exist_ok=True)
    origin = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    filenames = (
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    )
    paths = [
        tf.keras.utils.get_file(
            fname=filename,
            origin=origin + filename,
            cache_dir=str(directory),
            cache_subdir="fashion-mnist",
        )
        for filename in filenames
    ]
    # Skip the file headers and read the pixels as 28x28 images and the labels as class IDs.
    with gzip.open(paths[0], "rb") as file:
        images = np.frombuffer(file.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(paths[1], "rb") as file:
        labels = np.frombuffer(file.read(), dtype=np.uint8, offset=8)
    # Check that both files contain the expected 60,000 samples.
    if len(images) != 60_000 or len(labels) != 60_000:
        raise ValueError("Expected 60,000 Fashion-MNIST training images and labels.")
    return images, labels


def split_data(
    images: np.ndarray,
    labels: np.ndarray,
    validation_split: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split images and labels into training and validation sets using validation_split."""
    if len(images) != len(labels):
        raise ValueError("Image and label counts must match.")
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between zero and one.")
    # validation_split comes from train.toml; 0.2 reserves 20% for validation and 80% for training.
    validation_size = int(len(images) * validation_split)
    if not 0 < validation_size < len(images):
        raise ValueError("The validation split must leave at least one sample in each subset.")
    # Use the same seeded ordering for images and labels to keep pairs aligned.
    indices = np.random.default_rng(random_seed).permutation(len(images))
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]
    return (
        preprocess_images(images[training_indices]),
        labels[training_indices],
        preprocess_images(images[validation_indices]),
        labels[validation_indices],
    )
