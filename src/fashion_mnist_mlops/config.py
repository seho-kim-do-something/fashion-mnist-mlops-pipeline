"""Training configuration."""

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training settings and resolved data and output paths."""

    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    loss_function: str
    random_seed: int
    validation_split: float
    data_directory: Path
    output_directory: Path


def load_config(path: str | Path) -> TrainingConfig:
    """Load TOML settings, validate their values, and resolve directory paths."""
    with Path(path).open("rb") as file:
        settings = tomllib.load(file)

    # Reject missing or unknown keys so configuration typos do not go unnoticed.
    expected = {
        "training": {
            "epochs", "batch_size", "learning_rate", "optimizer",
            "loss_function", "random_seed", "validation_split",
        },
        "data": {"directory"},
        "output": {"directory"},
    }
    if set(settings) != set(expected):
        raise ValueError("Configuration must contain only [training], [data], and [output].")
    for section, keys in expected.items():
        if not isinstance(settings[section], dict) or set(settings[section]) != keys:
            raise ValueError(f"[{section}] must contain exactly: {', '.join(sorted(keys))}.")

    training = settings["training"]
    for key in ("epochs", "batch_size", "random_seed"):
        value = training[key]
        minimum = 0 if key == "random_seed" else 1
        # Exact type checks reject TOML booleans, which Python treats as integers.
        if type(value) is not int or value < minimum:
            raise ValueError(f"training.{key} must be an integer >= {minimum}.")
    if training["random_seed"] >= 2**32:
        raise ValueError("training.random_seed must be less than 2**32.")

    for key in ("learning_rate", "validation_split"):
        value = training[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"training.{key} must be a finite number.")
    if training["learning_rate"] <= 0:
        raise ValueError("training.learning_rate must be greater than zero.")
    if not 0 < training["validation_split"] < 1:
        raise ValueError("training.validation_split must be between zero and one.")

    if training["optimizer"] not in ("adam", "sgd", "rmsprop"):
        raise ValueError("training.optimizer must be adam, sgd, or rmsprop.")
    if training["loss_function"] not in ("cross_entropy", "sparse_categorical_crossentropy"):
        raise ValueError(
            "training.loss_function must be cross_entropy or sparse_categorical_crossentropy."
        )

    # Resolve relative paths from the current working directory.
    directories = {}
    for section in ("data", "output"):
        value = settings[section]["directory"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{section}.directory must be a non-empty path.")
        directories[section] = Path(value).expanduser().resolve()

    return TrainingConfig(
        **training,
        data_directory=directories["data"],
        output_directory=directories["output"],
    )
