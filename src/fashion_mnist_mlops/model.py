"""LeNet-5-style classifier for 28x28 grayscale images."""

import tensorflow as tf

from fashion_mnist_mlops.config import TrainingConfig


def build_model(config: TrainingConfig) -> tf.keras.Model:
    """Build and compile a LeNet-5-style CNN for ten-class grayscale classification."""
    if tf.keras.backend.backend() != "tensorflow":
        raise RuntimeError("The Keras backend must be TensorFlow. Set KERAS_BACKEND=tensorflow.")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(6, kernel_size=5, activation="tanh"),
            tf.keras.layers.AveragePooling2D(pool_size=2),
            tf.keras.layers.Conv2D(16, kernel_size=5, activation="tanh"),
            tf.keras.layers.AveragePooling2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation="tanh"),
            tf.keras.layers.Dense(84, activation="tanh"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ],
        name="lenet5",
    )
    optimizers = {
        "adam": tf.keras.optimizers.Adam,
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop,
    }
    # Fashion-MNIST labels are integer class IDs, not one-hot vectors.
    model.compile(
        optimizer=optimizers[config.optimizer](learning_rate=config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model
