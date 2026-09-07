"""Train the original learning-project CNN on a local cats-vs-dogs split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf


IMAGE_SIZE = (180, 180)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_SEED = 42


def build_model() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(*IMAGE_SIZE, 3)),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="cats_vs_dogs_cnn",
    )


def load_datasets(data_root: Path, batch_size: int, seed: int):
    train_directory = data_root / "train"
    validation_directory = data_root / "val"

    train_data = tf.keras.utils.image_dataset_from_directory(
        train_directory,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )
    validation_data = tf.keras.utils.image_dataset_from_directory(
        validation_directory,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    expected_classes = ["cats", "dogs"]
    if train_data.class_names != expected_classes:
        raise ValueError(
            f"Expected class directories {expected_classes}, found {train_data.class_names}"
        )

    autotune = tf.data.AUTOTUNE
    train_data = (
        train_data.cache()
        .shuffle(1000, seed=seed, reshuffle_each_iteration=True)
        .prefetch(buffer_size=autotune)
    )
    validation_data = validation_data.cache().prefetch(buffer_size=autotune)
    return train_data, validation_data


def save_training_curves(history: tf.keras.callbacks.History, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Binary cross-entropy")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    train_data, validation_data = load_datasets(args.data, args.batch_size, args.seed)
    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=args.epochs,
    )
    validation_metrics = model.evaluate(validation_data, verbose=0, return_dict=True)

    model.save(args.output / "cats_dogs_model.keras")
    save_training_curves(history, args.output / "training_curves.png")
    metrics = {
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "validation": {key: float(value) for key, value in validation_metrics.items()},
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": list(IMAGE_SIZE),
        "seed": args.seed,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
