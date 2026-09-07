"""Classify one local image with a trained Keras model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


IMAGE_SIZE = (180, 180)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/cats_dogs_model.keras"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model(args.model)
    image = tf.keras.utils.load_img(args.image, target_size=IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(image)
    image_batch = np.expand_dims(image_array, axis=0)

    dog_probability = float(model.predict(image_batch, verbose=0)[0][0])
    label = "dog" if dog_probability >= 0.5 else "cat"
    confidence = dog_probability if label == "dog" else 1.0 - dog_probability

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Raw sigmoid output (dog probability): {dog_probability:.6f}")


if __name__ == "__main__":
    main()
