"""
Sample script to prepare a base UNet model for 2D image segmentation.

This mirrors the idea of `PrepareBaseModel` in `02_base_model.ipynb`, but as a
standalone Python module. It reuses the UNet architecture style from
`artifacts/sample/train_sample.py` and reads core hyperparameters from
`configs/params.yaml`.

Usage (from project root):

    python -m research.sample_conf

This will:
  - load IMAGE_SIZE, CLASSES, LEARNING_RATE from `configs/params.yaml`
  - build a small UNet with that input shape and number of classes
  - compile it with BinaryCrossentropy (for CLASSES == 1) or
    CategoricalCrossentropy (for CLASSES > 1)
  - save the model to `artifacts/base_model/unet_base.keras`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import tensorflow as tf
import yaml


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PrepareBaseModelConfig:
    image_size: Tuple[int, int, int]
    classes: int
    learning_rate: float
    base_filters: int = 32
    model_dir: Path = Path("artifacts") / "base_model"
    model_name: str = "unet_base.keras"

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_name


def load_params(path: Path = Path("configs") / "params.yaml") -> PrepareBaseModelConfig:
    with path.open() as f:
        params = yaml.safe_load(f)
    image_size = tuple(params.get("IMAGE_SIZE", [128, 128, 1]))
    classes = int(params.get("CLASSES", 1))
    lr = float(params.get("LEARNING_RATE", 1e-3))
    return PrepareBaseModelConfig(
        image_size=image_size,
        classes=classes,
        learning_rate=lr,
    )


# ---------------------------------------------------------------------------
# UNet definition (adapted from artifacts/sample/train_sample.py)
# ---------------------------------------------------------------------------


def conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def build_unet(input_shape: Tuple[int, int, int], num_classes: int, base_filters: int = 32) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    e1 = conv_block(inputs, base_filters)
    p1 = tf.keras.layers.MaxPool2D()(e1)

    e2 = conv_block(p1, base_filters * 2)
    p2 = tf.keras.layers.MaxPool2D()(e2)

    b = conv_block(p2, base_filters * 4)

    # Decoder
    u1 = tf.keras.layers.Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same")(b)
    u1 = tf.keras.layers.Concatenate()([u1, e2])
    d1 = conv_block(u1, base_filters * 2)

    u2 = tf.keras.layers.Conv2DTranspose(base_filters, 2, strides=2, padding="same")(d1)
    u2 = tf.keras.layers.Concatenate()([u2, e1])
    d2 = conv_block(u2, base_filters)

    # Output layer
    if num_classes == 1:
        # Binary segmentation: single-channel logits
        outputs = tf.keras.layers.Conv2D(1, 1, activation=None, name="logits")(d2)
    else:
        # Multi-class segmentation: per-pixel logits for each class
        outputs = tf.keras.layers.Conv2D(num_classes, 1, activation=None, name="logits")(d2)

    return tf.keras.Model(inputs, outputs, name="unet_base")


# ---------------------------------------------------------------------------
# PrepareBaseModel wrapper (similar to 02_base_model.ipynb)
# ---------------------------------------------------------------------------


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model: Optional[tf.keras.Model] = None

    def _prepare_unet_model(self) -> tf.keras.Model:
        """Build and compile a UNet model from scratch."""
        model = build_unet(
            input_shape=self.config.image_size,
            num_classes=self.config.classes,
            base_filters=self.config.base_filters,
        )

        if self.config.classes == 1:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            metrics = [tf.keras.metrics.MeanIoU(num_classes=2, name="miou")]
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            metrics = [tf.keras.metrics.MeanIoU(num_classes=self.config.classes, name="miou")]

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=loss,
            metrics=metrics,
        )
        return model

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)

    def update_base_model(self) -> tf.keras.Model:
        """
        Build and store the compiled UNet model according to the config,
        then return it.
        """
        self.model = self._prepare_unet_model()
        return self.model


def main() -> None:
    cfg = load_params()
    preparer = PrepareBaseModel(cfg)
    model = preparer.update_base_model()
    PrepareBaseModel.save_model(cfg.model_path, model)
    print(f"[sample_conf] Saved UNet base model to: {cfg.model_path}")


if __name__ == "__main__":
    main()


