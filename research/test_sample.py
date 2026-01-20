from dataclasses import dataclass

from pathlib import Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: str
    updated_base_model_path: str
    copy_updated_base_model_path: str
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int
    params_batch_size: int
    params_epochs: int
    # Optional base number of filters for the first UNet block.
    params_base_filters: int = 32

from LiverTumorSegmentation.constants import *
from LiverTumorSegmentation.utils.common import read_yaml, create_directories

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            copy_updated_base_model_path=Path(config.copy_updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            params_batch_size=self.params.BATCH_SIZE,
            params_epochs=self.params.EPOCHS
        )

        return prepare_base_model_config


import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import shutil
from typing import Tuple, Optional


class UNetBuilder:
    """
    Small UNet builder (2 down / 2 up) for 2D segmentation.
    Keeps model construction in one place instead of free functions.
    """

    def __init__(self, base_filters: int = 32):
        self.base_filters = int(base_filters)

    @staticmethod
    def conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    def build(self, input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=input_shape)

        # Encoder
        e1 = self.conv_block(inputs, self.base_filters)
        p1 = tf.keras.layers.MaxPool2D()(e1)

        e2 = self.conv_block(p1, self.base_filters * 2)
        p2 = tf.keras.layers.MaxPool2D()(e2)

        b = self.conv_block(p2, self.base_filters * 4)

        # Decoder
        u1 = tf.keras.layers.Conv2DTranspose(self.base_filters * 2, 2, strides=2, padding="same")(b)
        u1 = tf.keras.layers.Concatenate()([u1, e2])
        d1 = self.conv_block(u1, self.base_filters * 2)

        u2 = tf.keras.layers.Conv2DTranspose(self.base_filters, 2, strides=2, padding="same")(d1)
        u2 = tf.keras.layers.Concatenate()([u2, e1])
        d2 = self.conv_block(u2, self.base_filters)

        # Output layer (logits)
        if int(num_classes) == 1:
            outputs = tf.keras.layers.Conv2D(1, 1, activation=None, name="logits")(d2)
        else:
            outputs = tf.keras.layers.Conv2D(int(num_classes), 1, activation=None, name="logits")(d2)

        return tf.keras.Model(inputs, outputs, name="unet_base")


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model: Optional[tf.keras.Model] = None

    def _prepare_unet_model(self) -> tf.keras.Model:
        """Build and compile a UNet model from scratch."""
        builder = UNetBuilder(base_filters=self.config.params_base_filters)
        model = builder.build(
            input_shape=tuple(self.config.params_image_size),
            num_classes=self.config.params_classes,
        )

        if self.config.params_classes == 1:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            metrics = [tf.keras.metrics.MeanIoU(num_classes=2, name="miou")]
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            metrics = [tf.keras.metrics.MeanIoU(num_classes=self.config.params_classes, name="miou")]

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
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

    def copy_model(self) -> None:
        """
        Copy the updated base model file to the `copy_updated_base_model_path`
        directory defined in the configuration.
        """
        src = Path(self.config.updated_base_model_path)
        dst_dir = Path(self.config.copy_updated_base_model_path)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        shutil.copy2(src, dst)


try:
    config = ConfigurationManager()
    prepare_base_model_config = config.get_prepare_base_model_config()
    preparer = PrepareBaseModel(config=prepare_base_model_config)
    model = preparer.update_base_model()
    # Save the updated base model to the configured updated_base_model_path
    PrepareBaseModel.save_model(prepare_base_model_config.updated_base_model_path, model)
    # Also copy the updated model into the shared data artifacts directory
    preparer.copy_model()
except Exception as e:
    raise e
    