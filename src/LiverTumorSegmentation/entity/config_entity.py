from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    copy_data: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


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
    params_base_filters: int


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration dataclass for training parameters."""
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    copy_trained_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list

@dataclass(frozen=True)
class EvaluationConfig:
    """Config for evaluating a trained segmentation model."""

    model_path: Path
    training_data: Path
    image_size: list  # [H, W, C]
    batch_size: int
    predictions_subdir: str = "Dice"
    val_split: float = 0.1
    mlflow_uri: Optional[str] = None