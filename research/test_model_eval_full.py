import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# Silence TensorFlow INFO logs in terminal output.
# Must be set BEFORE importing tensorflow (directly or indirectly).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
import mlflow

from LiverTumorSegmentation.config.configuration import ConfigurationManager
from LiverTumorSegmentation.components.training import (
    PKLSegmentationDataset,
    DatasetBuilder,
    SegmentationLoss,
    CombinedLossMetric,
)
from LiverTumorSegmentation.utils.common import save_json


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


class SegmentationEvaluator:
    """Evaluation pipeline for the pickle-based segmentation dataset."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model: Optional[tf.keras.Model] = None
        self.val_dataset: Optional[tf.data.Dataset] = None
        self.score = None

    def load_model(self) -> tf.keras.Model:
        """Load `.h5` model safely and recompile with our custom loss/metric."""
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.config.model_path}")

        model = tf.keras.models.load_model(
            self.config.model_path,
            custom_objects={
                "SegmentationLoss": SegmentationLoss,
                "CombinedLossMetric": CombinedLossMetric,
                "combined_loss": SegmentationLoss.combined_loss,
                "dice_loss": SegmentationLoss.dice_loss,
            },
            compile=False,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=SegmentationLoss.combined_loss,
            metrics=[CombinedLossMetric(name="combined_loss")],
        )

        self.model = model
        return model

    def build_val_dataset(self) -> tf.data.Dataset:
        """Build validation dataset from PKL ground-truth + predictions."""
        if not self.config.training_data.exists():
            raise FileNotFoundError(f"Training data directory not found: {self.config.training_data}")

        pred_dir = self.config.training_data / "Predictions" / self.config.predictions_subdir
        gt_dir = self.config.training_data / "GroundTruth"

        if not pred_dir.exists():
            raise FileNotFoundError(f"Predictions directory not found: {pred_dir}")
        if not gt_dir.exists():
            raise FileNotFoundError(f"GroundTruth directory not found: {gt_dir}")

        dataset = PKLSegmentationDataset(pred_dir, gt_dir)
        train_idx, val_idx = DatasetBuilder.split_indices(len(dataset), self.config.val_split)

        target_h, target_w = self.config.image_size[0], self.config.image_size[1]
        in_channels = self.config.image_size[2]

        _, val_dataset = DatasetBuilder.build_datasets(
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            target_h=target_h,
            target_w=target_w,
            in_channels=in_channels,
            batch_size=self.config.batch_size,
            shuffle_buffer=1,  # no need to shuffle val
        )

        self.val_dataset = val_dataset
        return val_dataset

    def evaluate(self):
        if self.model is None:
            self.load_model()
        if self.val_dataset is None:
            self.build_val_dataset()

        self.score = self.model.evaluate(self.val_dataset, verbose=1)
        return self.score

    def save_score(self):
        """Save evaluation scores to a JSON file."""
        if self.score is None:
            raise ValueError("Model must be evaluated before saving scores. Call evaluate() first.")
        
        scores = {
            "loss": float(self.score[0]),
            "combined_loss": float(self.score[1]) if len(self.score) > 1 else float(self.score[0]),
        }
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        """Log evaluation results to MLflow."""
        if self.config.mlflow_uri is None:
            print("MLflow URI not configured. Skipping MLflow logging.")
            return

        if self.score is None:
            raise ValueError("Model must be evaluated before logging to MLflow. Call evaluate() first.")

        # Set the tracking URI (not registry URI) - this is what tells MLflow where to log runs
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Prepare parameters to log
        params = {
            "image_size": str(self.config.image_size),
            "batch_size": self.config.batch_size,
            "predictions_subdir": self.config.predictions_subdir,
            "val_split": self.config.val_split,
            "model_path": str(self.config.model_path),
        }

        # Prepare metrics to log
        # score format: [loss, combined_loss_metric]
        metrics = {
            "val_loss": float(self.score[0]),
            "val_combined_loss": float(self.score[1]) if len(self.score) > 1 else float(self.score[0]),
        }

        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            # Log the model
            if self.model is not None:
                # Model registry does not work with file store
                # Try to register for non-file stores, but fall back to logging without registration if it fails
                if tracking_url_type_store != "file":
                    try:
                        # Attempt to register the model
                        # There are other ways to use the Model Registry, which depends on the use case,
                        # please refer to the doc for more information:
                        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                        mlflow.keras.log_model(
                            self.model, "model", registered_model_name="SegmentationModel"
                        )
                    except mlflow.exceptions.MlflowException as e:
                        # If registration fails (e.g., 403 Forbidden), log without registration
                        print(f"Warning: Model registration failed ({e}). Logging model without registration.")
                        mlflow.keras.log_model(self.model, "model")
                else:
                    # Log model without registration for file store
                    mlflow.keras.log_model(self.model, "model")

        print(f"Evaluation results logged to MLflow: {mlflow.get_tracking_uri()}")


def get_default_eval_config() -> EvaluationConfig:
    """Load evaluation config using the project ConfigurationManager."""
    cm = ConfigurationManager()
    train_cfg = cm.get_training_config()

    # Prefer explicit evaluation section if present, else fall back to training config.
    eval_section = getattr(cm.config, "evaluation", None)
    model_path = Path(getattr(eval_section, "path_of_model", train_cfg.trained_model_path)) if eval_section else train_cfg.trained_model_path
    training_data = Path(getattr(eval_section, "training_data", train_cfg.training_data)) if eval_section else train_cfg.training_data
    mlflow_uri = getattr(eval_section, "mlflow_uri", None) if eval_section else None

    # Optional overrides from params if provided (safe defaults).
    predictions_subdir = getattr(cm.params, "PREDICTIONS_SUBDIR", "Dice")
    val_split = float(getattr(cm.params, "VAL_SPLIT", 0.1))

    return EvaluationConfig(
        model_path=model_path,
        training_data=training_data,
        image_size=cm.params.IMAGE_SIZE,
        batch_size=cm.params.BATCH_SIZE,
        predictions_subdir=predictions_subdir,
        val_split=val_split,
        mlflow_uri=mlflow_uri,
    )


if __name__ == "__main__":
    cfg = get_default_eval_config()
    evaluator = SegmentationEvaluator(cfg)
    evaluator.evaluate()
    evaluator.save_score()
    evaluator.log_into_mlflow()