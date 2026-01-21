from LiverTumorSegmentation.entity.config_entity import EvaluationConfig
from LiverTumorSegmentation.components.training import (
    PKLSegmentationDataset,
    DatasetBuilder,
    SegmentationLoss,
    CombinedLossMetric,
    DiceMetric,
    IoUMetric,
)
from LiverTumorSegmentation.utils.common import save_json
from pathlib import Path
from typing import Optional
import tensorflow as tf
import mlflow
from urllib.parse import urlparse


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
                "DiceMetric": DiceMetric,
                "IoUMetric": IoUMetric,
                "combined_loss": SegmentationLoss.combined_loss,
                "dice_loss": SegmentationLoss.dice_loss,
            },
            compile=False,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=SegmentationLoss.combined_loss,
            metrics=[
                CombinedLossMetric(name="combined_loss"),
                DiceMetric(name="dice"),
                IoUMetric(name="iou"),
            ],
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
            "dice": float(self.score[2]) if len(self.score) > 2 else None,
            "iou": float(self.score[3]) if len(self.score) > 3 else None,
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
        # score format: [loss, combined_loss_metric, dice, iou]
        metrics = {
            "val_loss": float(self.score[0]),
            "val_combined_loss": float(self.score[1]) if len(self.score) > 1 else float(self.score[0]),
            "val_dice": float(self.score[2]) if len(self.score) > 2 else None,
            "val_iou": float(self.score[3]) if len(self.score) > 3 else None,
        }
        metrics = {k: v for k, v in metrics.items() if v is not None}

        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            # Log the model
            if self.model is not None:
                # Model registry does not work with file store
                # Try to register for non-file stores, but fall back to logging without registration if it fails
                
                """
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

                """
                mlflow.keras.log_model(self.model, "model")

        print(f"Evaluation results logged to MLflow: {mlflow.get_tracking_uri()}")