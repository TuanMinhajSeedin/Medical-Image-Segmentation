import os
from pathlib import Path

# Reduce TensorFlow console noise (must be set before importing tensorflow).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

from LiverTumorSegmentation.components.training import CombinedLossMetric, SegmentationLoss


def load_trained_model_h5(model_path: Path | None = None) -> tf.keras.Model:
    """Load the trained model from a legacy `.h5` file and recompile it."""
    project_root = Path(__file__).resolve().parent.parent
    model_path = model_path or (project_root / "artifacts" / "training" / "model.h5")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            f"Run training first, e.g. from project root:\n"
            f"  python train.py --config configs/exp_001.yaml"
        )

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "SegmentationLoss": SegmentationLoss,
            "CombinedLossMetric": CombinedLossMetric,
            "combined_loss": SegmentationLoss.combined_loss,
            "dice_loss": SegmentationLoss.dice_loss,
        },
        compile=False,
    )

    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(),
    #     loss=SegmentationLoss.combined_loss,
    #     metrics=[CombinedLossMetric(name="combined_loss")],
    # )

    return model


if __name__ == "__main__":
    model = load_trained_model_h5()
    model.summary()