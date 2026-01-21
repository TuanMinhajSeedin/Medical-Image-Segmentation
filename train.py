import os
import sys
import io
import argparse
from pathlib import Path

# Fix encoding for Windows compatibility
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Silence noisy TensorFlow INFO logs in terminal output.
# Must be set BEFORE importing tensorflow (directly or indirectly).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all, 1=filter INFO, 2=filter INFO+WARNING, 3=filter all
# Optional: disable oneDNN verbose info line about custom ops.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


from LiverTumorSegmentation.pipeline.stage_03_model_training import ModelTrainingPipeline
from LiverTumorSegmentation import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config file (e.g., configs/exp_001.yaml). If not provided, uses params.yaml"
    )
    return parser.parse_args()


STAGE_NAME = "Training Stage"

if __name__ == "__main__":
    args = parse_args()
    
    # Resolve config path if provided
    exp_config_path = None
    if args.config:
        exp_config_path = Path(args.config)
        if not exp_config_path.is_absolute():
            exp_config_path = Path.cwd() / exp_config_path
        if not exp_config_path.exists():
            raise FileNotFoundError(f"Experiment config file not found: {exp_config_path}")
        logger.info(f"Using experiment config: {exp_config_path}")
    
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline(experiment_config_path=exp_config_path)
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
