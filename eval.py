"""CLI entrypoint to run evaluation with optional params/ckpt overrides."""

import argparse
from pathlib import Path

from LiverTumorSegmentation.config.configuration import ConfigurationManager
from LiverTumorSegmentation.components.evaluation import SegmentationEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/params.yaml",
        help="Path to params YAML (e.g., configs/exp_001.yaml).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to a model checkpoint (.h5) to evaluate. Overrides config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Init configuration with optional params override
    cm = ConfigurationManager(params_filepath=Path(args.config))
    eval_config = cm.get_default_eval_config(ckpt_path=Path(args.ckpt) if args.ckpt else None)

    evaluator = SegmentationEvaluator(eval_config)
    evaluator.evaluate()
    evaluator.save_score()
    evaluator.log_into_mlflow()


if __name__ == "__main__":
    main()

