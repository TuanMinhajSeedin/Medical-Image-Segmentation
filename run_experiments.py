"""
Script to run multiple experiments and compare their results.
Executes training and evaluation for each experiment config file.
"""

import os
import sys
import io
import subprocess
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Fix encoding for Windows compatibility
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from LiverTumorSegmentation import logger
from LiverTumorSegmentation.utils.common import load_json, save_json, read_yaml


def find_experiment_configs(config_dir: Path = Path("configs")) -> List[Path]:
    """Find all experiment config files (exp_*.yaml)."""
    config_files = sorted(config_dir.glob("exp_*.yaml"))
    logger.info(f"Found {len(config_files)} experiment configs: {[f.name for f in config_files]}")
    return config_files


def run_training(config_path: Path) -> bool:
    """Run training for a given experiment config."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training with config: {config_path.name}")
    logger.info(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, "train.py", "--config", str(config_path)],
            check=True,
            capture_output=False,
            text=True
        )
        logger.info(f"Training completed successfully for {config_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed for {config_path.name}: {e}")
        return False


def run_evaluation(config_path: Path) -> tuple[bool, bool]:
    """Run evaluation for a given experiment config.
    
    Returns:
        tuple: (mlflow_success, scores_saved) - MLflow may fail but scores might still be saved
    """
    logger.info(f"\nEvaluating model trained with config: {config_path.name}")
    scores_path = Path("scores.json")
    
    try:
        result = subprocess.run(
            [sys.executable, "eval.py", "--config", str(config_path)],
            check=True,
            capture_output=False,
            text=True
        )
        logger.info(f"Evaluation completed successfully for {config_path.name}")
        return True, scores_path.exists()
    except subprocess.CalledProcessError as e:
        # Check if scores were saved even if MLflow failed
        scores_exist = scores_path.exists()
        if scores_exist:
            logger.warning(f"Evaluation encountered an error (likely MLflow logging), but scores were saved: {e}")
        else:
            logger.error(f"Evaluation failed for {config_path.name}: {e}")
        return False, scores_exist


def load_experiment_scores(scores_path: Path = Path("scores.json")) -> Dict:
    """Load scores from the scores.json file."""
    if not scores_path.exists():
        logger.warning(f"Scores file not found: {scores_path}")
        return {}
    
    try:
        scores = load_json(scores_path)
        # Convert ConfigBox to dict
        return dict(scores)
    except Exception as e:
        logger.error(f"Error loading scores: {e}")
        return {}


def load_experiment_config(config_path: Path) -> Dict:
    """Load experiment config parameters for comparison."""
    try:
        config = read_yaml(config_path)
        return {
            "batch_size": config.BATCH_SIZE,
            "epochs": config.EPOCHS,
            "learning_rate": config.LEARNING_RATE,
            "base_filters": getattr(config, "BASE_FILTERS", None),
        }
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        return {}


def run_all_experiments(config_files: List[Path], results_dir: Path = Path("experiment_results")) -> Dict[str, Dict]:
    """Run all experiments and collect results."""
    all_results = {}
    results_dir.mkdir(exist_ok=True)
    scores_path = Path("scores.json")
    
    for config_path in config_files:
        exp_name = config_path.stem  # e.g., "exp_001"
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"Starting experiment: {exp_name}")
        logger.info(f"{'#'*80}")
        
        # Run training
        training_success = run_training(config_path)
        if not training_success:
            logger.error(f"Skipping evaluation for {exp_name} due to training failure")
            all_results[exp_name] = {
                "config_path": str(config_path),
                "training_success": False,
                "evaluation_success": False,
                "scores": {},
                "config_params": {}
            }
            continue
        
        # Run evaluation (MLflow may fail, but scores might still be saved)
        mlflow_success, scores_saved = run_evaluation(config_path)
        
        # Load and save scores immediately (before next experiment overwrites scores.json)
        # Check for scores even if MLflow failed, since evaluation might have succeeded
        scores = {}
        if scores_path.exists():
            scores = load_experiment_scores(scores_path)
            # Save a copy of scores for this experiment
            exp_scores_path = results_dir / f"{exp_name}_scores.json"
            save_json(exp_scores_path, scores)
            logger.info(f"Saved scores for {exp_name} to {exp_scores_path}")
        
        config_params = load_experiment_config(config_path)
        
        # Evaluation is considered successful if scores were saved, even if MLflow failed
        evaluation_success = scores_saved
        
        all_results[exp_name] = {
            "config_path": str(config_path),
            "training_success": training_success,
            "evaluation_success": evaluation_success,
            "mlflow_success": mlflow_success,
            "scores": scores,
            "config_params": config_params
        }
        
        logger.info(f"\nResults for {exp_name}:")
        logger.info(f"  Training: {'[OK]' if training_success else '[FAIL]'}")
        logger.info(f"  Evaluation: {'[OK]' if evaluation_success else '[FAIL]'}")
        if not mlflow_success and evaluation_success:
            logger.info(f"  MLflow Logging: [FAIL] (non-fatal, scores saved)")
        if scores:
            logger.info(f"  Loss: {scores.get('loss', 'N/A')}")
            logger.info(f"  Combined Loss: {scores.get('combined_loss', 'N/A')}")
            logger.info(f"  Dice: {scores.get('dice', 'N/A')}")
            logger.info(f"  IoU: {scores.get('iou', 'N/A')}")
    
    return all_results


def compare_results(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a comparison DataFrame from all experiment results."""
    comparison_data = []
    
    for exp_name, result in results.items():
        row = {
            "Experiment": exp_name,
            "Batch Size": result["config_params"].get("batch_size", "N/A"),
            "Epochs": result["config_params"].get("epochs", "N/A"),
            "Learning Rate": result["config_params"].get("learning_rate", "N/A"),
            "Base Filters": result["config_params"].get("base_filters", "N/A"),
            "Training Success": "✓" if result["training_success"] else "✗",
            "Evaluation Success": "✓" if result["evaluation_success"] else "✗",
            "MLflow Success": "✓" if result.get("mlflow_success", False) else "✗",
            "Loss": result["scores"].get("loss", None),
            "Combined Loss": result["scores"].get("combined_loss", None),
            "Dice": result["scores"].get("dice", None),
            "IoU": result["scores"].get("iou", None),
        }
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df


def print_comparison(df: pd.DataFrame):
    """Print a formatted comparison table."""
    print("\n\n" + "="*100)
    print("EXPERIMENT COMPARISON RESULTS")
    print("="*100)
    
    # Print full DataFrame
    print("\nFull Results:")
    print(df.to_string(index=False))
    
    # Print summary statistics for successful experiments
    successful_exps = df[df["Evaluation Success"] == "✓"]
    if len(successful_exps) > 0:
        print("\n\nSummary Statistics (Successful Experiments Only):")
        print("-" * 100)
        
        metrics = ["Loss", "Combined Loss", "Dice", "IoU"]
        for metric in metrics:
            if metric in successful_exps.columns:
                values = successful_exps[metric].dropna()
                if len(values) > 0:
                    print(f"\n{metric}:")
                    print(f"  Best: {values.min():.6f} ({successful_exps.loc[values.idxmin(), 'Experiment']})")
                    print(f"  Worst: {values.max():.6f} ({successful_exps.loc[values.idxmax(), 'Experiment']})")
                    print(f"  Mean: {values.mean():.6f}")
                    print(f"  Std: {values.std():.6f}")
        
        # Find best experiment for each metric (lower is better for loss, higher is better for dice/iou)
        print("\n\nBest Experiments by Metric:")
        print("-" * 100)
        if "Loss" in successful_exps.columns:
            best_loss = successful_exps.loc[successful_exps["Loss"].idxmin()]
            print(f"Best Loss: {best_loss['Experiment']} ({best_loss['Loss']:.6f})")
        if "Combined Loss" in successful_exps.columns:
            best_combined = successful_exps.loc[successful_exps["Combined Loss"].idxmin()]
            print(f"Best Combined Loss: {best_combined['Experiment']} ({best_combined['Combined Loss']:.6f})")
        if "Dice" in successful_exps.columns:
            best_dice = successful_exps.loc[successful_exps["Dice"].idxmax()]
            print(f"Best Dice: {best_dice['Experiment']} ({best_dice['Dice']:.6f})")
        if "IoU" in successful_exps.columns:
            best_iou = successful_exps.loc[successful_exps["IoU"].idxmax()]
            print(f"Best IoU: {best_iou['Experiment']} ({best_iou['IoU']:.6f})")
    else:
        print("\nNo successful experiments to compare.")
    
    print("\n" + "="*100)


def save_comparison_results(results: Dict[str, Dict], df: pd.DataFrame, output_dir: Path = Path("experiment_results")):
    """Save comparison results to files."""
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results as JSON
    results_path = output_dir / "all_results.json"
    save_json(results_path, results)
    logger.info(f"Saved detailed results to: {results_path}")
    
    # Save comparison DataFrame as CSV
    csv_path = output_dir / "comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison table to: {csv_path}")
    
    # Save comparison DataFrame as Excel if openpyxl is available
    try:
        excel_path = output_dir / "comparison.xlsx"
        df.to_excel(excel_path, index=False)
        logger.info(f"Saved comparison table to: {excel_path}")
    except ImportError:
        logger.info("openpyxl not available, skipping Excel export")


def main():
    """Main function to run all experiments and compare results."""
    logger.info("="*80)
    logger.info("EXPERIMENT RUNNER - Running multiple configs and comparing results")
    logger.info("="*80)
    
    # Find all experiment config files
    config_files = find_experiment_configs()
    
    if not config_files:
        logger.error("No experiment config files found! Looking for exp_*.yaml in configs/")
        return
    
    # Run all experiments
    results_dir = Path("experiment_results")
    results = run_all_experiments(config_files, results_dir)
    
    # Compare results
    comparison_df = compare_results(results)
    
    # Print comparison
    print_comparison(comparison_df)
    
    # Save results
    save_comparison_results(results, comparison_df)
    
    logger.info("\n" + "="*80)
    logger.info("ALL EXPERIMENTS COMPLETED")
    logger.info("="*80)


if __name__ == "__main__":
    main()

