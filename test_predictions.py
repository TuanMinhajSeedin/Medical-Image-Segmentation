"""
Script to test predictions on multiple samples and save results.
"""
import os
import sys
import io
from pathlib import Path
from typing import List

# Fix encoding for Windows compatibility
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from LiverTumorSegmentation.pipeline.prediction import PredictionPipeline
from LiverTumorSegmentation import logger
from LiverTumorSegmentation.utils.common import read_yaml, save_json


def test_predictions(
    predictions_subdir: str = "Dice",
    sample_ids: List[str] = None,
    threshold: float = 0.5,
    config_path: str = "configs/config.yaml"
) -> dict:
    """
    Test predictions on multiple samples and save results.
    
    Args:
        predictions_subdir: Subdirectory in Predictions folder (e.g., "Dice", "CE")
        sample_ids: List of sample IDs to test (e.g., ["3", "6", "101"]). 
                   If None, tests all available samples.
        threshold: Threshold for binary classification
        config_path: Path to config.yaml
        
    Returns:
        Dictionary with all prediction results
    """
    config = read_yaml(Path(config_path))
    pred_config = config.prediction
    
    # Set up paths
    predictions_dir = Path("artifacts/data_ingestion/data/Predictions") / predictions_subdir
    results_dir = Path(pred_config.results_data)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    
    # Get list of samples to test
    if sample_ids is None:
        # Test all available .pkl files
        pkl_files = sorted(predictions_dir.glob("*.pkl"))
        sample_ids = [f.stem for f in pkl_files]
        logger.info(f"Found {len(sample_ids)} samples to test")
    else:
        # Test specific samples
        sample_ids = [str(sid) for sid in sample_ids]
    
    logger.info(f"Testing predictions on {len(sample_ids)} samples from {predictions_subdir}/")
    logger.info(f"Results will be saved to: {results_dir}")
    
    all_results = {}
    successful = 0
    failed = 0
    
    for sample_id in sample_ids:
        pickle_file = predictions_dir / f"{sample_id}.pkl"
        
        if not pickle_file.exists():
            logger.warning(f"Sample {sample_id} not found: {pickle_file}")
            all_results[sample_id] = {
                "status": "failed",
                "error": "File not found",
                "input_file": str(pickle_file)
            }
            failed += 1
            continue
        
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing sample: {sample_id}")
            logger.info(f"{'='*60}")
            
            # Initialize pipeline
            pipeline = PredictionPipeline(
                pickle_file_path=str(pickle_file),
                config_path=config_path
            )
            
            # Make prediction
            result = pipeline.predict(threshold=threshold)
            
            # Save individual prediction
            saved_path = pipeline.save_prediction(result, output_dir=results_dir)
            
            # Store in results
            all_results[sample_id] = {
                "status": "success",
                "input_file": str(pickle_file),
                "output_file": str(saved_path),
                "tumor_pixels": result["tumor_pixels"],
                "tumor_percentage": result["tumor_percentage"],
                "prediction_shape": result["prediction_shape"],
                "threshold_used": result["threshold_used"],
                "probability_max": result["probability_stats"]["max"],
            }
            
            successful += 1
            logger.info(f"✓ Successfully processed {sample_id}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {sample_id}: {e}")
            all_results[sample_id] = {
                "status": "failed",
                "error": str(e),
                "input_file": str(pickle_file)
            }
            failed += 1
    
    # Save summary
    summary = {
        "predictions_subdir": predictions_subdir,
        "threshold_used": threshold,
        "total_samples": len(sample_ids),
        "successful": successful,
        "failed": failed,
        "results": all_results
    }
    
    summary_path = results_dir / "predictions_summary.json"
    save_json(summary_path, summary)
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {len(sample_ids)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Summary saved to: {summary_path}")
    
    return summary


if __name__ == "__main__":
    # Test on a few samples
    test_predictions(
        predictions_subdir="Dice",
        sample_ids=["3", "6", "101", "128"],  # Test specific samples
        threshold=0.5,
        config_path="configs/config.yaml"
    )

