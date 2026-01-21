from pathlib import Path
from LiverTumorSegmentation.pipeline.prediction import PredictionPipeline
from LiverTumorSegmentation.utils.common import read_yaml

# Read config
config = read_yaml(Path("configs/config.yaml"))
pickle_file_path = config.prediction.pickle_file_path
prediction_data_dir = Path(config.prediction.prediction_data)

# Initialize pipeline
pipeline = PredictionPipeline(
    pickle_file_path=pickle_file_path,
    config_path="configs/config.yaml"
)

# Make prediction
result = pipeline.predict(threshold=0.5)

# Save prediction data
saved_path = pipeline.save_prediction(result, output_dir=prediction_data_dir)
print(f"\n✓ Prediction saved to: {saved_path}")

# Display results
print(f"\n{'='*60}")
print("PREDICTION RESULTS")
print(f"{'='*60}")
print(f"Input file: {pickle_file_path}")
print(f"Tumor pixels: {result['tumor_pixels']}")
print(f"Tumor percentage: {result['tumor_percentage']:.2f}%")
print(f"Prediction shape: {result['prediction_shape']}")
print(f"\nDiagnostic Information:")
print(f"  Raw logits - Min: {result['raw_logits_stats']['min']:.4f}, Max: {result['raw_logits_stats']['max']:.4f}, Mean: {result['raw_logits_stats']['mean']:.4f}")
print(f"  Probabilities - Min: {result['probability_stats']['min']:.4f}, Max: {result['probability_stats']['max']:.4f}, Mean: {result['probability_stats']['mean']:.4f}")
print(f"  Threshold used: {result['threshold_used']}")
print(f"{'='*60}")
