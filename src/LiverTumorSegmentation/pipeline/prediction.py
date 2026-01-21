import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Optional, Tuple
from LiverTumorSegmentation.utils.common import read_yaml
from LiverTumorSegmentation.components.training import (
    PickleLoader,
    PickleDataLoader,
)
from LiverTumorSegmentation.components.training import (
    SegmentationLoss,
    CombinedLossMetric,
    DiceMetric,
    IoUMetric,
)
from LiverTumorSegmentation import logger
from LiverTumorSegmentation.utils.common import save_json, create_directories
import tensorflow as tf


class PredictionPipeline:
    def __init__(self, pickle_file_path: str, config_path: str):
        """
        Initialize prediction pipeline for segmentation.
        
        Args:
            pickle_file_path: Path to the .pkl file to predict on (from Predictions folder)
            config_path: Path to config.yaml
        """
        self.pickle_file_path = Path(pickle_file_path)
        self.config_path = Path(config_path)
        self.config = read_yaml(self.config_path)
        self.model = None

    def _preprocess_input(self, x: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Preprocess input array for model inference.
        
        Args:
            x: Input array from pickle file
            target_size: Optional target size (H, W). If None, uses config IMAGE_SIZE
            
        Returns:
            Preprocessed array ready for model input
        """
        x = x.astype(np.float32)
        
        # Normalize inputs to [0,1]
        if x.max() > 1.0:
            x = x / 255.0
        
        # Convert to channel-last for TF: (C,H,W) -> (H,W,C)
        if x.ndim == 3:
            x = np.transpose(x, (1, 2, 0))
        
        # Ensure 3D shape (H, W, C)
        if x.ndim == 2:
            x = x[..., None]
        
        if x.ndim != 3:
            raise ValueError(f"Expected x to be 3D (H,W,C) or 2D (H,W). Got x.shape={x.shape}.")
        
        # Reduce to single-channel if multi-channel
        if x.shape[-1] > 1:
            x = x[..., :1]
        
        # Get target size from config if not provided
        if target_size is None:
            # Try to get from config, fallback to default
            try:
                params = read_yaml(Path("configs/params.yaml"))
                image_size = params.IMAGE_SIZE
                target_size = (image_size[0], image_size[1])
            except:
                target_size = (128, 128)  # Default
        
        # Resize to target size
        if x.shape[0] != target_size[0] or x.shape[1] != target_size[1]:
            x = tf.image.resize(x, target_size, method="bilinear").numpy()
        
        # Pad to be divisible by 4 (model requirement)
        h, w, _ = x.shape
        pad_h = (-h) % 4
        pad_w = (-w) % 4
        if pad_h or pad_w:
            x = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        
        return x

    def load_model(self) -> tf.keras.Model:
        """Load the trained segmentation model with custom objects."""
        model_path = Path(self.config.training.trained_model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model from: {model_path}")
        
        model = tf.keras.models.load_model(
            model_path,
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
        
        # Recompile for inference (ensures metrics work if needed)
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
        logger.info("Model loaded successfully")
        return model

    def predict(self, threshold: float = 0.5) -> dict:
        """
        Predict segmentation mask from pickle file.
        
        Args:
            threshold: Threshold for binary classification (default: 0.5).
                      Lower values (e.g., 0.3, 0.4) will predict more pixels as tumor.
        
        Returns:
            Dictionary containing:
            - 'segmentation_mask': Binary segmentation mask (numpy array as list)
            - 'prediction_shape': Shape of the prediction
            - 'tumor_pixels': Number of pixels predicted as tumor
            - 'background_pixels': Number of pixels predicted as background
            - 'tumor_percentage': Percentage of image predicted as tumor
            - 'threshold_used': The threshold value used
            - 'raw_logits_stats': Statistics of raw logits
            - 'probability_stats': Statistics of probabilities
        """
        if not self.pickle_file_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {self.pickle_file_path}")
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Load pickle file
        logger.info(f"Loading pickle file: {self.pickle_file_path}")
        pickle_data = PickleLoader.load(self.pickle_file_path)
        
        # Extract array from pickle
        input_array = PickleDataLoader.extract_array(pickle_data)
        logger.info(f"Extracted array shape: {input_array.shape}")
        
        # Preprocess input
        preprocessed_input = self._preprocess_input(input_array)
        logger.info(f"Preprocessed input shape: {preprocessed_input.shape}")
        
        # Add batch dimension for model input
        model_input = np.expand_dims(preprocessed_input, axis=0)
        logger.info(f"Model input shape: {model_input.shape}")
        
        # Run prediction
        logger.info("Running model prediction...")
        prediction_logits = self.model.predict(model_input, verbose=0)
        
        # Remove batch dimension
        prediction_logits = prediction_logits[0]  # Remove batch dimension
        
        # If logits have channel dimension, squeeze it
        if prediction_logits.ndim == 3 and prediction_logits.shape[-1] == 1:
            prediction_logits = prediction_logits[..., 0]
        
        # Convert logits to probabilities
        prediction_probs = tf.sigmoid(prediction_logits).numpy()
        
        # Log diagnostic information
        logger.info(f"Raw logits - Min: {prediction_logits.min():.4f}, Max: {prediction_logits.max():.4f}, Mean: {prediction_logits.mean():.4f}")
        logger.info(f"Probabilities - Min: {prediction_probs.min():.4f}, Max: {prediction_probs.max():.4f}, Mean: {prediction_probs.mean():.4f}")
        logger.info(f"Probabilities > 0.5: {(prediction_probs > 0.5).sum()} pixels ({(prediction_probs > 0.5).sum() / prediction_probs.size * 100:.2f}%)")
        logger.info(f"Probabilities > {threshold}: {(prediction_probs > threshold).sum()} pixels ({(prediction_probs > threshold).sum() / prediction_probs.size * 100:.2f}%)")
        
        # Suggest optimal threshold if max probability < threshold
        if prediction_probs.max() < threshold:
            suggested_threshold = max(0.1, prediction_probs.max() * 0.8)  # Use 80% of max as suggestion
            logger.warning(f"Max probability ({prediction_probs.max():.4f}) < threshold ({threshold}). "
                         f"Suggested threshold: {suggested_threshold:.4f}")
        
        # Create binary mask with threshold
        segmentation_mask = (prediction_probs > threshold).astype(np.uint8)
        
        logger.info(f"Prediction mask shape: {segmentation_mask.shape}")
        
        # Calculate statistics
        unique_values, counts = np.unique(segmentation_mask, return_counts=True)
        value_counts = dict(zip(unique_values.tolist(), counts.tolist()))
        tumor_pixels = value_counts.get(1, 0)
        background_pixels = value_counts.get(0, 0)
        total_pixels = segmentation_mask.size
        tumor_percentage = (tumor_pixels / total_pixels * 100) if total_pixels > 0 else 0
        
        result = {
            "input_file": str(self.pickle_file_path),  # Reference to input file
            "segmentation_mask": segmentation_mask.tolist(),  # Convert to list for JSON serialization
            "prediction_shape": list(segmentation_mask.shape),
            "tumor_pixels": int(tumor_pixels),
            "background_pixels": int(background_pixels),
            "tumor_percentage": float(tumor_percentage),
            "unique_values": unique_values.tolist(),
            "raw_logits_stats": {
                "min": float(prediction_logits.min()),
                "max": float(prediction_logits.max()),
                "mean": float(prediction_logits.mean()),
            },
            "probability_stats": {
                "min": float(prediction_probs.min()),
                "max": float(prediction_probs.max()),
                "mean": float(prediction_probs.mean()),
            },
            "threshold_used": threshold,
        }
        
        logger.info(f"Prediction complete. Tumor pixels: {tumor_pixels} ({tumor_percentage:.2f}%)")
        
        return result

    def save_prediction(self, result: dict, output_dir: Optional[Path] = None) -> Path:
        """
        Save prediction results to a JSON file.
        
        Args:
            result: Prediction result dictionary from predict() method
            output_dir: Directory to save prediction. If None, uses config.prediction.results_data
            
        Returns:
            Path to the saved prediction file
        """
        # Get output directory from config
        if output_dir is None:
            try:
                pred_config = self.config.prediction
                output_dir = Path(pred_config.results_data)
            except AttributeError:
                output_dir = Path("artifacts/prediction/results")
        
        # Create directory if it doesn't exist
        create_directories([output_dir])
        
        # Generate output filename from input filename
        input_filename = self.pickle_file_path.stem  # e.g., "3" from "3.pkl"
        output_filename = f"{input_filename}_prediction.json"
        output_path = output_dir / output_filename
        
        # Save prediction result
        save_json(output_path, result)
        logger.info(f"Saved prediction result to: {output_path}")
        
        return output_path


