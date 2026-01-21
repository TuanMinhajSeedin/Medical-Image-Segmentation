from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict
import gzip
import pickle
import types
import numpy as np
import tensorflow as tf
import shutil

from LiverTumorSegmentation.entity.config_entity import TrainingConfig
from LiverTumorSegmentation.utils.common import *


class PickleDataLoader:
    """Handles loading and extracting data from pickle files."""
    
    @staticmethod
    def _as_mapping(record) -> Dict:
        """Normalize record to a dict-like object."""
        if record is None:
            return {}
        if isinstance(record, np.ndarray):
            return {"_array": record}
        if isinstance(record, dict):
            return record
        if hasattr(record, "__dict__"):
            return vars(record)
        return {}

    @staticmethod
    def _collect_arrays(obj: Any) -> List[np.ndarray]:
        """Collect numpy arrays recursively from dict/list/tuple/object.__dict__."""
        arrays: List[np.ndarray] = []
        if obj is None:
            return arrays
        if isinstance(obj, np.ndarray):
            return [obj]
        if isinstance(obj, dict):
            for v in obj.values():
                arrays.extend(PickleDataLoader._collect_arrays(v))
            return arrays
        if isinstance(obj, (list, tuple)):
            for v in obj:
                arrays.extend(PickleDataLoader._collect_arrays(v))
            return arrays
        if hasattr(obj, "__dict__"):
            arrays.extend(PickleDataLoader._collect_arrays(vars(obj)))
            return arrays
        return arrays

    @staticmethod
    def _pick_best_array(arrays: List[np.ndarray]) -> np.ndarray | None:
        """Prefer image-like arrays over tiny 1D metadata."""
        if not arrays:
            return None
        arrays = [np.asarray(a) for a in arrays]
        arrays.sort(key=lambda a: (a.ndim, a.size), reverse=True)
        return arrays[0]

    @staticmethod
    def extract_array(record) -> np.ndarray:
        """Extract array from record using common keys."""
        record = PickleDataLoader._as_mapping(record)
        for key in ("image", "pred", "prediction", "logits", "mask", "label", "data", "arr", "array", "_array"):
            if key in record:
                arr = np.asarray(record[key])
                if arr.ndim == 2:
                    arr = arr[None, ...]
                return arr
        arr = PickleDataLoader._pick_best_array(PickleDataLoader._collect_arrays(record))
        if arr is not None:
            arr = np.asarray(arr)
            if arr.ndim == 2:
                arr = arr[None, ...]
            return arr
        raise KeyError("No known array keys found in record.")

    @staticmethod
    def extract_mask(record) -> np.ndarray:
        """Extract mask array from record using common keys."""
        record = PickleDataLoader._as_mapping(record)
        for key in ("mask", "label", "target", "segmentation", "gt", "_array", "data"):
            if key in record:
                arr = np.asarray(record[key])
                if arr.ndim == 2:
                    arr = arr[None, ...]
                return arr
        arr = PickleDataLoader._pick_best_array(PickleDataLoader._collect_arrays(record))
        if arr is not None:
            arr = np.asarray(arr)
            if arr.ndim == 2:
                arr = arr[None, ...]
            return arr
        raise KeyError("No known mask keys found in record.")


class SafeUnpickler(pickle.Unpickler):
    """Allows loading pickles that reference modules/classes not present locally."""
    
    def find_class(self, module, name):
        if module.startswith("evalseg"):
            placeholder = types.new_class(name, ())
            placeholder.__module__ = module
            return placeholder
        return super().find_class(module, name)


class PickleLoader:
    """Handles loading pickle files (supports .pkl and .pkl.gz)."""
    
    @staticmethod
    def load(path: Path):
        """Load pickle file, supporting both .pkl and .pkl.gz."""
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as f:
                return SafeUnpickler(f).load()
        else:
            with open(path, "rb") as f:
                return SafeUnpickler(f).load()


class DataPreprocessor:
    """Handles preprocessing of image and mask pairs."""
    
    @staticmethod
    def preprocess_pair(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess image and mask pair for training."""
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # Normalize inputs to [0,1]
        if x.max() > 1.0:
            x = x / 255.0

        # Convert to channel-last for TF: (C,H,W) -> (H,W,C)
        if x.ndim == 3:
            x = np.transpose(x, (1, 2, 0))
        if y.ndim == 3:
            y = np.transpose(y, (1, 2, 0))

        # Fallback for metadata-only predictions
        if x.ndim < 3 or x.size <= 16:
            y_min, y_max = y.min(), y.max()
            if y_max > y_min:
                x = (y - y_min) / (y_max - y_min)
            else:
                x = y.astype(np.float32)

        # Align shapes
        if y.ndim >= 2:
            h, w = y.shape[0], y.shape[1]
            c = y.shape[2] if y.ndim == 3 else 1
            if x.ndim == 1 and x.size == h * w * c:
                x = x.reshape(h, w, c)
            elif x.ndim == 2 and x.shape == (h, w):
                x = x[..., None]
            elif x.ndim == 2 and x.size == h * w * c:
                x = x.reshape(h, w, c)

        # Enforce (H,W,C)
        if x.ndim == 2:
            x = x[..., None]
        if y.ndim == 2:
            y = y[..., None]
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError(
                f"Expected x/y to be 3D (H,W,C). Got x.shape={x.shape}, y.shape={y.shape}."
            )

        # Reduce to single-channel
        if x.shape[-1] > 1:
            x = x[..., :1]
        if y.shape[-1] > 1:
            y = (y > 0.5).astype(np.float32)
            y = y[..., :1]

        # Resize to match if needed
        if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
            y_h, y_w = y.shape[0], y.shape[1]
            x = tf.image.resize(x, (y_h, y_w), method="bilinear").numpy()

        # Pad to be divisible by 4
        h, w, _ = x.shape
        pad_h = (-h) % 4
        pad_w = (-w) % 4
        if pad_h or pad_w:
            x = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            y = np.pad(y, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

        return x, y


class PKLSegmentationDataset:
    """Loads paired inputs and masks using matching filenames."""
    
    def __init__(self, pred_dir: Path, gt_dir: Path):
        self.pred_dir = pred_dir
        self.gt_dir = gt_dir
        self.items = self._collect_items()

    def _collect_items(self) -> List[Tuple[Path, Path]]:
        """Collect paired prediction and ground-truth files."""
        pairs: List[Tuple[Path, Path]] = []
        for gt_path in sorted(self.gt_dir.glob("*.pkl")):
            name = gt_path.name
            pred_pkl = self.pred_dir / name
            pred_gz = self.pred_dir / f"{gt_path.stem}.pkl.gz"
            if pred_pkl.exists():
                pairs.append((pred_pkl, gt_path))
            elif pred_gz.exists():
                pairs.append((pred_gz, gt_path))
        if not pairs:
            raise FileNotFoundError(
                f"No paired files found between {self.pred_dir} and {self.gt_dir}."
            )
        return pairs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get preprocessed image and mask pair."""
        pred_path, gt_path = self.items[idx]
        pred_rec = PickleLoader.load(pred_path)
        gt_rec = PickleLoader.load(gt_path)

        x = PickleDataLoader.extract_array(pred_rec)
        y = PickleDataLoader.extract_mask(gt_rec)

        return DataPreprocessor.preprocess_pair(x, y)


# ============================================================================
# Loss and Metrics Classes
# ============================================================================

class SegmentationLoss:
    """Loss functions for segmentation tasks."""
    
    @staticmethod
    def dice_loss(y_true: tf.Tensor, y_pred_logits: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
        """Dice loss for segmentation."""
        y_pred = tf.nn.sigmoid(y_pred_logits)
        numerator = 2.0 * tf.reduce_sum(y_pred * y_true, axis=(1, 2, 3))
        denominator = tf.reduce_sum(y_pred + y_true, axis=(1, 2, 3)) + eps
        return 1.0 - tf.reduce_mean(numerator / denominator)

    @staticmethod
    def combined_loss(y_true: tf.Tensor, y_pred_logits: tf.Tensor) -> tf.Tensor:
        """Combined BCE + Dice loss."""
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return bce(y_true, y_pred_logits) + SegmentationLoss.dice_loss(y_true, y_pred_logits)


class CombinedLossMetric(tf.keras.metrics.Mean):
    """Tracks the combined loss but accepts (y_true, y_pred)."""
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        value = SegmentationLoss.combined_loss(y_true, y_pred)
        return super().update_state(value, sample_weight=sample_weight)


class DiceMetric(tf.keras.metrics.Metric):
    """Dice coefficient metric computed from logits."""

    def __init__(self, name: str = "dice", eps: float = 1e-6, **kwargs):
        super().__init__(name=name, **kwargs)
        self.eps = eps
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred_logits, sample_weight=None):
        y_pred = tf.nn.sigmoid(y_pred_logits)
        numerator = 2.0 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3)) + self.eps
        dice = numerator / denominator
        value = tf.reduce_mean(dice)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=value.dtype)
            value *= sample_weight
        self.total.assign_add(value)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class IoUMetric(tf.keras.metrics.Metric):
    """Intersection-over-Union metric computed from logits."""

    def __init__(self, name: str = "iou", eps: float = 1e-6, **kwargs):
        super().__init__(name=name, **kwargs)
        self.eps = eps
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred_logits, sample_weight=None):
        y_pred = tf.nn.sigmoid(y_pred_logits)
        intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
        union = tf.reduce_sum(y_true + y_pred - y_true * y_pred, axis=(1, 2, 3)) + self.eps
        iou = intersection / union
        value = tf.reduce_mean(iou)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=value.dtype)
            value *= sample_weight
        self.total.assign_add(value)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


# ============================================================================
# Dataset Builder Class
# ============================================================================

class DatasetBuilder:
    """Builds tf.data.Dataset for training and validation."""
    
    @staticmethod
    def split_indices(n: int, val_split: float) -> Tuple[List[int], List[int]]:
        """Split indices into train and validation sets."""
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        val_len = max(1, int(n * val_split))
        val_idx = idxs[:val_len].tolist()
        train_idx = idxs[val_len:].tolist()
        return train_idx, val_idx

    @staticmethod
    def build_datasets(
        dataset: PKLSegmentationDataset,
        train_idx: List[int],
        val_idx: List[int],
        target_h: int,
        target_w: int,
        in_channels: int,
        batch_size: int,
        shuffle_buffer: int = 128
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Build train and validation tf.data.Dataset objects."""
        
        def gen(idxs: List[int]):
            for i in idxs:
                try:
                    x, y = dataset[i]
                except ValueError as e:
                    pred_path, gt_path = dataset.items[i]
                    print(f"[Training] Skipping sample pred={pred_path.name}, gt={gt_path.name}: {e}")
                    continue
                yield x, y

        def resize_pair(x, y):
            x = tf.image.resize(x, (target_h, target_w), method="bilinear")
            y = tf.image.resize(y, (target_h, target_w), method="nearest")
            x.set_shape((target_h, target_w, in_channels))
            y.set_shape((target_h, target_w, in_channels))
            return x, y

        train_ds = tf.data.Dataset.from_generator(
            lambda: gen(train_idx),
            output_signature=(
                tf.TensorSpec(shape=(None, None, in_channels), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, in_channels), dtype=tf.float32),
            ),
        )

        val_ds = tf.data.Dataset.from_generator(
            lambda: gen(val_idx),
            output_signature=(
                tf.TensorSpec(shape=(None, None, in_channels), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, in_channels), dtype=tf.float32),
            ),
        )

        train_dataset = (
            train_ds.map(resize_pair, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(buffer_size=min(shuffle_buffer, len(train_idx)))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_dataset = (
            val_ds.map(resize_pair, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        return train_dataset, val_dataset


# ============================================================================
# Training Class
# ============================================================================

class ModelTrainer:
    """Main training class for segmentation models."""
    
    def __init__(self, config: TrainingConfig, learning_rate: float = 0.001):
        self.config = config
        self.learning_rate = learning_rate
        self.model: Optional[tf.keras.Model] = None
        self.train_dataset: Optional[tf.data.Dataset] = None
        self.val_dataset: Optional[tf.data.Dataset] = None

    def load_model(self):
        """Load the base model and recompile with appropriate loss."""
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False
        )
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=SegmentationLoss.combined_loss,
            metrics=[
                CombinedLossMetric(name="combined_loss"),
                DiceMetric(name="dice"),
                IoUMetric(name="iou"),
            ],
        )

    def prepare_datasets(
        self,
        predictions_subdir: str = "Dice",
        val_split: float = 0.1,
        shuffle_buffer: int = 128
    ):
        """Prepare train and validation datasets."""
        if not self.config.training_data.exists():
            raise FileNotFoundError(
                f"Training data directory not found: {self.config.training_data}"
            )
        
        pred_dir = self.config.training_data / "Predictions" / predictions_subdir
        gt_dir = self.config.training_data / "GroundTruth"
        
        if not pred_dir.exists():
            available = list((self.config.training_data / "Predictions").iterdir()) if (self.config.training_data / "Predictions").exists() else []
            raise FileNotFoundError(
                f"Predictions directory not found: {pred_dir}\n"
                f"Available subdirectories: {available}"
            )
        
        if not gt_dir.exists():
            raise FileNotFoundError(f"GroundTruth directory not found: {gt_dir}")
        
        dataset = PKLSegmentationDataset(pred_dir, gt_dir)
        train_idx, val_idx = DatasetBuilder.split_indices(len(dataset), val_split)
        
        target_h, target_w = self.config.params_image_size[0], self.config.params_image_size[1]
        in_channels = self.config.params_image_size[2]
        
        self.train_dataset, self.val_dataset = DatasetBuilder.build_datasets(
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            target_h=target_h,
            target_w=target_w,
            in_channels=in_channels,
            batch_size=self.config.params_batch_size,
            shuffle_buffer=shuffle_buffer
        )
        
        print(f"Created datasets: {len(train_idx)} training samples, {len(val_idx)} validation samples")
    
    def train(self):
        """Train the model."""
        if self.train_dataset is None or self.val_dataset is None:
            raise ValueError("Datasets not initialized. Call prepare_datasets() first.")

        # Ensure checkpoint directory exists
        create_directories([self.config.trained_model_path.parent])

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.trained_model_path),
            monitor="val_dice",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )

        self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.config.params_epochs,
            callbacks=[checkpoint_cb],
        )


    def copy_model(self):
        """Copy the trained model to the copy directory."""
        src = Path(self.config.trained_model_path)
        dst_dir = Path(self.config.copy_trained_model_path)
        # dst_dir.mkdir(parents=True, exist_ok=True)
        create_directories([dst_dir])
        dst = dst_dir / src.name
        shutil.copy2(src, dst)