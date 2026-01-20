from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict
import gzip
import pickle
import types
import numpy as np
import tensorflow as tf
import shutil

# Force CPU execution to avoid GPU PTX compilation issues (matching train_sample.py)
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    # If GPUs are not present or configuration fails, just continue.
    pass
import os
import sys

# Ensure we're working from the project root
# Find project root by looking for configs directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent  # research/ -> project root
if (project_root / "configs" / "config.yaml").exists():
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
elif (script_dir / "configs" / "config.yaml").exists():
    os.chdir(script_dir)
    sys.path.insert(0, str(script_dir))
else:
    # Try to find project root by looking for src/LiverTumorSegmentation
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "configs" / "config.yaml").exists():
            os.chdir(parent)
            sys.path.insert(0, str(parent))
            project_root = parent
            break
    else:
        raise FileNotFoundError(
            f"Could not find project root. Looking for configs/config.yaml. "
            f"Current directory: {Path.cwd()}, Script directory: {script_dir}"
        )

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    copy_trained_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list


from LiverTumorSegmentation.constants import *
from LiverTumorSegmentation.utils.common import read_yaml, create_directories

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=None,
        params_filepath=None
    ):
        # Resolve paths relative to project root
        if config_filepath is None:
            config_filepath = Path("configs/config.yaml")
        if params_filepath is None:
            params_filepath = Path("configs/params.yaml")
        
        # Convert to absolute paths if they're relative
        if not Path(config_filepath).is_absolute():
            config_filepath = Path.cwd() / config_filepath
        if not Path(params_filepath).is_absolute():
            params_filepath = Path.cwd() / params_filepath
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = self.config.training.training_data
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            copy_trained_model_path=Path(training.copy_trained_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )
        return training_config


# ----------------------------
# Segmentation Data Loading Utilities
# ----------------------------

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


def _collect_arrays(obj: Any) -> List[np.ndarray]:
    """Collect numpy arrays recursively from dict/list/tuple/object.__dict__."""
    arrays: List[np.ndarray] = []
    if obj is None:
        return arrays
    if isinstance(obj, np.ndarray):
        return [obj]
    if isinstance(obj, dict):
        for v in obj.values():
            arrays.extend(_collect_arrays(v))
        return arrays
    if isinstance(obj, (list, tuple)):
        for v in obj:
            arrays.extend(_collect_arrays(v))
        return arrays
    if hasattr(obj, "__dict__"):
        arrays.extend(_collect_arrays(vars(obj)))
        return arrays
    return arrays


def _pick_best_array(arrays: List[np.ndarray]) -> np.ndarray | None:
    """Prefer image-like arrays over tiny 1D metadata."""
    if not arrays:
        return None
    arrays = [np.asarray(a) for a in arrays]
    arrays.sort(key=lambda a: (a.ndim, a.size), reverse=True)
    return arrays[0]


def extract_array(record) -> np.ndarray:
    """Grab the first available array-like entry from common keys."""
    record = _as_mapping(record)
    for key in ("image", "pred", "prediction", "logits", "mask", "label", "data", "arr", "array", "_array"):
        if key in record:
            arr = np.asarray(record[key])
            if arr.ndim == 2:
                arr = arr[None, ...]
            return arr
    arr = _pick_best_array(_collect_arrays(record))
    if arr is not None:
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = arr[None, ...]
        return arr
    raise KeyError("No known array keys found in record.")


def extract_mask(record) -> np.ndarray:
    """Return mask/label array from common keys."""
    record = _as_mapping(record)
    for key in ("mask", "label", "target", "segmentation", "gt", "_array", "data"):
        if key in record:
            arr = np.asarray(record[key])
            if arr.ndim == 2:
                arr = arr[None, ...]
            return arr
    arr = _pick_best_array(_collect_arrays(record))
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


def load_pickle(path: Path):
    """Load pickle file, supporting both .pkl and .pkl.gz."""
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return SafeUnpickler(f).load()
    else:
        with open(path, "rb") as f:
            return SafeUnpickler(f).load()


class PKLSegmentationDataset:
    """Loads paired inputs and masks using matching filenames."""
    def __init__(self, pred_dir: Path, gt_dir: Path):
        self.pred_dir = pred_dir
        self.gt_dir = gt_dir
        self.items = self._collect_items()

    def _collect_items(self) -> List[Tuple[Path, Path]]:
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
        pred_path, gt_path = self.items[idx]
        pred_rec = load_pickle(pred_path)
        gt_rec = load_pickle(gt_path)

        x = extract_array(pred_rec).astype(np.float32)
        y = extract_mask(gt_rec).astype(np.float32)

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
                f"Expected x/y to be 3D (H,W,C). Got x.shape={x.shape}, y.shape={y.shape} "
                f"from pred={pred_path.name}, gt={gt_path.name}."
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


# ----------------------------
# Loss Functions (matching train_sample.py)
# ----------------------------

def dice_loss(y_true: tf.Tensor, y_pred_logits: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    """Dice loss for segmentation."""
    y_pred = tf.nn.sigmoid(y_pred_logits)
    numerator = 2.0 * tf.reduce_sum(y_pred * y_true, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_pred + y_true, axis=(1, 2, 3)) + eps
    return 1.0 - tf.reduce_mean(numerator / denominator)


def split_indices(n: int, val_split: float) -> Tuple[List[int], List[int]]:
    """Split indices into train and validation sets."""
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    val_len = max(1, int(n * val_split))
    val_idx = idxs[:val_len].tolist()
    train_idx = idxs[val_len:].tolist()
    return train_idx, val_idx


class Training:
    def __init__(self, config: TrainingConfig, learning_rate: float = 0.001):
        self.config = config
        self.learning_rate = learning_rate
        self.model: Optional[tf.keras.Model] = None
        self.train_dataset: Optional[tf.data.Dataset] = None
        self.val_dataset: Optional[tf.data.Dataset] = None

    def get_base_model(self):
        """Load the base model from the updated base model path and recompile it."""
        # Load model without optimizer state to avoid variable mismatch
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False  # Don't load optimizer state
        )
        
        # Recompile with combined loss (BCE + Dice) matching train_sample.py
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        def combined_loss(y_true, y_pred):
            return bce(y_true, y_pred) + dice_loss(y_true, y_pred)
        
        class CombinedLossMetric(tf.keras.metrics.Mean):
            """Tracks the combined loss but accepts (y_true, y_pred)."""
            def update_state(self, y_true, y_pred, sample_weight=None):
                value = combined_loss(y_true, y_pred)
                return super().update_state(value, sample_weight=sample_weight)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=combined_loss,
            metrics=[CombinedLossMetric(name="combined_loss")],
        )

    def train_valid_generator(self, predictions_subdir: str = "Dice", val_split: float = 0.1, shuffle_buffer: int = 128):
        """Create train and validation datasets for segmentation."""
        # Check if training data directory exists
        if not self.config.training_data.exists():
            raise FileNotFoundError(
                f"Training data directory not found: {self.config.training_data}\n"
                f"Please ensure data ingestion has been completed and the directory exists."
            )
        
        # Setup paths
        pred_dir = self.config.training_data / "Predictions" / predictions_subdir
        gt_dir = self.config.training_data / "GroundTruth"
        
        if not pred_dir.exists():
            raise FileNotFoundError(
                f"Predictions directory not found: {pred_dir}\n"
                f"Available subdirectories in Predictions: {list((self.config.training_data / 'Predictions').iterdir()) if (self.config.training_data / 'Predictions').exists() else 'None'}"
            )
        
        if not gt_dir.exists():
            raise FileNotFoundError(f"GroundTruth directory not found: {gt_dir}")
        
        # Load dataset
        dataset = PKLSegmentationDataset(pred_dir, gt_dir)
        
        # Split indices using the same function as train_sample.py
        train_idx, val_idx = split_indices(len(dataset), val_split)
        
        # Get target size from config
        target_h, target_w = self.config.params_image_size[0], self.config.params_image_size[1]
        in_channels = self.config.params_image_size[2]
        
        def gen(idxs: List[int]):
            for i in idxs:
                try:
                    x, y = dataset[i]
                except ValueError as e:
                    pred_path, gt_path = dataset.items[i]
                    print(
                        f"[Training] Skipping sample pred={pred_path.name}, gt={gt_path.name}: {e}"
                    )
                    continue
                yield x, y
        
        # Create datasets
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
        
        def resize_pair(x, y):
            x = tf.image.resize(x, (target_h, target_w), method="bilinear")
            y = tf.image.resize(y, (target_h, target_w), method="nearest")
            x.set_shape((target_h, target_w, in_channels))
            y.set_shape((target_h, target_w, in_channels))
            return x, y
        
        # Apply transformations (matching train_sample.py)
        self.train_dataset = (
            train_ds.map(resize_pair, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(buffer_size=min(shuffle_buffer, len(train_idx)))
            .batch(self.config.params_batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        self.val_dataset = (
            val_ds.map(resize_pair, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.config.params_batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        print(f"Created datasets: {len(train_idx)} training samples, {len(val_idx)} validation samples")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model) -> None:
        """Save the trained model to the specified path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)

    def train(self):
        """Train the model using the configured datasets (matching train_sample.py)."""
        if self.train_dataset is None or self.val_dataset is None:
            raise ValueError("Datasets not initialized. Call train_valid_generator() first.")
        
        self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.config.params_epochs
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

    def copy_model(self) -> None:
        """Copy the trained model file to the copy directory."""
        src = Path(self.config.trained_model_path)
        dst_dir = Path(self.config.copy_trained_model_path)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        shutil.copy2(src, dst)


try:
    config = ConfigurationManager()
    training_config = config.get_training_config()
    learning_rate = config.params.LEARNING_RATE
    training = Training(config=training_config, learning_rate=learning_rate)
    training.get_base_model()
    training.train_valid_generator(predictions_subdir="Dice")  # Can be changed to other subdirs
    training.train()
    training.copy_model()
except Exception as e:
    raise e
