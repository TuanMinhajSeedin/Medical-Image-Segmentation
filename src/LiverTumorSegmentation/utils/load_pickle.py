"""
Utility functions to load pickle files safely.
Handles both pickle and joblib formats, and missing module errors.
"""
import pickle
import joblib
import sys
import importlib
from pathlib import Path
from typing import Any, Optional
from LiverTumorSegmentation import logger


def _create_mock_module(module_name: str):
    """
    Create a mock module to satisfy pickle's import requirements.
    This allows loading pickle files that reference missing modules.
    """
    if module_name not in sys.modules:
        import types
        
        # Create a minimal mock module
        mock_module = types.ModuleType(module_name)
        mock_module.__file__ = '<mock>'
        
        # Add common attributes that might be expected
        class MockClass:
            def __init__(self, *args, **kwargs):
                # Store args/kwargs for potential inspection
                self._args = args
                self._kwargs = kwargs
                # Allow any attribute access
                for key, value in kwargs.items():
                    setattr(self, key, value)
            
            def __getattr__(self, name):
                # Return None for any missing attribute
                return None
        
        # Add some common class names that might be in evalseg
        mock_module.SegmentationResult = MockClass
        mock_module.EvaluationResult = MockClass
        mock_module.MetricResult = MockClass
        mock_module.Result = MockClass
        
        sys.modules[module_name] = mock_module
        logger.info(f"Created mock module: {module_name}")


def _try_install_module(module_name: str) -> bool:
    """
    Try to install a missing module using pip.
    
    Returns:
        True if installation succeeded or module already exists, False otherwise
    """
    try:
        # Check if module exists
        importlib.import_module(module_name)
        return True
    except ImportError:
        logger.info(f"Module {module_name} not found. Attempting to install...")
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", module_name],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                logger.info(f"Successfully installed {module_name}")
                return True
            else:
                logger.warning(f"Failed to install {module_name}: {result.stderr}")
                return False
        except Exception as e:
            logger.warning(f"Could not install {module_name}: {e}")
            return False


def load_pickle_file(file_path: str | Path, use_joblib: bool = True) -> Any:
    """
    Load a pickle file using either pickle or joblib.
    
    Args:
        file_path: Path to the pickle file
        use_joblib: If True, try joblib first (default). If False, use pickle.
        
    Returns:
        The loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If loading fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    
    logger.info(f"Loading pickle file: {file_path}")
    
    # Try joblib first (default for this project)
    if use_joblib:
        try:
            data = joblib.load(file_path)
            logger.info(f"Successfully loaded using joblib")
            return data
        except Exception as e:
            logger.warning(f"Failed to load with joblib: {e}. Trying pickle...")
            # Fall through to pickle
    
    # Try pickle
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Successfully loaded using pickle")
        return data
    except ModuleNotFoundError as e:
        # Handle missing module errors (like 'evalseg')
        module_name = str(e).split("'")[1] if "'" in str(e) else None
        logger.warning(f"Missing module required by pickle file: {e}")
        
        if module_name:
            # Try to install the module
            if _try_install_module(module_name):
                # Retry loading after installation
                logger.info("Retrying load after module installation...")
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    logger.info("Successfully loaded after installing module")
                    return data
                except Exception as retry_error:
                    logger.warning(f"Still failed after installation: {retry_error}")
            
            # If installation failed or retry failed, create mock module
            logger.info(f"Creating mock module for {module_name} to allow loading...")
            _create_mock_module(module_name)
            
            # Try loading again with mock module
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info("Successfully loaded using mock module")
                return data
            except Exception as mock_error:
                logger.error(f"Failed even with mock module: {mock_error}")
        
        # Final fallback: try joblib
        logger.info("Attempting to load with joblib as final fallback...")
        try:
            # Also create mock for joblib attempt
            if module_name:
                _create_mock_module(module_name)
            data = joblib.load(file_path)
            logger.info("Successfully loaded using joblib after creating mock module")
            return data
        except Exception as e2:
            raise Exception(
                f"Failed to load pickle file. Missing module: {e}\n"
                f"Tried installing module, creating mock, and joblib fallback.\n"
                f"Final error: {e2}\n"
                f"You may need to install missing dependencies: pip install {module_name if module_name else 'evalseg'}"
            ) from e
    except Exception as e:
        raise Exception(f"Failed to load pickle file: {e}") from e


def inspect_pickle_file(file_path: str | Path) -> dict:
    """
    Inspect a pickle file to understand its structure without fully loading it.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary with file information
    """
    file_path = Path(file_path)
    
    info = {
        "path": str(file_path),
        "exists": file_path.exists(),
        "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
        "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2) if file_path.exists() else 0,
    }
    
    if not file_path.exists():
        return info
    
    # Try to peek at the file to detect format
    try:
        # Try joblib first
        with open(file_path, 'rb') as f:
            # Read first few bytes to detect format
            header = f.read(100)
            info["format"] = "joblib" if b"joblib" in header or header.startswith(b"\x80\x02") else "pickle"
    except Exception:
        info["format"] = "unknown"
    
    return info


if __name__ == "__main__":
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description="Load and inspect pickle files")
    parser.add_argument("file_path", type=str, help="Path to pickle file")
    parser.add_argument("--inspect", action="store_true", help="Only inspect, don't load")
    parser.add_argument("--use-pickle", action="store_true", help="Force use of pickle instead of joblib")
    parser.add_argument("--show-array", action="store_true", help="Extract and show array values (for SegmentArray/dict objects)")
    
    args = parser.parse_args()
    
    if args.inspect:
        info = inspect_pickle_file(args.file_path)
        print("File Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        data = load_pickle_file(args.file_path, use_joblib=not args.use_pickle)
        print(f"\n✅ Successfully loaded pickle file!")
        print(f"Data type: {type(data)}")
        
        if hasattr(data, '__len__'):
            print(f"Length: {len(data)}")
        
        if hasattr(data, 'keys') and callable(data.keys):
            keys = list(data.keys())
            print(f"Keys: {keys[:10]}{'...' if len(keys) > 10 else ''}")
        
        if hasattr(data, 'shape'):
            print(f"Shape: {data.shape}")
        
        # Try to extract and show array if --show-array flag is used or if it's a SegmentArray
        if args.show_array or 'SegmentArray' in str(type(data)):
            try:
                # Try to use the training module's extract functions
                from LiverTumorSegmentation.components.training import PickleDataLoader
                
                # Try extracting as mask first (for GroundTruth)
                try:
                    arr = PickleDataLoader.extract_mask(data)
                    print(f"\n📊 Extracted Array (as mask):")
                    print(f"  Shape: {arr.shape}")
                    print(f"  Dtype: {arr.dtype}")
                    print(f"  Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.6f}")
                    print(f"  Unique values: {np.unique(arr)[:10]}{'...' if len(np.unique(arr)) > 10 else ''}")
                    
                    # Show array preview
                    if arr.ndim == 3:
                        print(f"\n  First slice ([:, :, 0]):")
                        print(arr[:, :, 0])
                        if arr.shape[2] > 1:
                            print(f"\n  Middle slice ([:, :, {arr.shape[2]//2}]):")
                            print(arr[:, :, arr.shape[2]//2])
                    elif arr.ndim == 2:
                        print(f"\n  Array values:")
                        print(arr)
                    else:
                        print(f"\n  Array values (first 100 elements):")
                        print(arr.flatten()[:100])
                except Exception as mask_err:
                    # Try extracting as array (for Predictions)
                    try:
                        arr = PickleDataLoader.extract_array(data)
                        print(f"\n📊 Extracted Array (as image/prediction):")
                        print(f"  Shape: {arr.shape}")
                        print(f"  Dtype: {arr.dtype}")
                        print(f"  Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.6f}")
                        
                        # Show array preview
                        if arr.ndim == 3:
                            print(f"\n  First slice ([:, :, 0]):")
                            print(arr[:, :, 0])
                            if arr.shape[2] > 1:
                                print(f"\n  Middle slice ([:, :, {arr.shape[2]//2}]):")
                                print(arr[:, :, arr.shape[2]//2])
                        elif arr.ndim == 2:
                            print(f"\n  Array values:")
                            print(arr)
                        else:
                            print(f"\n  Array values (first 100 elements):")
                            print(arr.flatten()[:100])
                    except Exception as array_err:
                        print(f"\n⚠️  Could not extract array using PickleDataLoader methods")
                        print(f"  Mask extraction error: {mask_err}")
                        print(f"  Array extraction error: {array_err}")
                        
                        # Fallback: try to convert SegmentArray to numpy directly
                        if hasattr(data, '__array__'):
                            try:
                                arr = np.array(data)
                                print(f"\n📊 Converted SegmentArray to numpy array:")
                                print(f"  Shape: {arr.shape}")
                                print(f"  Dtype: {arr.dtype}")
                                print(f"  Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.6f}")
                                
                                if arr.ndim == 3:
                                    print(f"\n  First slice ([:, :, 0]):")
                                    print(arr[:, :, 0])
                                elif arr.ndim == 2:
                                    print(f"\n  Array values:")
                                    print(arr[:20, :20] if arr.shape[0] > 20 else arr)
                                else:
                                    print(f"\n  Array values (first 100 elements):")
                                    print(arr.flatten()[:100])
                            except Exception as convert_err:
                                print(f"  Conversion error: {convert_err}")
            except ImportError:
                print(f"\n⚠️  Could not import PickleDataLoader. Install dependencies or use --inspect only.")
        
        # Show raw data preview if not showing array
        if not args.show_array and 'SegmentArray' not in str(type(data)):
            print(f"\nData preview:")
            print(data)

