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
    
    parser = argparse.ArgumentParser(description="Load and inspect pickle files")
    parser.add_argument("file_path", type=str, help="Path to pickle file")
    parser.add_argument("--inspect", action="store_true", help="Only inspect, don't load")
    parser.add_argument("--use-pickle", action="store_true", help="Force use of pickle instead of joblib")
    
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
            print(f"Keys: {list(data.keys())[:10]}...")  # Show first 10 keys
        
        if hasattr(data, 'shape'):
            print(f"Shape: {data.shape}")
        
        print(f"\nData preview:")
        print(data)

