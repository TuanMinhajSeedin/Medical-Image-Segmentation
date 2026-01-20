"""
Script to download a subset of the Kaggle dataset instead of the full 40GB.
This allows for faster development and iteration while maintaining dataset structure.
"""
import os
import json
import zipfile
import shutil
from pathlib import Path
from typing import List, Optional
from kaggle.api.kaggle_api_extended import KaggleApi
from LiverTumorSegmentation import logger


class DatasetSubsetDownloader:
    """
    Downloads a subset of the medical image segmentation dataset from Kaggle.
    Supports downloading only specific tasks or a sample of files.
    """
    
    def __init__(self, kaggle_json_path: Optional[str] = None):
        """
        Initialize the downloader with Kaggle credentials.
        
        Args:
            kaggle_json_path: Path to kaggle.json with username and key.
                            If None, tries to find it in artifacts/ or project root.
        """
        # Find kaggle.json - check multiple locations
        if kaggle_json_path is None:
            # Get project root (assuming this file is in src/LiverTumorSegmentation/data/)
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            
            possible_paths = [
                "artifacts/kaggle.json",
                str(project_root / "artifacts" / "kaggle.json"),
                str(Path.cwd() / "artifacts" / "kaggle.json"),
                os.path.expanduser("~/.kaggle/kaggle.json"),  # Default Kaggle location
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    kaggle_json_path = path
                    logger.info(f"Found kaggle.json at: {path}")
                    break
            else:
                raise FileNotFoundError(
                    f"Could not find kaggle.json. Tried: {possible_paths}\n"
                    f"Please ensure kaggle.json exists in artifacts/ directory or ~/.kaggle/"
                )
        
        self.kaggle_json_path = kaggle_json_path
        self.api = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Kaggle API using credentials from JSON file."""
        try:
            # Resolve absolute path
            kaggle_path = Path(self.kaggle_json_path)
            if not kaggle_path.is_absolute():
                # Try relative to current working directory first
                if not kaggle_path.exists():
                    # Try relative to project root
                    current_file = Path(__file__)
                    project_root = current_file.parent.parent.parent.parent
                    kaggle_path = project_root / kaggle_path
            
            if not kaggle_path.exists():
                raise FileNotFoundError(f"kaggle.json not found at: {kaggle_path}")
            
            with open(kaggle_path, 'r') as f:
                creds = json.load(f)
            
            os.environ['KAGGLE_USERNAME'] = creds['username']
            os.environ['KAGGLE_KEY'] = creds['key']
            
            self.api = KaggleApi()
            self.api.authenticate()
            logger.info("Kaggle API authenticated successfully")
        except Exception as e:
            logger.error(f"Failed to authenticate with Kaggle: {e}")
            raise
    
    def download_dataset(self, 
                        dataset_name: str = "modaresimr/medical-image-segmentation",
                        output_path: str = "artifacts/raw",
                        unzip: bool = False,
                        subset_size: Optional[int] = None) -> str:
        """
        Download the full dataset from Kaggle.
        
        Note: Kaggle API downloads the full dataset (~40GB). To work with a subset,
        download as zip (unzip=False), then use ZipSubsetExtractor to extract only what you need.
        
        Args:
            dataset_name: Kaggle dataset identifier
            output_path: Where to save the downloaded files
            unzip: Whether to unzip the downloaded file (default: False to save space)
            subset_size: Not used (Kaggle API doesn't support partial downloads)
            
        Returns:
            Path to downloaded zip file or extracted directory
        """
        os.makedirs(output_path, exist_ok=True)
        
        logger.info(f"Downloading dataset: {dataset_name}")
        logger.info("⚠️  Note: Kaggle API downloads the full ~40GB dataset.")
        logger.info("   Recommendation: Set unzip=False, then use extract_subset() to get only what you need.")
        
        # Download the dataset
        self.api.dataset_download_files(
            dataset_name,
            path=output_path,
            unzip=unzip
        )
        
        # Find the downloaded zip file
        dataset_folder = dataset_name.split('/')[-1]
        zip_path = os.path.join(output_path, f"{dataset_folder}.zip")
        
        if os.path.exists(zip_path):
            logger.info(f"✅ Dataset downloaded to: {zip_path}")
            return zip_path
        else:
            # Check if it was extracted
            extracted_path = os.path.join(output_path, dataset_folder)
            if os.path.exists(extracted_path):
                logger.info(f"✅ Dataset extracted to: {extracted_path}")
                return extracted_path
            else:
                # Sometimes Kaggle downloads to a different location
                files = os.listdir(output_path)
                zip_files = [f for f in files if f.endswith('.zip')]
                if zip_files:
                    zip_path = os.path.join(output_path, zip_files[0])
                    logger.info(f"✅ Dataset downloaded to: {zip_path}")
                    return zip_path
                else:
                    logger.warning(f"Could not find downloaded file. Check: {output_path}")
                    return output_path
    
    def download_and_extract_subset(self,
                                   dataset_name: str = "modaresimr/medical-image-segmentation",
                                   output_path: str = "artifacts/data_subset",
                                   task_name: str = "Task001_LiverTumor",
                                   max_samples: int = 50,
                                   keep_zip: bool = True,
                                   cleanup_full_extract: bool = True) -> str:
        """
        Download full dataset from Kaggle and immediately extract only a subset.
        This minimizes disk usage by extracting only what you need.
        
        Workflow:
        1. Download full dataset as zip (~40GB)
        2. Extract only the subset you need (~500MB-1GB)
        3. Optionally delete the zip to save space
        4. Optionally clean up any full extraction
        
        Args:
            dataset_name: Kaggle dataset identifier
            output_path: Where to save the extracted subset
            task_name: Which task to extract (e.g., "Task001_LiverTumor")
            max_samples: Number of samples to extract
            keep_zip: Whether to keep the zip file after extraction
            cleanup_full_extract: Whether to clean up if full dataset was extracted
            
        Returns:
            Path to extracted subset directory
        """
        logger.info("=" * 60)
        logger.info("Downloading from Kaggle and extracting subset...")
        logger.info("=" * 60)
        
        # Step 1: Download as zip (don't unzip to save space)
        temp_download_path = "artifacts/raw"
        os.makedirs(temp_download_path, exist_ok=True)
        
        logger.info("Step 1/3: Downloading dataset from Kaggle (this may take a while for ~40GB)...")
        zip_path = self.download_dataset(
            dataset_name=dataset_name,
            output_path=temp_download_path,
            unzip=False  # Download as zip
        )
        
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Download failed. Zip not found at: {zip_path}")
        
        # Step 2: Extract subset
        logger.info(f"Step 2/3: Extracting {max_samples} samples from zip...")
        extractor = ZipSubsetExtractor(zip_path)
        subset_path = extractor.extract_subset(
            output_path=output_path,
            task_name=task_name,
            max_samples=max_samples
        )
        
        # Step 3: Cleanup
        logger.info("Step 3/3: Cleaning up...")
        
        if not keep_zip:
            logger.info(f"Deleting zip file to save space: {zip_path}")
            os.remove(zip_path)
        else:
            logger.info(f"Keeping zip file at: {zip_path}")
        
        if cleanup_full_extract:
            # Check if full dataset was extracted somewhere
            dataset_folder = dataset_name.split('/')[-1]
            full_extract_path = os.path.join(temp_download_path, dataset_folder)
            if os.path.exists(full_extract_path) and os.path.isdir(full_extract_path):
                logger.info(f"Removing full extracted dataset: {full_extract_path}")
                shutil.rmtree(full_extract_path, ignore_errors=True)
        
        logger.info("=" * 60)
        logger.info(f"✅ Complete! Subset extracted to: {subset_path}")
        logger.info(f"   Size: ~{max_samples * 10}MB (vs ~40GB full dataset)")
        logger.info("=" * 60)
        
        return subset_path


class ZipSubsetExtractor:
    """
    Extracts only a subset of files from a large zip archive.
    Useful when you have the full zip but want to work with a sample.
    """
    
    def __init__(self, zip_path: str):
        """
        Initialize with path to zip file.
        
        Args:
            zip_path: Path to the zip file
        """
        self.zip_path = zip_path
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    def list_contents(self, max_items: int = 50) -> List[str]:
        """
        List contents of the zip file.
        
        Args:
            max_items: Maximum number of items to list
            
        Returns:
            List of file paths in the zip
        """
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            logger.info(f"Total files in zip: {len(file_list)}")
            return file_list[:max_items]
    
    def extract_subset(self,
                      output_path: str,
                      task_name: str = "Task001_LiverTumor",
                      max_samples: Optional[int] = None,
                      sample_ratio: float = 0.1) -> str:
        """
        Extract only a subset of files from the zip.
        
        Args:
            output_path: Where to extract files
            task_name: Which task folder to extract (e.g., "Task001_LiverTumor")
            max_samples: Maximum number of image pairs to extract
            sample_ratio: Ratio of files to extract (0.1 = 10%)
            
        Returns:
            Path to extracted directory
        """
        os.makedirs(output_path, exist_ok=True)
        
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            # Get all files
            all_files = zip_ref.namelist()
            
            # Filter files for the specified task
            task_files = [f for f in all_files if task_name in f]
            
            logger.info(f"Found {len(task_files)} files for {task_name}")
            
            # Determine how many to extract
            if max_samples:
                # Extract first N samples (assuming paired images/masks)
                # Group by case ID if possible
                files_to_extract = task_files[:max_samples * 2]  # *2 for images + masks
            else:
                # Extract based on ratio
                num_to_extract = int(len(task_files) * sample_ratio)
                files_to_extract = task_files[:num_to_extract]
            
            logger.info(f"Extracting {len(files_to_extract)} files...")
            
            # Extract files
            for file_path in files_to_extract:
                try:
                    zip_ref.extract(file_path, output_path)
                except Exception as e:
                    logger.warning(f"Failed to extract {file_path}: {e}")
            
            logger.info(f"Extraction complete. Files saved to: {output_path}")
            return output_path
    
    def extract_by_pattern(self,
                          output_path: str,
                          include_patterns: List[str],
                          exclude_patterns: Optional[List[str]] = None) -> str:
        """
        Extract files matching specific patterns.
        
        Args:
            output_path: Where to extract files
            include_patterns: List of patterns to include (e.g., ["imagesTr", "labelsTr"])
            exclude_patterns: List of patterns to exclude
            
        Returns:
            Path to extracted directory
        """
        os.makedirs(output_path, exist_ok=True)
        
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            
            # Filter files
            filtered_files = []
            for file_path in all_files:
                # Check include patterns
                if any(pattern in file_path for pattern in include_patterns):
                    # Check exclude patterns
                    if exclude_patterns:
                        if any(pattern in file_path for pattern in exclude_patterns):
                            continue
                    filtered_files.append(file_path)
            
            logger.info(f"Extracting {len(filtered_files)} files matching patterns: {include_patterns}")
            
            for file_path in filtered_files:
                try:
                    zip_ref.extract(file_path, output_path)
                except Exception as e:
                    logger.warning(f"Failed to extract {file_path}: {e}")
            
            return output_path


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download subset of medical image segmentation dataset from Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from Kaggle and extract 50 samples (recommended)
  python download_subset.py --mode download-subset --samples 50
  
  # Extract from existing zip file
  python download_subset.py --mode extract --zip-path artifacts/raw/medical-image-segmentation.zip --samples 50
  
  # Download full dataset only
  python download_subset.py --mode download
        """
    )
    parser.add_argument("--mode", 
                       choices=["download", "extract", "download-subset"], 
                       default="download-subset",
                       help="Mode: 'download'=download full dataset, 'extract'=extract from zip, 'download-subset'=download and extract subset")
    parser.add_argument("--zip-path", type=str, default="artifacts/raw/medical-image-segmentation.zip",
                       help="Path to zip file (for extract mode)")
    parser.add_argument("--output", type=str, default="artifacts/data_subset",
                       help="Output directory")
    parser.add_argument("--task", type=str, default="Task001_LiverTumor",
                       help="Task name to extract")
    parser.add_argument("--max-samples", type=int, default=50,
                       help="Maximum number of samples to extract")
    parser.add_argument("--samples", type=int, default=None,
                       help="Alias for --max-samples")
    parser.add_argument("--ratio", type=float, default=None,
                       help="Ratio of files to extract (0.0-1.0). Overrides --max-samples if set.")
    parser.add_argument("--keep-zip", action="store_true",
                       help="Keep the zip file after extraction (default: delete to save space)")
    parser.add_argument("--kaggle-json", type=str, default=None,
                       help="Path to kaggle.json (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Handle --samples as alias for --max-samples
    max_samples = args.samples if args.samples is not None else args.max_samples
    
    if args.mode == "extract":
        # Extract from existing zip - no Kaggle auth needed
        extractor = ZipSubsetExtractor(args.zip_path)
        extractor.extract_subset(
            output_path=args.output,
            task_name=args.task,
            max_samples=max_samples if args.ratio is None else None,
            sample_ratio=args.ratio if args.ratio is not None else 0.1
        )
    elif args.mode == "download-subset":
        # Download from Kaggle and extract subset
        downloader = DatasetSubsetDownloader(kaggle_json_path=args.kaggle_json)
        downloader.download_and_extract_subset(
            output_path=args.output,
            task_name=args.task,
            max_samples=max_samples,
            keep_zip=args.keep_zip
        )
    else:
        # Download full dataset only
        downloader = DatasetSubsetDownloader(kaggle_json_path=args.kaggle_json)
        downloader.download_dataset(output_path=args.output, unzip=False)

