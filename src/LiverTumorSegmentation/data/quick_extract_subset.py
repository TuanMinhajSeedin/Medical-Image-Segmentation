"""
Quick script to extract a subset from the downloaded zip file.
Run this to avoid working with the full 40GB dataset.

Usage:
    python quick_extract_subset.py
    python quick_extract_subset.py --samples 100
    python quick_extract_subset.py --ratio 0.15
"""
import argparse
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.LiverTumorSegmentation.data.download_subset import ZipSubsetExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract a subset from the medical image segmentation dataset"
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default="artifacts/raw/medical-image-segmentation.zip",
        help="Path to the downloaded zip file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/data_subset",
        help="Output directory for extracted subset"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Task001_LiverTumor",
        help="Task name (e.g., Task001_LiverTumor)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to extract (default: 50)"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Ratio of files to extract (0.0-1.0). Overrides --samples if set."
    )
    
    args = parser.parse_args()
    
    # Check if zip exists
    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        print(f"❌ Error: Zip file not found at {zip_path}")
        print(f"   Please download the dataset first or check the path.")
        return
    
    print(f"📦 Extracting subset from: {zip_path}")
    print(f"   Task: {args.task}")
    print(f"   Output: {args.output}")
    
    try:
        extractor = ZipSubsetExtractor(str(zip_path))
        
        if args.ratio:
            print(f"   Using ratio: {args.ratio * 100}%")
            extractor.extract_subset(
                output_path=args.output,
                task_name=args.task,
                sample_ratio=args.ratio
            )
        else:
            print(f"   Extracting {args.samples} samples")
            extractor.extract_subset(
                output_path=args.output,
                task_name=args.task,
                max_samples=args.samples
            )
        
        print(f"\n✅ Success! Subset extracted to: {args.output}")
        print(f"   Update your config.yaml to use: {args.output}/Task001_LiverTumor")
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        raise


if __name__ == "__main__":
    main()

