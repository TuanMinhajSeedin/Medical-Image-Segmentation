"""
Download dataset from Kaggle and extract only a subset.
This avoids storing the full 40GB extracted dataset.

Usage:
    # Download and extract 50 samples (recommended)
    python download_from_kaggle.py --samples 50
    
    # Download and extract 100 samples, keep the zip file
    python download_from_kaggle.py --samples 100 --keep-zip
    
    # Just download the full dataset (don't extract)
    python download_from_kaggle.py --mode download
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from LiverTumorSegmentation.data.download_subset import DatasetSubsetDownloader, ZipSubsetExtractor


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download medical image segmentation dataset from Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and extract 50 samples (recommended - saves space)
  python download_from_kaggle.py --samples 50
  
  # Download and extract 100 samples, keep zip file
  python download_from_kaggle.py --samples 100 --keep-zip
  
  # Just download full dataset (40GB zip, no extraction)
  python download_from_kaggle.py --mode download
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["download", "download-subset"],
        default="download-subset",
        help="'download'=download only, 'download-subset'=download and extract subset"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to extract (default: 50)"
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
        "--keep-zip",
        action="store_true",
        help="Keep the zip file after extraction (default: delete to save space)"
    )
    parser.add_argument(
        "--kaggle-json",
        type=str,
        default=None,
        help="Path to kaggle.json (default: auto-detect from artifacts/)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Medical Image Segmentation Dataset Downloader")
    print("=" * 70)
    
    try:
        if args.mode == "download-subset":
            print(f"\n📥 Downloading from Kaggle and extracting {args.samples} samples...")
            print(f"   This will download ~40GB zip, then extract only {args.samples} samples (~{args.samples * 10}MB)")
            print(f"   The zip will be {'kept' if args.keep_zip else 'deleted'} after extraction.\n")
            
            downloader = DatasetSubsetDownloader(kaggle_json_path=args.kaggle_json)
            subset_path = downloader.download_and_extract_subset(
                output_path=args.output,
                task_name=args.task,
                max_samples=args.samples,
                keep_zip=args.keep_zip
            )
            
            print(f"\n✅ Success! Your subset is ready at: {subset_path}")
            print(f"   Update your config.yaml to use this path for training.")
            
        else:
            print("\n📥 Downloading full dataset from Kaggle (~40GB)...")
            print("   Note: This downloads the full dataset. Use --mode download-subset to extract only a subset.\n")
            
            downloader = DatasetSubsetDownloader(kaggle_json_path=args.kaggle_json)
            zip_path = downloader.download_dataset(
                output_path="artifacts/raw",
                unzip=False
            )
            
            print(f"\n✅ Download complete! Zip file at: {zip_path}")
            print(f"   To extract a subset, run:")
            print(f"   python download_from_kaggle.py --mode download-subset --samples 50")
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure you have kaggle.json in artifacts/ directory with format:")
        print('   {"username": "your_username", "key": "your_api_key"}')
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

