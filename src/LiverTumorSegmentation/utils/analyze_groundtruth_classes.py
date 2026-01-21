"""
Analyze all GroundTruth pickle files and count classes in each.
"""
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from LiverTumorSegmentation.utils.load_pickle import load_pickle_file
from LiverTumorSegmentation.components.training import PickleDataLoader

def analyze_groundtruth_files(gt_dir: Path):
    """Analyze all .pkl files in GroundTruth directory."""
    gt_dir = Path(gt_dir)
    
    if not gt_dir.exists():
        print(f"Error: Directory not found: {gt_dir}")
        return
    
    pkl_files = sorted(gt_dir.glob("*.pkl"))
    print(f"Found {len(pkl_files)} pickle files in {gt_dir}\n")
    print("=" * 80)
    print(f"{'File':<10} {'Status':<12} {'Shape':<20} {'Classes':<15} {'Class Counts'}")
    print("=" * 80)
    
    all_classes = set()
    file_stats = []
    
    for pkl_file in pkl_files:
        file_id = pkl_file.stem
        try:
            # Load pickle file
            data = load_pickle_file(pkl_file)
            
            # Extract mask
            try:
                arr = PickleDataLoader.extract_mask(data)
                unique_classes, counts = np.unique(arr, return_counts=True)
                
                # Store stats
                class_dict = dict(zip(unique_classes.tolist(), counts.tolist()))
                all_classes.update(unique_classes.tolist())
                
                file_stats.append({
                    'file': file_id,
                    'shape': arr.shape,
                    'classes': sorted(unique_classes.tolist()),
                    'counts': class_dict,
                    'status': 'OK'
                })
                
                # Format class counts string
                class_counts_str = ", ".join([f"{c}: {cnt}" for c, cnt in class_dict.items()])
                print(f"{file_id:<10} {'OK':<12} {str(arr.shape):<20} {str(sorted(unique_classes.tolist())):<15} {class_counts_str}")
                
            except Exception as e:
                file_stats.append({
                    'file': file_id,
                    'shape': None,
                    'classes': [],
                    'counts': {},
                    'status': f'Extract Error: {str(e)[:30]}'
                })
                print(f"{file_id:<10} {'Extract Error':<12} {'N/A':<20} {'N/A':<15} {str(e)[:50]}")
                
        except Exception as e:
            file_stats.append({
                'file': file_id,
                'shape': None,
                'classes': [],
                'counts': {},
                'status': f'Load Error: {str(e)[:30]}'
            })
            print(f"{file_id:<10} {'Load Error':<12} {'N/A':<20} {'N/A':<15} {str(e)[:50]}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    all_classes_sorted = sorted(all_classes)
    print(f"\nAll unique classes found across all files: {all_classes_sorted}")
    
    # Count files with each class
    class_file_counts = {cls: 0 for cls in all_classes_sorted}
    for stat in file_stats:
        for cls in stat['classes']:
            class_file_counts[cls] += 1
    
    print(f"\nNumber of files containing each class:")
    for cls in all_classes_sorted:
        print(f"  Class {cls}: {class_file_counts[cls]} files")
    
    # Aggregate counts for each class
    class_total_counts = {cls: 0 for cls in all_classes_sorted}
    for stat in file_stats:
        for cls, count in stat['counts'].items():
            class_total_counts[cls] += count
    
    print(f"\nTotal pixel/voxel counts for each class (across all files):")
    for cls in all_classes_sorted:
        print(f"  Class {cls}: {class_total_counts[cls]:,} pixels/voxels")
    
    # Files with multiple classes
    multi_class_files = [stat for stat in file_stats if len(stat['classes']) > 1]
    if multi_class_files:
        print(f"\nFiles with multiple classes ({len(multi_class_files)} files):")
        for stat in multi_class_files:
            print(f"  {stat['file']}: Classes {stat['classes']} - Counts {stat['counts']}")
    else:
        print(f"\nAll files contain only single class (binary segmentation)")
    
    # Files with errors
    error_files = [stat for stat in file_stats if 'Error' in stat['status']]
    if error_files:
        print(f"\nFiles with errors ({len(error_files)} files):")
        for stat in error_files:
            print(f"  {stat['file']}: {stat['status']}")
    
    return file_stats, all_classes_sorted

if __name__ == "__main__":
    gt_dir = Path("artifacts/data_ingestion/data/GroundTruth")
    analyze_groundtruth_files(gt_dir)

