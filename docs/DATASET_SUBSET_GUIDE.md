# Dataset Subset Guide: Working with 40GB Dataset Efficiently

## Problem
The Kaggle medical image segmentation dataset is ~40GB, which is too large for:
- Quick development iterations
- Limited storage space
- Faster experimentation

## Solution Strategies

### Strategy 1: Extract Subset from Existing Zip (Recommended) ⭐

If you already have the zip file downloaded, extract only a sample:

```bash
# Extract 50 samples (images + masks)
python -m src.LiverTumorSegmentation.data.download_subset \
    --mode extract \
    --zip-path artifacts/raw/medical-image-segmentation.zip \
    --output artifacts/data_subset \
    --task Task001_LiverTumor \
    --max-samples 50
```

**Benefits:**
- No re-download needed
- Fast extraction (~1-2 minutes)
- Maintains dataset structure
- ~500MB-1GB instead of 40GB

### Strategy 2: Download Only Specific Task Folder

The dataset contains multiple tasks. Focus on one task (e.g., Task001_LiverTumor):

```python
from src.LiverTumorSegmentation.data.download_subset import ZipSubsetExtractor

extractor = ZipSubsetExtractor("artifacts/raw/medical-image-segmentation.zip")
extractor.extract_by_pattern(
    output_path="artifacts/data_subset",
    include_patterns=["Task001_LiverTumor/imagesTr", "Task001_LiverTumor/labelsTr"]
)
```

### Strategy 3: Use Sample Ratio

Extract 10% of the dataset:

```bash
python -m src.LiverTumorSegmentation.data.download_subset \
    --mode extract \
    --zip-path artifacts/raw/medical-image-segmentation.zip \
    --output artifacts/data_subset \
    --ratio 0.1
```

### Strategy 4: Progressive Download (For First-Time Setup)

If you haven't downloaded yet, you can still download the full dataset but work with subsets:

1. Download full dataset (one-time, ~40GB):
   ```python
   from src.LiverTumorSegmentation.data.download_subset import DatasetSubsetDownloader
   
   downloader = DatasetSubsetDownloader()
   downloader.download_dataset(
       output_path="artifacts/raw",
       unzip=False  # Keep as zip to save space
   )
   ```

2. Extract subset as needed:
   ```bash
   python -m src.LiverTumorSegmentation.data.download_subset --mode extract --max-samples 50
   ```

## Recommended Workflow for Assignment

### Phase 1: Development (Use Subset)
- Extract 50-100 samples for initial development
- Train/test your pipeline
- Iterate quickly on model architecture

### Phase 2: Validation (Use Larger Subset)
- Extract 200-500 samples
- Validate your approach works at scale
- Fine-tune hyperparameters

### Phase 3: Final Training (Optional - Full Dataset)
- If time permits, use full dataset
- Or document that subset training is sufficient

## Configuration

Update your `configs/config.yaml` to point to the subset:

```yaml
data:
  dataset_path: "artifacts/data_subset/Task001_LiverTumor"
  use_subset: true
  subset_size: 50
```

## Size Estimates

| Samples | Approximate Size | Use Case |
|---------|-----------------|----------|
| 10-20   | ~100-200 MB     | Quick testing |
| 50      | ~500 MB         | Development |
| 100     | ~1 GB           | Validation |
| 200+    | ~2-5 GB         | Full training |
| Full    | ~40 GB          | Production |

## Tips

1. **Use DVC**: Track your subset with DVC so you can version it:
   ```bash
   dvc add artifacts/data_subset
   ```

2. **Document Your Choice**: In your README, explain:
   - Why you used a subset
   - How many samples
   - How you ensured it's representative

3. **Stratified Sampling**: If possible, ensure your subset includes:
   - Different image sizes
   - Various tumor sizes
   - Different scan qualities

4. **Lazy Loading**: Design your data loader to work with any dataset size:
   ```python
   # Your data loader should work with both subset and full dataset
   dataset_path = config.data.dataset_path  # Can be subset or full
   ```

## Example: Complete Setup

```bash
# 1. Extract subset (if you have the zip)
python -m src.LiverTumorSegmentation.data.download_subset \
    --mode extract \
    --max-samples 50 \
    --output artifacts/data_subset

# 2. Update config to use subset
# Edit configs/config.yaml: dataset_path = "artifacts/data_subset/..."

# 3. Train with subset
python train.py --config configs/exp_001.yaml

# 4. When ready, extract more samples
python -m src.LiverTumorSegmentation.data.download_subset \
    --mode extract \
    --max-samples 200 \
    --output artifacts/data_subset_large
```

## Justification for Evaluators

**Why use a subset?**
- Assignment focuses on MLOps pipeline quality, not dataset size
- Demonstrates efficient data management practices
- Shows understanding of iterative development
- Faster iteration = more experiments = better MLOps demonstration

**How to ensure quality?**
- Use stratified sampling to maintain dataset characteristics
- Document subset size and sampling strategy
- Show that pipeline scales (works with 50 or 5000 samples)
- Compare results across different subset sizes




