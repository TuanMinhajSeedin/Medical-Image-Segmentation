# PowerShell script to download/extract a subset of the dataset
# Usage: .\scripts\download_subset.ps1

# Extract a small subset (50 samples) from existing zip
python -m src.LiverTumorSegmentation.data.download_subset `
    --mode extract `
    --zip-path artifacts/raw/medical-image-segmentation.zip `
    --output artifacts/data_subset `
    --task Task001_LiverTumor `
    --max-samples 50

Write-Host "Subset extraction complete! Check artifacts/data_subset/"






