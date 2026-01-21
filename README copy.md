# Medical-Image-Segmentation

This repo focuses on Medical Image Segmentation, identifying and separating organs or disease regions (e.g., tumors) from medical images such as MRI or CT scans using deep learning models like U-Net. It enables accurate analysis for diagnosis and treatment, evaluated using Dice and IoU metrics.

---

## 📊 Diagrams

*Add project architecture diagrams here:*
- Data pipeline flow
- Model architecture (U-Net)
- Training workflow
- Evaluation pipeline

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.9
- pip
- Git
- (Optional) DVC for data versioning

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Medical-Image-Segmentation
```

### Step 2: Create Virtual Environment
```bash
conda create -n medical-seg python=3.9 -y
conda activate medical-seg
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### Step 4: DVC Setup (Optional)
```bash
dvc init
```

---

## 🎯 How to Run

### Training
Train a model using a specific experiment configuration:
```bash
python train.py --config configs/exp_001.yaml
```

Train with default params:
```bash
python train.py
```

### Evaluation
Evaluate a trained model:
```bash
python eval.py --config configs/exp_001.yaml
```

Evaluate a specific checkpoint:
```bash
python eval.py --config configs/exp_001.yaml --ckpt artifacts/training/model.h5
```

### Inference/Prediction
Run inference on a single pickle file:
```bash
python infer.py --input artifacts/data_ingestion/data/Predictions/Dice/3.pkl --threshold 0.5 --save
```

Quick test with app.py:
```bash
python app.py
```

Batch predictions:
```bash
python test_predictions.py --predictions_subdir Dice --sample_ids 3 4 6
```

### Run All Experiments
Run all experiment configurations and compare results:
```bash
python run_experiments.py
```

### Run Complete Pipeline
Run all stages (data ingestion → base model → training → evaluation):
```bash
python main.py
```

---

## 🔧 Workflows

When adding new features or modifying the pipeline, follow this workflow:

1. **Update `configs/config.yaml`**
   - Add new configuration sections
   - Update paths and settings

2. **Update `configs/params.yaml` [Optional]**
   - Add/modify hyperparameters
   - Update training parameters

3. **Update the Entity (`src/LiverTumorSegmentation/entity/config_entity.py`)**
   - Define new configuration dataclasses
   - Add new entity classes if needed

4. **Update the Configuration Manager (`src/LiverTumorSegmentation/config/configuration.py`)**
   - Add methods to load new configurations
   - Update configuration parsing logic

5. **Update the Components (`src/LiverTumorSegmentation/components/`)**
   - Modify existing components
   - Create new component classes if needed

6. **Update the Pipeline (`src/LiverTumorSegmentation/pipeline/`)**
   - Update pipeline stages
   - Add new pipeline steps

7. **Update `main.py`**
   - Add new pipeline stages
   - Update execution flow

8. **Update `dvc.yaml`**
   - Add new DVC stages
   - Update dependencies and outputs

9. **MLflow**
   - Configure MLflow tracking URI
   - Test MLflow logging

10. **Documentation**
    - Update README
    - Add docstrings
    - Update experiment results

---

## 📈 Experiment Results

### Results Table

| Experiment | Key Change | Val Dice | Val IoU |
|------------|------------|----------|---------|
| `exp_000` | baseline UNet + BCE/Dice | 0.0867 | 0.0509 |
| `exp_001` | + augmentations + LR schedule | 0.0884 | 0.0504 |
| `exp_002` | loss tweak / encoder (64 filters) | 0.0508 | 0.0268 |

### Experiment Configurations

- **exp_000**: Baseline configuration with no augmentation, standard UNet (32 filters)
- **exp_001**: Added data augmentation and learning rate scheduling (LR: 0.002)
- **exp_002**: Modified encoder with increased filters (64) and enhanced loss tuning

*Note: Results shown are from validation sets. Full results are available in `experiment_results/all_results.json`*

---

## 📦 MLflow

### MLflow Tutorial

MLflow is used for experiment tracking, model logging, and versioning.

### Local MLflow (Default)

The project is configured to use local MLflow tracking by default.

#### Start MLflow UI
```bash
mlflow ui
```

Or specify a port:
```bash
mlflow ui --port 5000
```

Access the UI at: `http://localhost:5000`

#### Configuration
Local MLflow stores runs in `./mlruns/` directory. Configure in `configs/config.yaml`:
```yaml
evaluation:
  mlflow_uri: ./mlruns  # Local file-based tracking
```

### DagsHub MLflow (Optional)

To use DagsHub for remote MLflow tracking:

#### 1. Update `configs/config.yaml`
```yaml
evaluation:
  mlflow_uri: https://dagshub.com/TuanMinhajSeedin/Medical-Image-Segmentation.mlflow
```

#### 2. Set Environment Variables

**On Linux/Mac:**
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/TuanMinhajSeedin/Medical-Image-Segmentation.mlflow
export MLFLOW_TRACKING_USERNAME=your-username
export MLFLOW_TRACKING_PASSWORD=your-token
```

**On Windows (PowerShell):**
```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/TuanMinhajSeedin/Medical-Image-Segmentation.mlflow"
$env:MLFLOW_TRACKING_USERNAME="your-username"
$env:MLFLOW_TRACKING_PASSWORD="your-token"
```

**On Windows (CMD):**
```cmd
set MLFLOW_TRACKING_URI=https://dagshub.com/TuanMinhajSeedin/Medical-Image-Segmentation.mlflow
set MLFLOW_TRACKING_USERNAME=your-username
set MLFLOW_TRACKING_PASSWORD=your-token
```

#### 3. Run Your Scripts
```bash
python train.py --config configs/exp_001.yaml
python eval.py --config configs/exp_001.yaml
```

---

## 📁 DVC Commands

DVC (Data Version Control) is used for lightweight experiment tracking and pipeline orchestration.

### Initialize DVC
```bash
dvc init
```

### Run DVC Pipeline
Reproduce all stages defined in `dvc.yaml`:
```bash
dvc repro
```

### Run Specific Stage
```bash
dvc repro training
dvc repro evaluation
```

### View Pipeline DAG
```bash
dvc dag
```

### Check Pipeline Status
```bash
dvc status
```

### Add Data/Models to DVC
```bash
dvc add artifacts/training/model.h5
```

---

## 🛠️ About MLflow & DVC

### MLflow

**Why MLflow?**
- ✅ **Production Grade**: Enterprise-ready experiment tracking
- ✅ **Trace all experiments**: Complete experiment history and reproducibility
- ✅ **Logging & tagging**: Comprehensive model versioning and metadata
- ✅ **Model Registry**: Centralized model management
- ✅ **UI Dashboard**: Visual experiment comparison and analysis

**Best For**: Production environments, team collaboration, model registry

### DVC

**Why DVC?**
- ✅ **Lightweight**: Minimal overhead, perfect for POC
- ✅ **Lightweight experiments tracker**: Simple experiment versioning
- ✅ **Pipeline Orchestration**: Define and run data pipelines
- ✅ **Data Versioning**: Track data changes alongside code

**Best For**: 
- Proof of Concept (POC) projects
- Data pipeline orchestration
- Lightweight experiment tracking
- Development and research phases

### Using Both Together

This project uses both tools:
- **DVC**: For pipeline orchestration and data versioning
- **MLflow**: For experiment tracking, model logging, and metrics comparison

---

## 📂 Project Structure

```
Medical-Image-Segmentation/
├── artifacts/          # Model outputs, data, predictions
├── configs/            # Configuration files (config.yaml, params.yaml, exp_*.yaml)
├── docs/              # Documentation
├── experiment_results/ # Experiment comparison results
├── logs/              # Application logs
├── mlruns/            # MLflow local tracking data
├── research/          # Jupyter notebooks and research scripts
├── scripts/           # Utility scripts
├── src/
│   └── LiverTumorSegmentation/
│       ├── components/    # Core components (data, training, evaluation)
│       ├── config/        # Configuration management
│       ├── constants/     # Constants and paths
│       ├── entity/        # Configuration entities
│       ├── models/        # Model definitions
│       ├── pipeline/      # Pipeline stages
│       └── utils/         # Utility functions
├── app.py             # Quick inference script
├── eval.py            # Evaluation entry point
├── infer.py           # Inference CLI
├── main.py            # Complete pipeline runner
├── run_experiments.py # Batch experiment runner
├── test_predictions.py # Batch prediction tester
├── train.py           # Training entry point
├── dvc.yaml           # DVC pipeline definition
├── requirements.txt   # Python dependencies
└── setup.py           # Package setup
```

---

## 📝 Configuration

### Main Config (`configs/config.yaml`)
Contains paths and settings for:
- Data ingestion
- Base model preparation
- Training
- Evaluation
- Prediction

### Parameters (`configs/params.yaml`)
Hyperparameters:
- Image size
- Batch size
- Epochs
- Learning rate
- Base filters
- Augmentation flags

### Experiment Configs (`configs/exp_*.yaml`)
Individual experiment configurations that override default parameters.

---

## 🔍 Troubleshooting

### MLflow Connection Issues
- Check MLflow URI in `configs/config.yaml`
- Verify environment variables are set (if using DagsHub)
- Ensure MLflow UI is running: `mlflow ui`

### DVC Pipeline Issues
- Run `dvc status` to check pipeline state
- Use `dvc repro` to regenerate outputs
- Check `dvc.lock` for dependency versions

### Model Not Found
- Ensure training completed successfully
- Check model path in `configs/config.yaml`
- Verify `artifacts/training/model.h5` exists

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

## 👤 Author

**TuanMinhajSeedin**
- Email: tuanminhajseedin@gmail.com
- GitHub: [TuanMinhajSeedin](https://github.com/TuanMinhajSeedin)

---

## 🙏 Acknowledgments

- U-Net architecture for medical image segmentation
- TensorFlow/Keras for deep learning framework
- MLflow for experiment tracking
- DVC for data versioning
