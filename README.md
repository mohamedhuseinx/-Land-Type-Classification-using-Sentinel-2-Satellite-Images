# 🛰️ Land Type Classification Using Sentinel-2 Satellite Images

**DEPI Project 6** — Digital Egypt Pioneers Initiative  
Deep Neural Network (CNN) to classify land types from ESA Sentinel-2 multispectral satellite imagery.

**Classes**: Agriculture, Water, Urban, Desert, Roads, Trees  
**Best Model**: Custom CNN — **97.23% test accuracy**, **0.9972 Macro AUC**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
   - [EDA & PCA Analysis](#1-eda--pca-analysis)
   - [Training](#2-training)
   - [Evaluation](#3-evaluation)
   - [API Server](#4-api-server)
6. [Results](#results)
7. [Model Architecture](#model-architecture)
8. [Deployment](#deployment)
9. [Project Roadmap](#project-roadmap)

---

## Project Overview

This project builds and deploys a deep learning model to classify land types from satellite imagery. It uses the **EuroSAT** dataset (27,000 Sentinel-2 RGB patches) supplemented with custom desert patches from Egypt's Western Desert.

### Business Applications for Egypt
- **Urban Planning**: Monitor informal settlements and urban sprawl around Cairo
- **Agriculture**: Track Nile Delta cropland health and irrigation optimization
- **Environment**: Desertification early warning, water body monitoring (Lake Nasser)
- **Infrastructure**: Road network mapping and land registry

### Architecture Pipeline

```
Sentinel-2 / EuroSAT → Preprocessing → EDA & PCA → CNN Training → Evaluation → FastAPI Deployment
```

---

## Dataset

### EuroSAT (Primary Source)
- 27,000 geo-referenced Sentinel-2 patches (64×64 px RGB)
- 10 original EuroSAT classes mapped to 6 target classes
- Downloaded from the EuroSAT TFRecord dataset

### Custom Desert Supplement
- ~4,000 desert patches from Egypt's Western Desert (27.0–27.5°N, 27.0–27.5°E)
- Downloaded from ESRI World Imagery at zoom level 14
- Sliced into 64×64 patches to match EuroSAT patch size

### Class Mapping

| EuroSAT Class | Mapped Target | Count (Train) |
|---------------|--------------|:------------:|
| AnnualCrop, PermanentCrop, Pasture | Agriculture | 5,250 |
| Forest, HerbaceousVegetation | Trees | 4,200 |
| River, SeaLake | Water | 3,850 |
| Residential, Industrial | Urban | 3,850 |
| — (Custom) | Desert | 2,800 |
| Highway | Roads | 1,750 |

**Total**: 31,000 images (train: 21,700 / val: 4,650 / test: 4,650)

---

## Project Structure

```
sentinel2-land-classification/
├── data/
│   ├── raw/                    # Class-organized image patches (6 folders)
│   ├── splits/                 # train.csv, val.csv, test.csv
├── src/
│   ├── __init__.py
│   ├── data_loading.py         # PyTorch Dataset + DataLoaders + augmentations
│   ├── preprocessing.py        # Train/val/test split creation
│   ├── eda.py                  # Exploratory data analysis (class dist, histograms)
│   ├── pca_analysis.py         # PCA dimensionality reduction & visualization
│   ├── model.py                # Custom CNN + ResNet50 architectures
│   ├── train.py                # Two-phase training pipeline
│   ├── evaluate.py             # Confusion matrix, ROC, Grad-CAM
│   ├── extract_eurosat.py      # EuroSAT TFRecord extraction
│   ├── download_desert.py      # Desert patch downloader
│   └── verify_milestone1.py    # Data pipeline verification
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI inference server
├── models/
│   └── checkpoints/            # Saved model weights
├── outputs/
│   ├── evaluation/             # Confusion matrix, ROC curves
│   ├── pca_*.png               # PCA analysis plots
│   └── verification_batch.png  # Sample augmented batch
├── logs/                       # TensorBoard event files
├── requirements.txt
├── Dockerfile
├── depi_project_roadmap.md     # Full project plan & report structure
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
# Navigate to project
cd "C:\DEPI project"

# Create virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. EDA & PCA Analysis

```bash
# Run exploratory data analysis (class distributions, pixel histograms)
python src/eda.py

# Run PCA analysis (explained variance, class separability)
python src/pca_analysis.py
```

Outputs saved to `outputs/`:
- `class_distribution.png`
- `pixel_distributions.png`
- `channel_correlations.png`
- `pca_explained_variance.png`
- `pca_scatter.png`

### 2. Training

```bash
# Train Custom CNN baseline (faster, ~30 min on CPU)
python src/train.py --arch custom_cnn --phase1_epochs 30 --phase2_epochs 0

# Train ResNet50 with two-phase protocol (1-2 hours on CPU)
python src/train.py --arch resnet50 --phase1_epochs 15 --phase2_epochs 30
```

**Arguments**:
| Flag | Default | Description |
|------|---------|-------------|
| `--arch` | `resnet50` | Model architecture: `custom_cnn` or `resnet50` |
| `--phase1_epochs` | `15` | Epochs with base frozen (head training) |
| `--phase2_epochs` | `50` | Epochs with fine-tuning (unfrozen layers) |

Training produces:
- Model checkpoints in `models/checkpoints/`
- TensorBoard logs in `logs/`
- TensorBoard: `tensorboard --logdir logs`

### 3. Evaluation

```bash
# Evaluate Custom CNN
python src/evaluate.py --arch custom_cnn --checkpoint models/checkpoints/CustomCNN_best.pth

# Evaluate ResNet50
python src/evaluate.py --arch resnet50 --checkpoint models/resnet50_final.pth
```

Outputs saved to `outputs/evaluation/`:
- `confusion_matrix.png` — Counts and normalized heatmaps
- `roc_curves.png` — Per-class ROC with macro average AUC
- `results.json` — Accuracy and AUC metrics

**Arguments**:
| Flag | Default | Description |
|------|---------|-------------|
| `--arch` | `resnet50` | Model architecture |
| `--checkpoint` | `None` | Path to `.pth` checkpoint |
| `--history` | `None` | Path to training history `.npy` file |

### 4. API Server

```bash
# Start FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Endpoints**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | Health check with model status |
| `/predict` | POST | Upload image → classification result |
| `/predict/batch` | POST | Batch upload multiple images |
| `/docs` | GET | Interactive Swagger UI |
| `/redoc` | GET | ReDoc documentation |

**Example API call**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@data/raw/Agriculture/AnnualCrop_1.jpg"
```

**Response**:
```json
{
  "predicted_class": "Agriculture",
  "confidence": 0.9234,
  "all_probabilities": {
    "Agriculture": 0.9234,
    "Water": 0.0312,
    "Urban": 0.0156,
    "Desert": 0.0098,
    "Roads": 0.0123,
    "Trees": 0.0077
  },
  "model_version": "1.0.0",
  "timestamp": "2026-06-22T19:48:47"
}
```

---

## Results

### Custom CNN (Best Model)

| Metric | Value |
|--------|:-----:|
| **Test Accuracy** | **97.23%** |
| **Macro AUC** | **0.9972** |

| Class | Precision | Recall | F1-Score |
|-------|:---------:|:------:|:--------:|
| Agriculture | 0.9567 | 0.9618 | **0.9592** |
| Water | 0.9961 | 0.9358 | **0.9650** |
| Urban | 0.9915 | 0.9927 | **0.9921** |
| Desert | 1.0000 | 1.0000 | **1.0000** |
| Roads | 0.9677 | 0.9600 | **0.9639** |
| Trees | 0.9387 | 0.9867 | **0.9621** |

### ResNet50 (Transfer Learning)

| Metric | Value |
|--------|:-----:|
| **Test Accuracy** | **95.27%** |
| **Macro AUC** | **0.9957** |

### Key Findings
- **Desert** achieves perfect classification (100% F1) — distinct spectral signature
- **Water** has high precision (99.6%) but slightly lower recall (93.6%) — some water misclassified as Urban/Agriculture
- **Roads vs Urban** is the most challenging distinction — they share similar built-up surface materials
- Custom CNN outperformed ResNet50 (97.2% vs 95.3%) because satellite patches (64×64) lose detail when upscaled to 224×224 for ResNet50

---

## Model Architecture

### Custom CNN (1.2M parameters)

```
Input (64×64 RGB)
  └─ Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool(2)
  └─ Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool(2)
  └─ Conv2D(128) → BN → ReLU → MaxPool(2)
  └─ Conv2D(256) → BN → ReLU → GlobalAvgPool
  └─ Dropout(0.4) → Dense(256) → Dropout(0.4) → Dense(128) → Dense(6)
```

### ResNet50 (25.6M params, 2.3M trainable)

```
ResNet50 backbone (ImageNet pretrained, frozen in Phase 1)
  └─ Layer4+ unfrozen in Phase 2 (fine-tuning, LR=1e-5)
  └─ GlobalAvgPool → Dense(512) → BN → Dropout(0.5)
  └─ Dense(256) → Dropout(0.3) → Dense(6)
```

### Training Protocol
- **Optimizer**: AdamW with weight decay (1e-4)
- **Loss**: Cross-entropy with label smoothing (0.1 → 0.05 in Phase 2)
- **Schedule**: ReduceLROnPlateau (factor 0.5, patience 5)
- **Early stopping**: Patience 10 epochs on val_loss
- **Gradient clipping**: Max norm 1.0

---

## Deployment

### Docker

```bash
# Build image
docker build -t sentinel2-classifier .

# Run container
docker run -p 8000:8000 sentinel2-classifier
```

### Cloud Deployment
The FastAPI app can be deployed to:
- **Google Cloud Run** (serverless, auto-scaling)
- **AWS ECS / Fargate**
- **Azure Container Instances**
- Any Docker-compatible hosting platform

---

## Project Roadmap

| Milestone | Status | Deliverables |
|-----------|--------|--------------|
| **M1**: Data Collection & Preprocessing | ✅ Complete | EuroSAT extraction, desert download, data splits, augmentation, EDA |
| **M2**: Advanced Analysis & Model Selection | ✅ Complete | PCA analysis, Custom CNN baseline, ResNet50 transfer learning |
| **M3**: Model Development & Training | ✅ Complete | Two-phase training, evaluation suite, Grad-CAM, model comparison |
| **M4**: MLOps & Deployment | ✅ Complete | FastAPI endpoint, Dockerfile, prediction logging |
| **M5**: Documentation & Presentation | ✅ Complete | README, roadmap doc, project structure |

### Future Work
- Multi-temporal classification (time series of Sentinel-2 images)
- Object detection for individual building/infrastructure mapping
- Hyperspectral data integration (EnMAP)
- Deploy to cloud with automated retraining pipeline
- Class imbalance handling for Roads class (currently undersampled)
