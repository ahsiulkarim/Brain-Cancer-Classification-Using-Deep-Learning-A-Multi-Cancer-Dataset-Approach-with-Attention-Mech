# 🧠 Brain Cancer Classification Using Deep Learning

> A multi-model deep learning framework for automated brain tumor classification from MRI images, featuring attention mechanisms, 5-fold cross-validation, and statistical model comparison.

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Statistical Analysis](#statistical-analysis)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Team Members](#team-members)

---

## 📖 About the Project

Brain cancer is among the most devastating neurological disorders, with primary brain tumors accounting for approximately **308,102 new cases globally per year** (WHO). Early and accurate classification is critical for treatment planning and patient outcomes.

This project builds a reproducible, end-to-end deep learning pipeline for **4-class MRI-based brain tumor classification** — Glioma, Meningioma, Pituitary, and No Tumor. The best-performing model, a ResNet-50 backbone augmented with a **Convolutional Block Attention Module (CBAM)**, achieves **97.8% test accuracy**, outperforming all baselines.

The system serves three primary goals:
- **Clinical Relevance** — Automated MRI classification to reduce radiologist workload in resource-limited settings.
- **Interpretability** — Attention maps that highlight diagnostically meaningful tumor regions, critical for clinical trust.
- **Reproducibility** — A fully modular, commented codebase enabling replication and further research.

---

## ✨ Features

- 🔬 4-class brain tumor classification: Glioma, Meningioma, Pituitary, No Tumor
- 🏗️ Multiple model architectures compared under consistent training conditions
- 🎯 CBAM Attention Mechanism for improved accuracy and interpretability
- 🔁 5-Fold Stratified Cross-Validation for robust evaluation
- 📊 Per-class metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- 📈 Statistical significance testing via Paired T-Test across models
- 🖼️ Confusion matrix heatmaps and per-class ROC curve visualizations
- 💾 Best model checkpoints saved per fold (`best_custom_cnn_fold{N}.pth`)
- 🌱 Global seed (42) across Python, NumPy, PyTorch CPU & CUDA for full reproducibility

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch, TorchVision |
| Pretrained Models | ResNet-50, VGG-16, EfficientNet-B0 (ImageNet weights) |
| Attention Module | CBAM (Channel + Spatial Attention) |
| Validation Strategy | Scikit-learn StratifiedKFold |
| Statistics | SciPy (paired t-test) |
| Visualization | Matplotlib, Seaborn |
| Platform | Kaggle (GPU: CUDA) |
| Language | Python 3 |

---

## 📁 Project Structure

```
brain-cancer-classification/
│
├── task-01-eda.ipynb                        # Task 1 — Exploratory Data Analysis
├── resnet50-pretrained-model.ipynb          # Task 2a — ResNet-50 Transfer Learning
├── efficientnet-b0-pretrained-model.ipynb   # Task 2b — EfficientNet-B0 Transfer Learning
├── vgg16.ipynb                              # Task 2c — VGG-16 Transfer Learning
├── task-3-customcnn-with-k-fold.ipynb       # Task 3 — Custom CNN (No Attention) + 5-Fold CV
├── task-04.ipynb                            # Task 4 — Custom CNN + CBAM Attention + 5-Fold CV
├── task-5.ipynb                             # Task 5 — Final Model Evaluation
├── paired-t-test-brain-cancer.ipynb         # Statistical Comparison (Paired T-Test)
│
└── README.md
```

---

## 📂 Dataset

**Training Dataset:** Multi-Cancer Dataset, publicly available on [Kaggle](https://www.kaggle.com/datasets/obulisainaren/multi-cancer).

**Validation Dataset:** Brain Tumor MRI Dataset — compiled from Figshare, SARTAJ, and Br35H.

| Class | Description | Train | Test |
|---|---|---|---|
| Glioma | Malignant brain tumor | 1,321 | 300 |
| Meningioma | Benign/malignant tumor | 1,339 | 306 |
| Pituitary | Pituitary adenoma | 1,457 | 300 |
| No Tumor | Healthy brain scan | 1,595 | 405 |
| **Total** | | **5,712** | **1,311** |

**Kaggle Dataset Path:**
```
/kaggle/input/datasets/obulisainaren/multi-cancer/Multi Cancer/Multi Cancer/Brain Cancer
```

**Preprocessing applied:**
- Resize all images to 224×224 pixels
- Normalize using ImageNet mean/std: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- Training augmentation: random horizontal flip, rotation (±15°), color jitter, random affine translation

---

## ⚙️ Methodology

### Training Pipeline

```
Raw MRI Images (224×224)
        ↓
Augmentation (training split only)
        ↓
80% Train / 10% Val / 10% Test split
        ↓
5-Fold Stratified Cross-Validation (Custom CNNs)
        ↓
Best checkpoint saved per fold → Final test evaluation
```

### Shared Hyperparameters

| Parameter | Value |
|---|---|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 1e-3 (Custom CNNs) / 1e-4 (ResNet-50 + CBAM) |
| Weight Decay | 1e-4 |
| Max Epochs | 50 |
| Early Stopping Patience | 5 epochs |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Loss Function | CrossEntropyLoss |
| Random Seed | 42 |

---

## 🤖 Models

### Task 1 — Exploratory Data Analysis (`task-01-eda.ipynb`)
Class distribution bar charts, per-class sample visualization, pixel statistics, and dataset balance assessment.

---

### Task 2 — Transfer Learning Baselines

Three ImageNet-pretrained models fine-tuned on the brain tumor dataset with a custom 4-class classification head.

| Notebook | Model | Split |
|---|---|---|
| `resnet50-pretrained-model.ipynb` | ResNet-50 | 80 / 20 |
| `efficientnet-b0-pretrained-model.ipynb` | EfficientNet-B0 | 80 / 20 |
| `vgg16.ipynb` | VGG-16 | 70 / 15 / 15 |

---

### Task 3 — Custom CNN with 5-Fold CV (`task-3-customcnn-with-k-fold.ipynb`)

A 5-block convolutional network built from scratch in PyTorch (no pretrained weights):

```
Input (3, 224, 224)
    → ConvBlock(3→32) → ConvBlock(32→64) → ConvBlock(64→128)
    → ConvBlock(128→256) → ConvBlock(256→512)
    → AdaptiveAvgPool2d(1)
    → Linear(512, 256) → ReLU → Dropout(0.5) → Linear(256, 4)
```

Each `ConvBlock`: `Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → MaxPool2d`

---

### Task 4 — Custom CNN + CBAM Attention (`task-04.ipynb`)

Extends Task 3 with a **CBAM** module inserted between the convolutional blocks and classifier head.

**Channel Attention:**
```
Mc(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) )
```

**Spatial Attention:**
```
Ms(F) = σ( f^{7×7}( [AvgPool(F); MaxPool(F)] ) )
```

**Full Architecture Flow:**
```
(B, 3, 224, 224) → 5 ConvBlocks → (B, 512, 7, 7)
    → Channel Attention → Spatial Attention
    → AdaptiveAvgPool → FC Head → (B, 4)
```

| Property | Baseline CNN | CNN + CBAM |
|---|---|---|
| Pre-trained | No | No |
| Conv Blocks | 5 (3 → 512) | 5 (3 → 512) |
| Attention | None | CBAM (ratio=16) |
| Training | 5-Fold K-Fold CV | 5-Fold K-Fold CV |
| Optimizer | Adam (lr=1e-3) | Adam (lr=1e-3) |

---

### Task 5 — Final Model Evaluation (`task-5.ipynb`)
Consolidated evaluation of the best-performing model with confusion matrix, per-class ROC curves, and full JSON results export for inter-model comparison.

---

### Best Model — ResNet-50 + CBAM (Research Paper)

ResNet-50 backbone with CBAM inserted after the final ResNet block. Top 30 layers unfrozen for fine-tuning.

```
ResNet-50 (ImageNet pretrained, top 30 layers unfrozen)
    → CBAM on 7×7×2048 feature map
    → Global Average Pooling
    → Dense(512, ReLU) → Dropout(0.5) → Dense(4, Softmax)
```

---

## 📊 Results

### Per-Class Performance — ResNet-50 + CBAM (Best Model)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Glioma | 0.982 | 0.973 | 0.977 |
| Meningioma | 0.964 | 0.961 | 0.962 |
| Pituitary | 0.991 | 0.990 | 0.990 |
| No Tumor | 0.985 | 0.992 | 0.988 |
| **Weighted Avg** | **0.980** | **0.979** | **0.979** |

### Model Comparison

| Model | Test Accuracy | Parameters |
|---|---|---|
| SVM + Handcrafted Features | 78.4% | — |
| VGG-16 (Transfer Learning) | 91.3% | 138M |
| InceptionV3 (Transfer Learning) | 94.7% | 23.9M |
| ResNet-50 (No Attention) | 96.1% | 25.6M |
| **ResNet-50 + CBAM (Ours)** | **97.8%** | **26.1M** |

The CBAM attention module delivers a consistent **+1.7 percentage point** improvement over the non-attention ResNet-50 baseline while adding only 0.5M parameters.

---

## 📈 Statistical Analysis

**Notebook:** `paired-t-test-brain-cancer.ipynb`

A **paired t-test** (`scipy.stats.ttest_rel`) is conducted across all three pretrained transfer learning models to determine whether accuracy differences are statistically significant rather than due to random chance.

Pairwise comparisons performed on identical held-out test samples:
- ResNet-50 vs. VGG-16
- ResNet-50 vs. EfficientNet-B0
- VGG-16 vs. EfficientNet-B0

A p-value < 0.05 is considered statistically significant.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python >= 3.8
- CUDA-enabled GPU (recommended) or CPU
- Kaggle account (for dataset access)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/brain-cancer-classification.git
   cd brain-cancer-classification
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision scikit-learn matplotlib seaborn pandas numpy scipy
   ```

3. **Download the dataset**
   - Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/obulisainaren/multi-cancer)
   - Download and place under:
     ```
     data/Brain Cancer/glioma/
     data/Brain Cancer/meningioma/
     data/Brain Cancer/notumor/
     data/Brain Cancer/pituitary/
     ```

4. **Update `DATA_DIR`** in each notebook to point to your local dataset path.

---

## 🚀 Usage

Run notebooks in this recommended order:

```
1. task-01-eda.ipynb                         ← Understand the dataset first
2. resnet50-pretrained-model.ipynb           ← Pretrained baselines
   efficientnet-b0-pretrained-model.ipynb
   vgg16.ipynb
3. task-3-customcnn-with-k-fold.ipynb        ← Custom CNN (no attention)
4. task-04.ipynb                             ← Custom CNN + CBAM Attention
5. task-5.ipynb                              ← Final evaluation
6. paired-t-test-brain-cancer.ipynb          ← Statistical comparison
```

All notebooks use a **global random seed of 42** for full reproducibility.

---

## ⚠️ Limitations

1. **Dataset Scope** — Trained and evaluated on a single publicly available dataset; performance on clinical MRI scans from different scanners may vary.
2. **No 3D Volumetric Analysis** — The model processes 2D MRI slices; 3D spatial context across slices is not utilized.
3. **Manual Annotation Dependency** — Ground truth labels rely on expert radiologist annotations which may contain inter-observer variability.
4. **Compute Requirements** — Full 5-Fold training over 50 epochs requires GPU acceleration; not practical on CPU-only machines.
5. **No Real-Time Inference** — The current pipeline is not optimized for edge deployment or real-time clinical use.

---

## 🔮 Future Work

- 🧊 Extension to 3D volumetric MRI classification using 3D CNNs and transformer-based architectures
- 🔗 Multi-modal framework integrating radiomics features with deep features
- 📱 Lightweight clinical deployment via model quantization and edge inference
- 🏥 Federated learning to train across multiple hospital datasets without compromising patient privacy
- 📝 Integration with DICOM standards for real-world radiology workflow compatibility

---

## 👨‍💻 Team Members

| Name | Student ID | Role |
|---|---|---|
| Marsia Akter Nafisha | 2022-2-60-148 | Team Member |
| Monisha Akter Sumaiya | 2023-3-60-369 | Team Member |
| MD. Ahsiul Karim | 2022-3-60-074 | Team Member |
| Md. Irfan Sadik (Saad) | 2023-1-60-095 | Team Member |
| Azizullah | 2022-3-60-065 | Team Member |
