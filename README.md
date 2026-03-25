# Credit Card Fraud Detection: Multi-Level Anomaly Detection with ML & DL

**DATA 994 Capstone Project** | Siddartha Bandi | University of North Dakota

---

## Overview

This capstone project presents a comprehensive comparative analysis of classical Machine Learning and Deep Learning models for detecting credit card fraud. The core challenge is extreme class imbalance (~578:1 legitimate-to-fraud ratio), addressed through SMOTE rebalancing and a novel **multi-level risk classification framework** (Low / Medium / High Risk) that goes beyond binary fraud detection.

The project spans three research phases: Exploratory Data Analysis, supervised/unsupervised ML modeling, and deep learning with hyperparameter optimization.

---

## Data Source

| Dataset | Source | Size |
|---------|--------|------|
| [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Kaggle (ULB Machine Learning Group) | ~144 MB, 284,807 transactions |

**Features:** 30 columns — `Time`, `Amount`, 28 anonymized PCA components (`V1`–`V28`), and binary `Class` label (0 = Legitimate, 1 = Fraudulent).

> **Note:** The raw dataset is not included in this repository due to its size. Download it via the Kaggle API (see Setup below).

---

## Methodology

### Phase 1 — Exploratory Data Analysis
- Class distribution analysis (fraud rate: ~0.172%)
- Transaction amount comparison (Mann-Whitney U test)
- Temporal pattern analysis across transaction timestamps
- Correlation heatmap and KDE feature separability plots
- Outlier analysis using IQR method

### Phase 2 — Machine Learning Models
- **SMOTE** (Synthetic Minority Oversampling Technique) to address 578:1 class imbalance
- **Supervised Models:** Random Forest, XGBoost
- **Unsupervised Models:** Isolation Forest
- Evaluation metric: **AUPRC** (Area Under Precision-Recall Curve) — preferred over AUC-ROC for imbalanced datasets

### Phase 3 — Deep Learning Models
- **Autoencoder** — unsupervised anomaly detection via reconstruction error
- **LSTM** — sequential pattern modeling on transaction time-series
- **Hyperparameter tuning** via grid search for all five models
- **Multi-level risk stratification:** each model outputs Low / Medium / High risk scores

---

## Results

### Model Performance (AUPRC)

| Model | Baseline AUPRC | Optimized AUPRC | Improvement |
|-------|---------------|----------------|-------------|
| **XGBoost** | 0.8479 | **0.8687** | +2.45% |
| Random Forest | 0.8643 | 0.8636 | -0.08% |
| LSTM | 0.8021 | 0.8133 | +1.40% |
| Autoencoder | 0.4606 | 0.4606 | 0.00% |
| Isolation Forest | 0.2089 | 0.2693 | +28.90% |

### Multi-Level Risk Distribution (% of Fraud Transactions)

| Model | Low Risk | Medium Risk | High Risk |
|-------|----------|-------------|-----------|
| Random Forest | 2% | 1% | **95%** |
| XGBoost | 1% | 2% | **95%** |
| Isolation Forest | 3% | 1% | **94%** |
| Autoencoder | 2% | 1% | **95%** |
| LSTM | 1% | 3% | **94%** |

### Key Findings
- **XGBoost** achieved the best overall performance after tuning (AUPRC = 0.8687)
- All supervised/deep learning models correctly classified **94–95% of fraud transactions as High Risk**
- SMOTE significantly improved supervised model recall without sacrificing precision
- Isolation Forest showed the largest relative improvement (+28.9%) from hyperparameter tuning

---

## Project Structure

```
Capstone Project/
├── README.md
├── Data/
│   ├── hyperparameter_tuning_results.csv   # Model tuning comparison results
│   └── level2_risk_distribution.csv        # Multi-level risk output per model
├── Python files/
│   ├── DATA994.ipynb                        # Main Jupyter Notebook (Google Colab)
│   ├── data994.py                           # Python script version
│   └── plots/                              # 22 generated visualization plots
│       ├── plot_01_class_distribution.png
│       ├── plot_10_ml_comparison.png
│       ├── plot_14_all_models_comparison.png
│       └── ... (22 total plots)
├── Paper1 - Bandi.pdf                       # Research Paper 1: EDA & Problem Framing
├── Paper2 - Bandi.pdf                       # Research Paper 2: ML Modeling
├── Paper3 - Bandi.pdf                       # Research Paper 3: DL & Final Analysis
└── Proposal - Bandi.pdf                   # Original project proposal
```

---

## Setup & Usage

### Prerequisites
- Python 3.9+
- Google Colab (recommended) or local Jupyter environment
- Kaggle account with API token

### 1. Download the Dataset

```bash
# Install Kaggle API
pip install kaggle

# Place your kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn \
            imbalanced-learn xgboost tensorflow keras
```

### 3. Run the Notebook

Open `Python files/DATA994.ipynb` in Google Colab or Jupyter. Run cells sequentially. All 22 plots will be saved automatically to the `plots/` directory.

---

## Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn (RF, Isolation Forest), XGBoost |
| DL Models | TensorFlow/Keras (Autoencoder, LSTM) |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Environment | Google Colab |

---

## Author

**Siddartha Bandi**
DATA 994 — Capstone Project
University of North Dakota
