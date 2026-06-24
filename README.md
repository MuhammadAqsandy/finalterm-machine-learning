# 🤖 Final Term — Machine Learning Individual Task

**"Hands-On End-to-End Models: Machine Learning & Deep Learning"**

---

## 📌 Repository Info

| Field | Detail |
|-------|--------|
| **Name** | Muhammad Aqsandy |
| **Class** | TK-46-02 |
| **NIM** | 1103220214 |
| **Course** | Machine Learning — Universitas Telkom |
| **Task** | UAS Individual Task |

---

## 🎯 Project Overview

This repository contains two end-to-end deep learning pipelines built with **PyTorch**, tuned with **Optuna**, and tracked with **MLflow**.

### Problem 1: Fraud Detection (Binary Classification)
Predict whether an online transaction is fraudulent using the IEEE-CIS Fraud Detection dataset (`train_transaction.csv`).

### Problem 2: Song Release Year Prediction (Regression)
Predict the release year of a song from its audio features using the MSD-style regression dataset (`midterm-regresi-dataset.csv`).

---

## 📂 Repository Structure

```
finalterm-machine-learning/
├── notebook1_fraud_detection.ipynb   # Problem 1: Fraud Detection
├── notebook2_regression.ipynb        # Problem 2: Year Regression
└── README.md                         # This file
```

---

## 🔬 Models & Methods

### Notebook 1 — Fraud Detection
| Component | Details |
|-----------|---------|
| **Model** | MLP (Multi-Layer Perceptron) — PyTorch |
| **Imbalance Handling** | SMOTE oversampling |
| **Tuning** | Optuna TPE Sampler (20 trials) |
| **Optimization** | Adam + ReduceLROnPlateau |
| **Evaluation** | ROC-AUC, F1-Score, Precision, Recall, Confusion Matrix |
| **Tracking** | MLflow (`fraud_detection_mlp` experiment) |

### Notebook 2 — Regression
| Component | Details |
|-----------|---------|
| **Model** | MLP Regression — PyTorch |
| **Outlier Handling** | IQR 1%–99% clipping |
| **Tuning** | Optuna TPE Sampler (20 trials, max R²) |
| **Optimization** | Adam + CosineAnnealingLR |
| **Evaluation** | MSE, RMSE, MAE, R² |
| **Interpretation** | LIME (local feature importance) |
| **Tracking** | MLflow (`song_year_regression_mlp` experiment) |

---

## 🚀 How to Run (Google Colab)

1. Open notebook in Google Colab (Runtime → Change runtime type → **GPU**)
2. Upload your dataset file or mount Google Drive
3. Update `DATA_PATH` variable in Cell 3
4. Run all cells sequentially (**Runtime → Run all**)

### Required Libraries (auto-installed in Cell 1)
```
optuna, mlflow, torch, imbalanced-learn, lime
```

---

## 📊 Results Summary

### Fraud Detection
| Metric | Value |
|--------|-------|
| ROC-AUC | *(run to see)* |
| F1-Score | *(run to see)* |

### Regression
| Metric | Value |
|--------|-------|
| RMSE | *(run to see, in years)* |
| MAE | *(run to see, in years)* |
| R² | *(run to see)* |

---

## 🔗 Datasets
- **Fraud Detection**: IEEE-CIS Fraud Detection — `train_transaction.csv`
- **Regression**: YearPredictionMSD-style — `midterm-regresi-dataset.csv`

> Datasets are **not** included in this repository. Download from the course-provided links and place them in the same directory as the notebooks.

---

*Universitas Telkom — Fakultas Teknik Elektro — Teknik Komputer*
