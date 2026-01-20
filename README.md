# Hybrid-CNN-HOG-PCA-SVM-with-QSVM-for-TB-Detection
Hybrid deep learning and handcrafted feature framework for tuberculosis detection from chest X-rays using CNN, HOG, PCA, classical SVM, and exploratory QSVM, with external validation.

# Hybrid CNN–HOG–PCA–SVM Framework for Tuberculosis Detection

This repository presents a **hybrid classical–deep learning framework** for automated tuberculosis (TB) detection from chest X-ray images. The proposed approach integrates **CNN-based deep features**, **handcrafted HOG descriptors**, **PCA-based dimensionality reduction**, and **SVM classification**, with an exploratory **Quantum SVM (QSVM)** analysis.

---

## 🔬 Key Contributions
- Hybrid feature fusion using CNN + HOG
- PCA-based dimensionality reduction (PCA-64 optimal)
- High-performance classical SVM classifier
- Exploratory QSVM evaluation under quantum constraints
- External validation on Shenzhen TB dataset
- Extensive ablation and segmentation analysis

---

## 🧠 Methodology Overview
1. Image preprocessing & lung segmentation
2. CNN feature extraction (EfficientNet backbone)
3. HOG feature extraction
4. Feature fusion + PCA
5. Classification using SVM and QSVM
6. Internal & external validation

---

## 📊 Results (Kaggle Dataset)
| Model | Accuracy |
|------|----------|
| CNN-only | 96.5% |
| HOG-only | 97.4% |
| CNN + HOG | 97.9% |
| **Proposed (CNN+HOG+PCA-64)** | **98.2%** |

- ROC–AUC (Proposed): **0.999**
- External validation (Shenzhen): performance drop observed due to domain shift

---
Due to license restrictions, datasets are not included.

Kaggle TB dataset:
https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset

Shenzhen TB dataset:
https://www.kaggle.com/datasets/nih-chest-xrays/data

