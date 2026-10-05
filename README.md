# RCNNWS

# Indian Facial Emotion Recognition - Custom Residual CNN (v3)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org)
[![Keras 3](https://img.shields.io/badge/Keras-v3-red.svg)](https://keras.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Deep Learning system engineered for robust **Facial Emotion Recognition (FER)** on the Indian Facial Emotion dataset. The system features a custom dual-convolution **Residual CNN (v3)** designed from scratch, paired with an offline augmentation and dynamic class weighting pipeline to eliminate class imbalance and prevent overfitting.

---

## 📌 Project Overview

Facial Emotion Recognition (FER) in demographic-specific settings (such as Indian facial datasets) frequently suffers from severe class imbalance, varying lighting, and head orientations. Standard off-the-shelf sequential CNN architectures often plateau around 60–63% accuracy due to gradient dissipation and feature degradation.

**RCNNWS** implements a modern **Custom Residual CNN** with skip connections, Spatial Dropout, Batch Normalization, and Global Average Pooling (GAP) that achieves:
- **70.93% Test Accuracy** across 6 facial emotion categories: `Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, and `Sad`.
- **Generalization Gap under 7%** through dual-layer regularization (Offline Balancing + Online Data Augmentation + Dropout + L2 Weight Decay).
- Real-time face detection and emotion classification inference interface.

---

## 🗂️ Repository Architecture

```text
RCNNWS/
├── src/                        # Modular source code
│   ├── config.py               # Hyperparameters, constants & paths
│   ├── data_preprocessing.py   # Face cropping, deduplication & offline balancing
│   ├── model.py                # Custom Residual CNN architecture definition
│   ├── train.py                # Training pipeline with class-weighted loss
│   ├── evaluate.py             # Evaluation routines & metric visualizations
│   └── predict.py              # Single-image / webcam inference engine
├── models/                     # Trained weights & serialized Keras model
├── results/                    # Confusion matrix, history curves & metrics
├── notebooks/                  # Interactive experimentation & Colab workflow
├── templates/                  # Interactive web demo UI
├── app.py                      # Flask web application & REST API
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment definition
└── README.md                   # Project documentation
```

---

## 🚀 Quick Setup

```bash
# Clone the repository
git clone https://github.com/sarifulrana78/RCNNWS.git
cd RCNNWS

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
