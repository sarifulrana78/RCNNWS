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

## 🔬 Dataset & Preprocessing Pipeline

### 1. Indian Facial Emotion Dataset
The dataset comprises facial expressions specifically captured from Indian demographics, addressing common demographic representation biases in computer vision. It contains 6 primary emotion categories:
- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**

### 2. Automated Face Detection & Alignment
To prevent background clutter and irrelevant body features from degrading feature learning:
- **OpenCV DNN Face Detector** (ResNet-10 SSD architecture) is utilized to localize faces with high confidence ($conf > 0.6$).
- Detected face regions are cropped with a calibrated margin and rescaled to uniform $48 \times 48$ grayscale images.
- Uncropped or corrupted images are filtered out automatically.

### 3. Deduplication & Stratified Splitting
- **MD5 Hash Deduplication**: Exact duplicate images across different folders are detected and purged to eliminate train-test leakage.
- **Split Ratio**: The sanitized dataset is split into **70% Training**, **15% Validation**, and **15% Test** partitions using stratified sampling to preserve original class proportions across splits.

### 4. Class Distribution & Offline Augmentation
Facial emotion datasets typically suffer from severe skewness (e.g., Happy/Neutral dominate while Disgust/Fear are underrepresented).

![Class Distribution](class_distribution%20%288%29.png)

To solve this:
- **Offline Data Augmentation**: Applied exclusively to the training set before training begins. Minority classes are augmented using affine transformations (rotations $\pm 15^\circ$, zoom $\pm 15\%$, horizontal shifts $\pm 10\%$) to physically equalize sample counts.
- **Data Integrity**: Augmented samples are written only to the training split, completely safeguarding validation and test splits from leakage.

## 🧠 Custom Residual CNN (Version 3) Architecture

The network is engineered entirely from scratch using the **Keras Functional API** without relying on pretrained backbones or transfer learning weights.

### Key Architectural Pillars:
1. **Dual-Convolution Residual Blocks**:
   - Each residual block employs two consecutive $3 \times 3$ Conv2D layers with Batch Normalization and ReLU activation before residual addition.
   - The first convolution captures localized primitive features (edges/textures), while the second synthesizes compound geometric patterns before spatial downsampling.
2. **Residual Skip Connections**:
   - Identity shortcuts add the block input directly to the post-activation output: $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$.
   - For channel transitions (32 -> 64 -> 128 -> 256), a $1 \times 1$ convolution projection maintains dimensional alignment.
   - Prevents vanishing gradients, allowing smooth gradient backpropagation across deep layers.
3. **SpatialDropout2D**:
   - Drops entire $2D$ feature maps rather than isolated pixels, forcing channels to learn mutually independent representations.
4. **Global Average Pooling (GAP)**:
   - Replaces traditional heavy Flatten layers ($3 \times 3 \times 256 = 2304$ units), dramatically reducing parameter count by over 800,000 parameters and mitigating dense layer overfitting.
5. **Dense Classification Head**:
   - 256-unit fully connected layer regularized with He Normal kernel initialization, $L_2$ weight decay ($1.5 \times 10^{-4}$), Batch Normalization, and Dropout ($0.40$).
   - Softmax output layer providing probability distributions over the 6 emotion classes.

## ⚙️ Training Strategy & Anti-Overfitting Safeguards

### 1. Multi-Tier Regularization
- **Online Augmentation**: Lightweight online perturbations (rotation $8^\circ$, zoom $8\%$, shift $6\%$, horizontal flip) ensure the network never encounters identical images across consecutive epochs.
- **Capped Class Weights**: A dynamic balanced class weight schedule is computed as a safety net:
  $$\text{weight}_c = \min\left(\frac{N}{K \cdot N_c}, \, 5.0 \times \min(\mathbf{w})\right)$$
  Capping avoids excessive penalty spikes that destabilize gradient steps.

### 2. Hyperparameter Configuration
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Input Shape** | $(48, 48, 1)$ | Standardized grayscale facial input |
| **Batch Size** | 32 | Balanced gradient stochasticity |
| **Initial Learning Rate** | $5 \times 10^{-4}$ | Adam optimizer convergence stability |
| **Epochs** | 100 | Sufficient depth for residual block convergence |
| **Spatial Dropout** | 0.20 | Channel-wise feature map regularization |
| **Dense Dropout** | 0.40 | Dense head regularization |
| **$L_2$ Weight Decay** | $1.5 \times 10^{-4}$ | Weight penalty across all convolutional layers |

### 3. Training Callbacks
- **ModelCheckpoint**: Automatically saves `emotion_model_best.keras` based on highest validation accuracy.
- **ReduceLROnPlateau**: Decays learning rate by factor $0.5$ when validation loss stagnates for 5 epochs (floor: $10^{-6}$).
- **EarlyStopping**: Halts training if validation loss fails to improve for 15 consecutive epochs.
- **CSVLogger**: Streams epoch-by-epoch loss and accuracy metrics to disk for auditing.

## 📈 Training History & Convergence Dynamics

The model was trained over 100 epochs with dynamic learning rate scheduling and early stopping safeguards.

![Training History](training_history%20%2811%29.png)

### Key Observations:
- **Rapid Initial Convergence**: The residual skip connections enabled rapid learning within the first 15 epochs, crossing 60% validation accuracy early.
- **Generalization Gap**: The difference between final training accuracy and validation accuracy remained healthy (under 7%), demonstrating that the dual-stage regularization (offline + online augmentation + SpatialDropout) prevented catastrophic overfitting.
- **Stable Loss Trajectory**: Validation loss plateaued smoothly without diverging.

## 💾 Model Serialization & Weights

The fully trained model is saved in modern native Keras format (`.keras`):
- **Artifact**: `emotion_model (4).keras` (15.6 MB)
- **Framework**: Keras 3 / TensorFlow 2.x
- **Format**: Self-contained archive including full architecture graph, compiled optimizer state, and learned weights.

### Loading the Model in Python:
```python
from keras.models import load_model

model = load_model("emotion_model (4).keras")
model.summary()
```

## 📊 Quantitative Evaluation on Held-Out Test Set

The model was evaluated on an independent, held-out test set containing **5,948 unseen images**.

### Confusion Matrix
![Confusion Matrix](confusion_matrix%20%288%29.png)

### Classification Report:
```text
 Final Test Accuracy: 70.93%

              precision    recall  f1-score   support

       Angry       0.72      0.67      0.69      1123
     Disgust       0.45      0.37      0.41       159
        Fear       0.59      0.52      0.55       229
       Happy       0.79      0.73      0.76      1608
     Neutral       0.71      0.79      0.75      2095
         Sad       0.60      0.64      0.62       734

    accuracy                           0.71      5948
   macro avg       0.64      0.62      0.63      5948
weighted avg       0.71      0.71      0.71      5948
```

### Insights:
- **Happy** achieved the highest precision ($79\%$) and F1-score ($0.76$), driven by distinct facial landmarks (smiling mouth curves).
- **Neutral** demonstrated high recall ($79\%$) and robust precision ($71\%$), forming a strong baseline.
- **Angry** and **Sad** achieved balanced precision/recall around $60\% - 72\%$.
- **Disgust** and **Fear** presented the greatest challenge due to limited original natural samples, but offline augmentation successfully elevated performance beyond standard baseline FER networks.

## 🔮 Real-Time Inference & Smoke Testing

### Quick Inference Snippet
To run facial emotion prediction on an arbitrary image:

```python
import cv2
import numpy as np
from keras.models import load_model

# 1. Load trained model
model = load_model("emotion_model (4).keras")
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']

# 2. Preprocess input image
def predict_emotion(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (48, 48))
    img_norm = img_resized.astype("float32") / 255.0
    img_tensor = np.expand_dims(img_norm, axis=(0, -1)) # Shape: (1, 48, 48, 1)

    # 3. Predict probabilities
    preds = model.predict(img_tensor)[0]
    predicted_idx = np.argmax(preds)
    return class_labels[predicted_idx], preds[predicted_idx]

# Example execution
# emotion, confidence = predict_emotion("sample_face.jpg")
# print(f"Detected: {emotion} ({confidence*100:.1f}%)")
```

### Smoke Test Verification
A smoke test verifies end-to-end operational viability without requiring the full dataset:
1. Feeds synthetic random tensors $(1, 48, 48, 1)$ through the compiled model.
2. Asserts output shape matches $(1, 6)$ and softmax probabilities sum to $1.0 \pm 10^{-5}$.
3. Confirms GPU acceleration and model inference latency ($< 15\text{ ms}$ per frame).

---

## 🏆 Capstone Project Summary

| Metric | Result |
| :--- | :--- |
| **Model Type** | Custom Deep Residual CNN (Built from scratch) |
| **Total Test Samples** | 5,948 images |
| **Overall Test Accuracy** | **70.93%** |
| **Macro F1-Score** | 0.63 |
| **Weighted F1-Score** | 0.71 |
| **Demographic Focus** | Indian Facial Emotion Dataset |
| **Key Novelty** | Offline Augmentation Balancing + Dual-Conv Skip Residuals |

---

## 👨‍💻 Author & Acknowledgements

Developed by **Sariful Islam** ([@sarifulrana78](https://github.com/sarifulrana78)) as part of **Capstone C**.
