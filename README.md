# Deepfake Image Detection using EfficientNet-B4/B5

This project presents a deep learning-based solution for detecting deepfake images using the EfficientNet-B0 CNN model. The model is trained and fine-tuned on a large dataset comprising over 200,000 images from FaceForensics++, DeepFake Detection Challenge (DFD), and NVIDIA datasets. It achieves high accuracy and AUC scores and is deployed via a user-friendly web application.

## 🔍 Project Overview

Deepfakes are AI-generated images or videos that can convincingly mimic real people, posing serious threats to digital trust and security. This project aims to build a reliable system to detect such manipulated images using advanced deep learning techniques.

## 🚀 Features

- EfficientNet-B0 CNN architecture
- Three-phase training strategy
- 97%+ validation accuracy
- 98%+ AUC score
- Confusion matrix and ROC curve evaluation
- Web application for real-time image prediction
- Trained on Kaggle with GPU support

## 🧠 Model Architecture

- **Base Model:** EfficientNet-B0 (pre-trained on ImageNet)
- **Input Size:** 224x224 RGB images
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Evaluation Metrics:** Accuracy, AUC, Confusion Matrix

## 🧑‍🏫 Training Strategy

1. **Phase 1:** Train top classification layers
2. **Phase 2:** Fine-tune middle layers
3. **Phase 3:** Train full model end-to-end

## 📊 Results

- **Validation Accuracy:** 97%+
- **AUC Score:** 98%+
- **Confusion Matrix:** High TP and TN rates
- **ROC Curve:** Strong separability between real and fake images

## 🌐 Web Application

A lightweight web app built using Flask or Streamlit allows users to upload an image and receive real-time predictions on whether it is real or deepfake.

## ⚙️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/deepfake-detector.git
   cd deepfake-detector
   ```
