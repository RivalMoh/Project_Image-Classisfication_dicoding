# 🐱🐶 Cat & Dog Image Classification Project

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.0-orange.svg)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Keras](https://img.shields.io/badge/Keras-3.4.1-red.svg)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A comprehensive deep learning project for binary image classification to distinguish between cats and dogs using Convolutional Neural Networks (CNN)**

## 📋 Table of Contents
- [🎯 Project Overview](#-project-overview)
- [✨ Features](#-features)
- [🏗️ Model Architecture](#️-model-architecture)
- [📊 Dataset](#-dataset)
- [🚀 Getting Started](#-getting-started)
- [💻 Usage](#-usage)
- [📈 Model Performance](#-model-performance)
- [🎯 Model Deployment](#-model-deployment)
- [📁 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [👨‍💻 Author](#-author)

## 🎯 Project Overview

This project implements a **Convolutional Neural Network (CNN)** for binary image classification to distinguish between cats and dogs. The model achieves high accuracy through advanced techniques including:

- **Data Augmentation** for improved generalization
- **Regularization techniques** to prevent overfitting
- **Multiple model export formats** for deployment flexibility
- **Comprehensive data preprocessing** pipeline

### 🔬 Technical Highlights
- Built with **TensorFlow/Keras**
- Implements **L2 regularization** and **Dropout**
- Uses **Batch Normalization** for training stability
- **Learning Rate Scheduling** with warmup
- **Early Stopping** and **ReduceLROnPlateau** callbacks
- Exports to **SavedModel**, **TensorFlow.js**, and **TFLite** formats

## ✨ Features

### 🔄 Data Augmentation Pipeline
- **Rotation** (clockwise & anticlockwise)
- **Brightness adjustment**
- **Gaussian blur**
- **Image shearing**
- **Vertical flipping**
- **Warp shifting**

### 🧠 Advanced Model Architecture
- **4 Convolutional layers** with increasing filters (16→32→64→128)
- **MaxPooling** for dimensionality reduction
- **Batch Normalization** for training stability
- **Dropout layers** for regularization
- **Dense layers** with L2 regularization

### 📱 Multi-Format Model Export
- **TensorFlow SavedModel** - For production deployment
- **TensorFlow.js** - For web applications
- **TensorFlow Lite** - For mobile and edge devices

## 🏗️ Model Architecture

```
Input Shape: (150, 150, 3)
├── Conv2D(16) + MaxPool2D + BatchNorm + Dropout(0.25)
├── Conv2D(32) + MaxPool2D + BatchNorm + Dropout(0.25)
├── Conv2D(64) + MaxPool2D + BatchNorm + Dropout(0.25)
├── Conv2D(128) + MaxPool2D + BatchNorm + Dropout(0.25)
├── Flatten
├── Dense(512) + L2 Regularization
└── Dense(1, sigmoid) - Binary Classification
```

## 📊 Dataset

**Source**: [Kaggle - Cat Dog Images for Classification](https://www.kaggle.com/datasets/ashfakyeafi/cat-dog-images-for-classification)

### Dataset Characteristics:
- **Binary Classification**: Cats vs Dogs
- **Image Format**: Various resolutions (normalized to 150x150)
- **Split Ratio**: 80% Training, 20% Testing
- **Validation Split**: 20% of training data
- **Data Augmentation**: Applied to balance dataset

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd 3_Image_Classification_(dog&cat)
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download dataset**
```bash
# Download from Kaggle
wget https://www.kaggle.com/datasets/ashfakyeafi/cat-dog-images-for-classification

# Or use the notebook extraction
# The dataset will be automatically extracted in the notebook
```

### Dataset Setup

**Option 1: Manual Download**
1. Download `cat_dog.zip` from the Kaggle link above
2. Place it in the project root directory
3. Run the notebook - extraction is automated

**Option 2: Programmatic Extraction**
```python
import shutil
shutil.unpack_archive('cat_dog.zip', 'cat_dog')
```

## 💻 Usage

### Training the Model

1. **Open the Jupyter Notebook**
```bash
jupyter notebook Project.ipynb
```

2. **Run all cells sequentially**
   - Data extraction and preprocessing
   - Data augmentation
   - Model training (30 epochs)
   - Model evaluation
   - Model export

### Key Training Parameters
- **Optimizer**: Adam with gradient clipping (clipvalue=1.5)
- **Learning Rate**: 0.001 with ReduceLROnPlateau
- **Batch Size**: 128
- **Epochs**: 30 (with early stopping)
- **Loss Function**: Binary Crossentropy

## 📈 Model Performance

The model demonstrates excellent performance with:
- **Training Accuracy**: 95%+
- **Validation Accuracy**: 90%+
- **Robust generalization** through regularization
- **Stable training** with batch normalization

### Performance Visualizations
The notebook generates comprehensive plots showing:
- Training vs Validation Accuracy
- Training vs Validation Loss
- Learning curves over epochs

## 🎯 Model Deployment

### Available Model Formats

#### 1. **TensorFlow SavedModel** 📁
```
saved_model_new/my_model/
├── saved_model.pb
├── variables/
└── assets/
```
- **Use case**: Production servers, TensorFlow Serving
- **Loading**: `tf.keras.models.load_model('saved_model_new/my_model')`

#### 2. **TensorFlow.js** 🌐
```
tfjs_model/
├── model.json
└── group1-shard*.bin
```
- **Use case**: Web applications, browser-based inference
- **Loading**: `await tf.loadLayersModel('tfjs_model/model.json')`

#### 3. **TensorFlow Lite** 📱
```
tflite/
└── model.tflite
```
- **Use case**: Mobile apps, IoT devices, edge computing
- **Size**: Optimized for mobile deployment

### Deployment Examples

**Web Application (TensorFlow.js)**
```javascript
const model = await tf.loadLayersModel('tfjs_model/model.json');
const prediction = model.predict(imageData);
```

**Mobile App (TensorFlow Lite)**
```python
interpreter = tf.lite.Interpreter(model_path='tflite/model.tflite')
interpreter.allocate_tensors()
```

## 📁 Project Structure

```
3_Image_Classification_(dog&cat)/
├── 📓 Project.ipynb           # Main notebook with complete pipeline
├── 📄 README.md              # Project documentation
├── 📋 requirements.txt       # Python dependencies
├── 📁 saved_model_new/       # TensorFlow SavedModel format
│   └── my_model/
├── 📁 tfjs_model/            # TensorFlow.js format
│   ├── model.json
│   └── *.bin files
├── 📁 tflite/                # TensorFlow Lite format
│   └── model.tflite
├── 📁 cat_dog/               # Extracted dataset (created during runtime)
└── 📁 dataset/               # Processed dataset (train/test split)
    ├── train/
    │   ├── cat/
    │   └── dog/
    └── test/
        ├── cat/
        └── dog/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👨‍💻 Author

**Rival Moh. Wahyudi**
- 📧 Email: vallwhy@students.unnes.ac.id
- 🎓 Institution: Universitas Negeri Semarang (UNNES)
- 🏆 Dicoding ID: Rival Moh. Wahyudi

---

### 🙏 Acknowledgments

- [Dicoding](https://dicoding.com) for the learning platform
- [Kaggle](https://kaggle.com) for the dataset
- TensorFlow/Keras community for excellent documentation

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
⭐ **Star this repository if you found it helpful!**
