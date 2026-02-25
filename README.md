## Drowsiness Detection System
A real-time driver drowsiness detection system using deep learning and computer vision to prevent road accidents caused by driver fatigue. 
The system monitors facial states (eye closure and yawning) through a webcam and triggers audio alerts when drowsiness is detected.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-3.0-green)

## Table of Contents
- [Features](#features)
- [Demo](#demo)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Architecture (Best Model - Deep CNN)](#architecture-best-model---deep-cnn)
- [Project Structure](#project-structure)

## Features

- **Real-time Detection**: Monitors driver facial states every 1.5 seconds
- **High Accuracy**: Achieves 96.5% accuracy on test data
- **Audio Alerts**: Triggers sound alarm when signs of drowsiness are detected
- **Live Confidence Scores**: Displays prediction probabilities for all classes
- **Lightweight**: Runs efficiently on standard hardware
- **Offline Capable**: Runs fully offline after initial dependency installation

## Demo
- **Alert State**: Detects closed eyes and yawning states
- **Safe State**: Monitors open eyes and normal facial expression
- **Confidence Display**: Shows real-time prediction probabilities

## Machine Learning Pipeline
```
Data Ingestion → Data Transformation → Model Training → Prediction Pipeline → Web App
```

## Tech Stack
- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Flask (Python)
- **ML Framework**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Model**: Convolutional Neural Network (CNN)

## Dataset

- **Source**: [Kaggle Drowsiness Dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)
- **Classes**: 4 (Closed, Open, yawn, no_yawn)
- **Total Images**: ~2,900
- **Distribution**: Balanced across all classes
- **Image Size**: Resized to 128x128 pixels

## Model Performance

- **Test Accuracy**: 96.5% on unseen test data
- **Training Accuracy**: 97.7%
- **Model**: Deep CNN with BatchNormalization

## Architecture (Best Model - Deep CNN)
```
Input (128x128x3)
    ↓
Conv2D(32) → MaxPool
    ↓
Conv2D(64) → MaxPool
    ↓
Conv2D(128) → MaxPool
    ↓
Flatten → Dense(128) → Dropout(0.5)
    ↓
Output (4 classes - Softmax)
```

## Project Structure
```
drowsiness_detection_system/
│
├── app.py                          # Flask application
├── setup.py                        # Package installer
├── requirements.txt                # Dependencies
│
├── src/
│   ├── __init__.py
│   ├── logger.py                   # Logging utility
│   ├── exception.py                # Custom exceptions
│   ├── utils.py                    # Helper functions
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py      # Load and split data
│   │   ├── data_transformation.py # Preprocess images
│   │   └── model_trainer.py       # Train CNN models
│   │
│   └── pipeline/
│       ├── __init__.py
│       └── predict_pipeline.py    # Prediction logic
│
├── templates/
│   └── index.html                 # Web interface
│
├── static/
│   └── uploads/                   # Temporary image storage
│
├── artifact/
│   ├── model.h5                   # Trained model
│   ├── preprocessor.pkl           # Preprocessing config
│   ├── train.npy                  # Training data
│   ├── test.npy                   # Testing data
│   └── *.npy                      # Label files
│
├── notebook/
│   ├── data/                      # Raw dataset
│   ├── EDA_drowsiness.ipynb      # Exploratory analysis
│   └── MODEL_TRAINING.ipynb      # Model experiments
│
└── logs/                          # Application logs
```


