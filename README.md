# Skin Cancer Detection Using CNN

A Deep Learning-based skin lesion classification project that uses Convolutional Neural Networks (CNN) to analyze dermoscopic images and classify skin lesions into different categories.

## Overview

Skin cancer detection at an early stage can help support timely medical evaluation. This project explores the use of Deep Learning and image classification techniques to identify different skin lesion categories from dermoscopic images.

The system uses a CNN model trained on the HAM10000 dataset to learn image features and perform multi-class skin lesion classification.

## Features

- **Image Classification:** Classifies dermoscopic images into multiple skin lesion categories.
- **CNN-Based Model:** Uses Convolutional Neural Networks for automated image classification.
- **Image Preprocessing:** Applies image preprocessing techniques to prepare images for model training.
- **Data Augmentation:** Uses augmentation techniques to improve model generalization.
- **Prediction:** Generates a predicted skin lesion category for an input image.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow & Keras | Deep Learning model development |
| CNN | Image classification |
| HAM10000 | Dermoscopic image dataset |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |

## Dataset

**HAM10000 (Human Against Machine with 10,000 training images)**

- Contains dermoscopic images of different skin lesion categories.
- Used for training and evaluating the classification model.
- Supports multi-class skin lesion classification.

## Project Workflow

1. **Dataset Collection:** Obtain dermoscopic images from the HAM10000 dataset.
2. **Image Preprocessing:** Resize and prepare images for model input.
3. **Data Augmentation:** Apply image transformations to improve training.
4. **Model Development:** Build a CNN-based classification model.
5. **Model Training:** Train the model using the prepared dataset.
6. **Model Evaluation:** Evaluate classification performance using appropriate metrics.
7. **Prediction:** Use the trained model to classify input skin lesion images.


## Disclaimer

This project is developed for educational and research purposes only. It is not intended to replace professional medical diagnosis or clinical evaluation. Always consult a qualified healthcare professional for medical advice.
