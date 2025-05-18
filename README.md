# Deep-Learning-Group-Project

## Students: Jonah Levine, Ethan Sandoval, Kevin Chen

## Group Name: Medical AI Sorcerers

# Project Title: Classifying Skin Cancer Images

## Abstract
Skin cancer is one of the most common and potentially deadly forms of cancer, making early and accurate diagnosis crucial. This deep learning-based image classification project makes use of medical imaging and metadata to predict whether a given skin lesion is benign or malignant. Using a dataset from the International Skin Imaging Collaboration (ISIC), which includes over 500,000 entries, we employ convolutional neural networks to analyze skin images. Metadata features such as patient age, anatomical site of the picture, lesion size, and image type are incorporated to enhance model performance. The goal is to develop a robust AI model that assists medical professionals in providing reliable skin cancer classification, improving early detection and guiding clinical decisions.

# Skin Cancer Detection Using Deep Learning

This project builds and evaluates multiple deep learning models for classifying skin lesions as **benign** or **malignant** using both image data and associated metadata. The models are trained and tested using various configurations including data augmentation, ResNet-based transfer learning, and metadata fusion.

## Project Structure

- `PreProcessing.ipynb` – Prepares the data, including image resizing and metadata extraction.
- `Basic - Metadata MLP Baseline.ipynb` – Simple MLP model using metadata only.
- `Basic - Metadata Linear Regression Baseline.ipynb` – Linear model using metadata only.
- `Basic - Image CNN Baseline.ipynb` – Simple CNN for image-based classification.
- `multi_model.ipynb` – Custom model with no pretrained base model
- `subset5000.ipynb` – Model with MobileNetV2 as base model and balanced train and val data (5000 from each class)
- `subset10000.ipynb` – Model with 10000 of each class instead of 5000
- `subset5000-unbalancedval.ipynb` – Trained on balanced data, validated on unbalanced data.
- `subset5000-aug.ipynb` – Includes image data augmentation.
- `subset5000-aug-no-unfreeze.ipynb` – Includes image data augmentation without unfreezing base model
- `subset5000-RESN50.ipynb` – Using RESN50 as base model instead of MobileNetV2
- `subset5000-unbalanced.ipynb` – Trained and validated on unbalanced data.
- `subset5000-metadata.ipynb` – Implemented metadata in training
- `multiclass-froze.ipynb` – Split malignant diagnoses into multiple categories for multiclass classification (no unfreezing base model)
- `multiclass-withunfreeze.ipynb` – Multiclass training with unfreezing base model
- `model_5000_aug.keras` – Final model with the best results (augmentation with no unfreezing)
- `requirements.txt` – List of required Python packages.

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/skin-cancer-detection.git
   cd skin-cancer-detection

2. **Create a virtual environment (optional)**

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies**

pip install -r requirements.txt

4. **Run the notebooks**
Launch Jupyter:

jupyter notebook

Open and run notebooks in the following recommended order:

PreProcessing.ipynb

metadata_splits (2).ipynb

Choose baseline models from the "Basic - ..." notebooks

Proceed to subset5000... variants and multi_model.ipynb

Note: Some code cells are commented out to avoid rerunning long or resource-heavy operations. Uncomment these if running for the first time or retraining is necessary.


## Model Variants
Baseline: Simple CNN and MLP models

Balanced/Unbalanced: Trained with different data distributions

Augmented: Includes random flip, rotation, brightness changes, etc.

Metadata Fusion: Combines image features with patient metadata

Multiclass Classification: Adds support for multiple lesion types

## Results Summary
All models are evaluated using confusion matrices, ROC curves, and classification reports (precision, recall, accuracy, f1, etc). No model had great results, but the best results came from `subset5000-aug-no-unfreeze.ipynb` which had data augmentation without unfreezing base model (MobileNetV2).

## This project is for academic use at AIT Budapest.
