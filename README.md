# Deep-Learning-Group-Project

## Students: Jonah Levine, Ethan Sandoval, Kevin Chen

## Group Name: Medical AI Sorcerers

# Project Title: Classifying Skin Cancer Images

## Abstract
Skin cancer is one of the most common and potentially deadly forms of cancer, making early and accurate diagnosis crucial. This deep learning-based image classification project makes use of medical imaging and metadata to predict whether a given skin lesion is benign or malignant. Using a dataset from the International Skin Imaging Collaboration (ISIC), which includes over 500,000 entries, we employ convolutional neural networks to analyze skin images. Metadata features such as patient age, anatomical site of the picture, lesion size, and image type are incorporated to enhance model performance. The goal is to develop a robust AI model that assists medical professionals in providing reliable skin cancer classification, improving early detection and guiding clinical decisions.

## Running the Code
1. Run PreProcessing.ipynb to download the data and process it
2. Run each of the "Basic" programs to get the baseline models for our data
3. Run the model "multi_model" notebook for the full deep learning model using both MLP and CNN models, using both metadata and image data.

For clarification the image models are trained with just a portion of the dataset in order to have an equal number of positive and negative observations. For the final submission we plan to use data augmentation for this so that we can use more of our data.

# Skin Cancer Detection Using Deep Learning

This project builds and evaluates multiple deep learning models for classifying skin lesions as **benign** or **malignant** using both image data and associated metadata. The models are trained and tested using various configurations including data augmentation, ResNet-based transfer learning, and metadata fusion.

## Project Structure

- `PreProcessing.ipynb` – Prepares the data, including image resizing and metadata extraction.
- `Basic - Metadata MLP Baseline.ipynb` – Simple MLP model using metadata only.
- `Basic - Metadata Linear Regression Baseline.ipynb` – Linear model using metadata only.
- `Basic - Image CNN Baseline.ipynb` – Simple CNN for image-based classification.
- `subset5000.ipynb` – CNN model with balanced training and evaluation datasets.
- `subset5000-unbalancedval.ipynb` – Trained on balanced data, validated on unbalanced data.
- `subset5000-aug.ipynb` – Includes image data augmentation.
- `subset5000-unbalanced.ipynb` – Trained and validated on unbalanced data.
- `subset5000-unbalanced-weighted.ipynb` – Uses class weighting to address imbalance.
- `multi_model.ipynb` – Final model integrating image and metadata inputs.
- `TESTsubset5000-aug-classesTEST.ipynb` – Multiclass classification with data augmentation.
- `requirements.txt` – List of required Python packages.

## 🛠 Setup Instructions

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

Choose baseline models from the "Basic - ..." notebooks

Proceed to subset5000... variants and multi_model.ipynb

Note: Some code cells are commented out to avoid rerunning long or resource-heavy operations. Uncomment these if running for the first time or retraining is necessary.


## Model Variants
Baseline: Simple CNN and MLP models

Balanced/Unbalanced: Trained with different data distributions

Augmented: Includes random flip, rotation, brightness changes, etc.

Transfer Learning: ResNet50 pre-trained on ImageNet

Metadata Fusion: Combines image features with patient metadata

Multiclass Classification: Adds support for multiple lesion types

## Results Summary
All models are evaluated using confusion matrices and ROC curves. The final integrated model with ResNet50 and metadata inputs achieved the best results, demonstrating the value of multimodal learning for skin cancer detection.

## This project is for academic use at AIT Budapest.
