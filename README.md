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
