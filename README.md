# VGG16-Eurosat-Classification

This repository contains a deep learning project that uses a custom VGG16 architecture for classifying satellite images from the EuroSAT dataset. The model is trained to identify 10 different land use classes based on satellite imagery.

## Project Overview

The project implements the following steps:

1. **Data Preparation:**
   - The EuroSAT dataset is organized into training, validation, and testing directories.
   - Image augmentation is performed using `ImageDataGenerator` for better generalization.

2. **Model Architecture:**
   - A custom VGG16 model is built from scratch using TensorFlow and Keras. The model is designed with multiple convolutional layers, followed by max-pooling layers, and fully connected layers.

3. **Training:**
   - The model is trained on the training dataset and validated on the validation dataset. The Adam optimizer is used with categorical cross-entropy loss.

4. **Evaluation:**
   - The model's performance is evaluated on the test dataset. Various metrics such as accuracy, precision, recall, F1-score, and Jaccard index are calculated for each class.

5. **Results:**
   - The results, including a classification report and detailed metrics for each class, are printed and analyzed.

## Dependencies

The following libraries are required to run the project:

- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab (optional)

  Model Architecture
The model is based on the VGG16 architecture, which consists of multiple convolutional layers followed by max-pooling layers and dense layers. The final output layer has 10 neurons with a softmax activation function for multi-class classification.
