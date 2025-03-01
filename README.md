# SVHN

SVHN Digit Classification using CNN

Overview

This project implements a Convolutional Neural Network (CNN) to classify digits from the Street View House Numbers (SVHN) dataset. The model is trained using TensorFlow and Keras, and it utilizes data augmentation, batch normalization, and dropout to improve generalization.

Dataset

The SVHN dataset consists of digit images collected from street signs. It is loaded using tensorflow_datasets and split into training and test sets.

Features

Data Normalization using mean and standard deviation.

One-Hot Encoding for labels.

Data Augmentation using ImageDataGenerator.

CNN Architecture with multiple convolutional and pooling layers.

Batch Normalization & Dropout to improve training stability.

Adam Optimizer for efficient training.

Model Evaluation using test accuracy and loss.

Visualization of predictions and training history.

Dependencies

To run this project, install the following dependencies:

pip install numpy matplotlib tensorflow tensorflow-datasets scikit-learn

Steps to Run the Project

Load the Dataset: The SVHN dataset is loaded using TensorFlow Datasets (tfds).

Preprocess Data:

Normalize images using mean and standard deviation.

Convert labels to one-hot encoding.

Data Augmentation:

Rotation, width/height shifts, and zooming are applied to improve generalization.

Build CNN Model:

Multiple Conv2D, MaxPooling2D, BatchNormalization, and Dense layers are used.

Compile & Train Model:

Optimizer: Adam

Loss function: Categorical Crossentropy

Train the model using an augmented dataset.

Evaluate Model:

Test the model on unseen SVHN test data.

Compute and display accuracy.

Visualize Results:

Plot training accuracy and loss.

Display sample predictions.

Model Summary

Input Shape: (32, 32, 3)

Architecture:

Conv2D (64 filters, ReLU, Batch Normalization)

MaxPooling2D

Conv2D (128 filters, ReLU, Batch Normalization)

MaxPooling2D

Conv2D (256 filters, ReLU, Batch Normalization)

MaxPooling2D

Flatten, Dense (512, ReLU), Dropout (0.5)

Output Layer: Dense (10 classes, Softmax)

Results

The model is trained for 27 epochs.

Achieved test accuracy of 0.9542.

Training history is plotted for analysis.

Sample test images are displayed with predicted labels.




