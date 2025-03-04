# -*- coding: utf-8 -*-
"""svhn.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dKmjoC3wMepSor_gz_FOG-5HmbS3AHUR
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#Load SVHN Dataset
dataset, info = tfds.load("svhn_cropped", split=["train", "test"], as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset

def dataset_to_numpy(dataset):
    images, labels = [], []
    for img, label in tfds.as_numpy(dataset):
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

X_train, y_train = dataset_to_numpy(train_dataset)
X_test, y_test = dataset_to_numpy(test_dataset)

# Normalize the Data (Using Mean & Std)
mean = np.mean(X_train, axis=(0,1,2))
std = np.std(X_train, axis=(0,1,2))

X_train = (X_train - mean) / (std + 1e-7)
X_test = (X_test - mean) / (std + 1e-7)

#  One-Hot Encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#  Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

#  Build CNN Model
model = Sequential([
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Helps prevent overfitting
    Dense(10, activation='softmax')  # 10 output classes
])

#  Compile Model with Adam Optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  split data into training and validation sets
X_train_split, X_value, y_train_split, y_value = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# training data
train_generator = datagen.flow(X_train_split, y_train_split, batch_size=64)

# Train Model using train_generator and validation_data
history = model.fit(train_generator,
                    epochs=27,
                    validation_data=(X_value, y_value),  # Pass validation set separately
                    verbose=1)

# Evaluate Model on Test Data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f'Test Accuracy: {test_acc:.4f}')

# Plot Training History
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Model Loss')

plt.show()

# Visualize Some Predictions (Fixed Clipping Issue)
predictions = model.predict(X_test)

# Denormalize images before displaying
X_test_denorm = (X_test * std) + mean  # Reverse normalization

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.clip(X_test_denorm[i] / 255.0, 0, 1))  # Ensure values are in [0,1]
    plt.xlabel(f"True: {np.argmax(y_test[i])}, Pred: {np.argmax(predictions[i])}")
plt.show()

