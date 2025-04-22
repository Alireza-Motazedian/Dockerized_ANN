#!/usr/bin/env python3
"""
Script to download and preprocess the MNIST dataset.
This script:
1. Downloads the MNIST dataset using TensorFlow/Keras
2. Preprocesses the data (normalizes, reshapes)
3. Saves the processed data to the data/mnist/ directory
"""

import os
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Create directory if it doesn't exist
os.makedirs('data/mnist', exist_ok=True)

# Download and load MNIST dataset
print("Downloading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Print dataset shapes
print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Preprocess data
print("Preprocessing data...")

# Normalize pixel values to range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape data for the model (flattening the images)
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

print(f"Flattened training data shape: {X_train_flattened.shape}")
print(f"Flattened test data shape: {X_test_flattened.shape}")

# Save processed data to files
print("Saving processed data...")
np.save('data/mnist/X_train.npy', X_train)
np.save('data/mnist/y_train.npy', y_train)
np.save('data/mnist/X_test.npy', X_test)
np.save('data/mnist/y_test.npy', y_test)
np.save('data/mnist/X_train_flattened.npy', X_train_flattened)
np.save('data/mnist/X_test_flattened.npy', X_test_flattened)

print("Data preparation complete.") 