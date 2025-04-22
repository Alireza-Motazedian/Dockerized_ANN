#!/usr/bin/env python3
"""
Script to extract sample images from the MNIST dataset.
This script:
1. Loads the MNIST dataset (assuming it's already downloaded)
2. Extracts 20 examples of each digit (0-9)
3. Creates a grid visualization showing examples of each digit
4. Saves individual samples to data/mnist_samples/ folder
5. Creates mnist_samples.csv with features
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Create directories if they don't exist
os.makedirs('data/mnist_samples', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Load MNIST dataset
print("Loading MNIST dataset...")
try:
    # Try to load from saved files first
    X_train = np.load('data/mnist/X_train.npy')
    y_train = np.load('data/mnist/y_train.npy')
except FileNotFoundError:
    # If files don't exist, download the dataset
    print("Saved dataset not found. Downloading MNIST dataset...")
    (X_train, y_train), (_, _) = mnist.load_data()
    X_train = X_train.astype('float32') / 255.0

# Extract sample images (20 for each digit)
print("Extracting sample images...")
samples_per_digit = 20
sample_images = []
sample_labels = []
sample_features = []

# Create a figure for the grid visualization
plt.figure(figsize=(15, 10))
plt.suptitle('MNIST Sample Digits', fontsize=16)

# Process each digit (0-9)
for digit in range(10):
    # Find all images of this digit
    digit_indices = np.where(y_train == digit)[0]
    
    # Select 20 random images of this digit
    selected_indices = np.random.choice(digit_indices, samples_per_digit, replace=False)
    
    # Store these samples
    for i, idx in enumerate(selected_indices):
        # Add to our sample lists
        img = X_train[idx]
        sample_images.append(img)
        sample_labels.append(digit)
        
        # Save individual image
        plt.imsave(f'data/mnist_samples/digit_{digit}_sample_{i+1}.png', img, cmap='gray')
        
        # Calculate features for the CSV
        # Simple features: mean, std, min, max pixel values
        flattened_img = img.flatten()
        features = {
            'digit': digit,
            'sample_id': i+1,
            'filename': f'digit_{digit}_sample_{i+1}.png',
            'mean_pixel': np.mean(flattened_img),
            'std_pixel': np.std(flattened_img),
            'min_pixel': np.min(flattened_img),
            'max_pixel': np.max(flattened_img)
        }
        sample_features.append(features)
        
        # Add to the grid visualization (show 5 samples per digit)
        if i < 5:  # Only show 5 examples per digit in the grid
            plt.subplot(10, 5, digit * 5 + i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"Digit: {digit}")
            plt.axis('off')

# Save the grid visualization
print("Saving grid visualization...")
plt.tight_layout()
plt.savefig('figures/mnist_samples.png')
plt.close()

# Create and save CSV with features
print("Creating features CSV...")
features_df = pd.DataFrame(sample_features)
features_df.to_csv('data/mnist_samples/mnist_samples.csv', index=False)

print("Sample extraction complete.") 