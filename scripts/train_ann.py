#!/usr/bin/env python3
"""
Script to build and train an Artificial Neural Network for MNIST digit recognition.
This script:
1. Loads the preprocessed MNIST dataset
2. Builds an ANN with multiple hidden layers and dropout
3. Trains the model with early stopping
4. Generates visualizations (confusion matrix, training history)
5. Saves the trained model to the models directory
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Load preprocessed MNIST data
print("Loading preprocessed MNIST data...")
try:
    X_train = np.load('data/mnist/X_train_flattened.npy')
    X_test = np.load('data/mnist/X_test_flattened.npy')
    y_train = np.load('data/mnist/y_train.npy')
    y_test = np.load('data/mnist/y_test.npy')
except FileNotFoundError:
    print("Preprocessed data not found. Please run data_prep.py first.")
    exit(1)

# Convert labels to one-hot encoding
y_train_onehot = to_categorical(y_train, 10)
y_test_onehot = to_categorical(y_test, 10)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Training labels shape: {y_train_onehot.shape}")
print(f"Test labels shape: {y_test_onehot.shape}")

# Build the ANN model
print("Building the ANN model...")
model = Sequential([
    # Input layer
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    
    # Hidden layers
    Dense(256, activation='relu'),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    Dropout(0.2),
    
    # Output layer - 10 classes (digits 0-9)
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Set up callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath='models/mnist_ann_best.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]

# Train the model
print("Training the model...")
history = model.fit(
    X_train, y_train_onehot,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model
print("Evaluating the model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Save the final model
model.save('models/mnist_ann_final.h5')
print("Model saved to models/mnist_ann_final.h5")

# Generate predictions for confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Create and save confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('figures/confusion_matrix.png')
plt.close()

# Create and save training history plots
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('figures/training_history.png')
plt.close()

# Generate classification report
report = classification_report(y_test, y_pred_classes, digits=4)
print("\nClassification Report:")
print(report)

# Save some predictions visualization
n_samples = 25  # 5x5 grid
plt.figure(figsize=(12, 12))
for i in range(n_samples):
    plt.subplot(5, 5, i+1)
    # Get random test image
    idx = np.random.randint(0, X_test.shape[0])
    img = X_test[idx].reshape(28, 28)
    true_label = y_test[idx]
    pred_label = y_pred_classes[idx]
    
    # Show image
    plt.imshow(img, cmap='gray')
    
    # Set title color based on prediction correctness
    title_color = 'green' if true_label == pred_label else 'red'
    plt.title(f"True: {true_label}, Pred: {pred_label}", color=title_color)
    plt.axis('off')

plt.tight_layout()
plt.savefig('figures/prediction_samples.png')
plt.close()

print("Training and evaluation complete.") 