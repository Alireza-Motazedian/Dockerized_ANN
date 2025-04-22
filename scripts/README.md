<div align="center">
  <h1>MNIST Neural Network Scripts</h1>
</div>

# Table of Contents 
<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#1-overview"><i><b>1. Overview</b></i></a>
</div>
&nbsp;

<details>
  <summary><a href="#2-scripts"><i><b>2. Scripts</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#21-data_preppy">2.1. data_prep.py</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#22-extract_sample_imagespy">2.2. extract_sample_images.py</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#23-train_annpy">2.3. train_ann.py</a><br>
  </div>
</details>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#3-running-order"><i><b>3. Running Order</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#4-dependencies"><i><b>4. Dependencies</b></i></a>
</div>
&nbsp;

# 1. Overview

This directory contains Python scripts for the MNIST digit recognition project. These scripts handle data preparation, feature extraction, model training, and evaluation. They can be run independently from the command line or from within the Jupyter notebook.

# 2. Scripts

## 2.1. data_prep.py

Downloads and preprocesses the MNIST dataset.

**Functionality:**
- Downloads the MNIST dataset using TensorFlow/Keras
- Normalizes pixel values to range [0, 1]
- Reshapes data for neural network input (flattens 28×28 images to 784 vectors)
- Saves processed data to the `data/mnist/` directory

**Usage:**
```bash
python scripts/data_prep.py
```

**Output:**
- `data/mnist/X_train.npy`: Training images (60,000 × 28 × 28)
- `data/mnist/y_train.npy`: Training labels (60,000)
- `data/mnist/X_test.npy`: Test images (10,000 × 28 × 28)
- `data/mnist/y_test.npy`: Test labels (10,000)
- `data/mnist/X_train_flattened.npy`: Flattened training images (60,000 × 784)
- `data/mnist/X_test_flattened.npy`: Flattened test images (10,000 × 784)

## 2.2. extract_sample_images.py

Extracts and saves sample images from the MNIST dataset.

**Functionality:**
- Loads the MNIST dataset (either from saved files or downloads it)
- Extracts 20 examples of each digit (0-9)
- Creates a grid visualization showing examples of each digit
- Saves individual sample images to `data/mnist_samples/` folder
- Creates a CSV file with image features

**Usage:**
```bash
python scripts/extract_sample_images.py
```

**Output:**
- `data/mnist_samples/digit_X_sample_Y.png`: Individual sample images (200 total)
- `data/mnist_samples/mnist_samples.csv`: CSV file with features for each sample
- `figures/mnist_samples.png`: Grid visualization of sample digits

## 2.3. train_ann.py

Builds, trains, and evaluates the ANN model for MNIST digit recognition.

**Functionality:**
- Loads preprocessed MNIST data
- Builds an ANN with multiple hidden layers and dropout for regularization
- Trains the model with early stopping and model checkpointing
- Evaluates the model on test data
- Generates visualizations (confusion matrix, training history, predictions)
- Saves the trained model to the `models/` directory

**Usage:**
```bash
python scripts/train_ann.py
```

**Output:**
- `models/mnist_ann_best.h5`: Best model (highest validation accuracy)
- `models/mnist_ann_final.h5`: Final trained model
- `figures/confusion_matrix.png`: Confusion matrix visualization
- `figures/training_history.png`: Training/validation accuracy and loss plots
- `figures/prediction_samples.png`: Sample predictions visualization

# 3. Running Order

The scripts should be run in the following order:

1. `data_prep.py`: Prepare the dataset
2. `extract_sample_images.py`: Create sample visualizations
3. `train_ann.py`: Train and evaluate the model

# 4. Dependencies

All scripts depend on the Python libraries specified in the project's `requirements.txt` file, including:
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Seaborn 