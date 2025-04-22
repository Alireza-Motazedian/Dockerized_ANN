<div align="center">
  <h1>MNIST Data Directory</h1>
</div>

# Table of Contents 
<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#1-overview"><i><b>1. Overview</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#2-directory-structure"><i><b>2. Directory Structure</b></i></a>
</div>
&nbsp;

<details>
  <summary><a href="#3-data-files"><i><b>3. Data Files</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#31-mnist-dataset-mnist">3.1. MNIST Dataset (mnist/)</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#32-mnist-samples-mnist_samples">3.2. MNIST Samples (mnist_samples/)</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#33-sample-csv-structure">3.3. Sample CSV Structure</a><br>
  </div>
</details>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#4-generating-the-data"><i><b>4. Generating the Data</b></i></a>
</div>
&nbsp;

# 1. Overview

This directory contains the MNIST dataset and samples for the artificial neural network project. The full MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), with 60,000 for training and 10,000 for testing.

# 2. Directory Structure

```
data/
├── mnist/                # Contains the processed MNIST dataset (not included in git)
│   ├── X_train.npy      # Training images (60,000 × 28 × 28)
│   ├── y_train.npy      # Training labels (60,000)
│   ├── X_test.npy       # Test images (10,000 × 28 × 28)
│   ├── y_test.npy       # Test labels (10,000)
│   ├── X_train_flattened.npy  # Flattened training images (60,000 × 784)
│   └── X_test_flattened.npy   # Flattened test images (10,000 × 784)
└── mnist_samples/       # Contains sample images extracted from MNIST
    ├── digit_0_sample_1.png  # Individual sample images
    ├── digit_0_sample_2.png
    ├── ...
    └── mnist_samples.csv      # CSV with features of sample images
```

# 3. Data Files

## 3.1. MNIST Dataset (mnist/)

**Note**: The full MNIST dataset files are not included in the git repository due to their size. They will be generated when you run the `scripts/data_prep.py` script.

The dataset is preprocessed with the following steps:
- Images are normalized to have pixel values between 0 and 1
- Images are reshaped from 28×28 to 784-length vectors for the neural network

## 3.2. MNIST Samples (mnist_samples/)

This directory contains:
- 20 sample images for each digit (0-9), saved as individual PNG files
- A CSV file (`mnist_samples.csv`) with features calculated for each sample image

## 3.3. Sample CSV Structure

The `mnist_samples.csv` file contains the following columns:
- `digit`: The actual digit (0-9)
- `sample_id`: Sample number for that digit (1-20)
- `filename`: The corresponding image filename
- `mean_pixel`: Average pixel value (brightness)
- `std_pixel`: Standard deviation of pixel values
- `min_pixel`: Minimum pixel value
- `max_pixel`: Maximum pixel value

# 4. Generating the Data

To generate the MNIST dataset files:

```bash
python scripts/data_prep.py
```

To generate the sample images:

```bash
python scripts/extract_sample_images.py
``` 