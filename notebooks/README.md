<div align="center">
  <h1>MNIST Jupyter Notebooks</h1>
</div>

# Table of Contents 
<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#1-overview"><i><b>1. Overview</b></i></a>
</div>
&nbsp;

<details>
  <summary><a href="#2-notebooks"><i><b>2. Notebooks</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#21-01_data_preparationipynb">2.1. 01_data_preparation.ipynb</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#22-02_model_trainingipynb">2.2. 02_model_training.ipynb</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#23-exploratory_analysisipynb">2.3. exploratory_analysis.ipynb</a><br>
  </div>
</details>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#3-running-the-notebooks"><i><b>3. Running the Notebooks</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#4-notebook-dependencies"><i><b>4. Notebook Dependencies</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#5-best-practices"><i><b>5. Best Practices</b></i></a>
</div>
&nbsp;

# 1. Overview

This directory contains Jupyter notebooks for the MNIST digit recognition project. These notebooks provide an interactive environment to explore the dataset, visualize the data, build and train the Artificial Neural Network (ANN) model, and evaluate its performance.

# 2. Notebooks

## 2.1. 01_data_preparation.ipynb

This notebook focuses on the data loading and preparation steps:

1. **Setup and Import**: Loading necessary libraries and setting up the environment
2. **Data Loading**: Downloading the MNIST dataset using TensorFlow's datasets module
3. **Data Exploration**: Exploring the structure of the dataset, shape, and basic statistics
4. **Data Visualization**: Visualizing sample images from different digit classes
5. **Data Preprocessing**: Normalizing and preparing the data for model training

## 2.2. 02_model_training.ipynb

This notebook covers the modeling, training, and evaluation pipeline:

1. **Data Loading**: Loading the preprocessed MNIST data
2. **Model Architecture**: Creating an Artificial Neural Network with:
   - Input layer (784 neurons)
   - Three hidden layers (512, 256, 128 neurons) with ReLU activation
   - Dropout layers (0.2, 0.3, 0.4) for regularization
   - Output layer (10 neurons) with softmax activation
3. **Model Compilation**: Setting up the optimizer, loss function, and evaluation metrics
4. **Model Training**: Training the model with callbacks for early stopping and model checkpointing
5. **Model Evaluation**: Evaluating performance on the test set with accuracy, confusion matrix, and classification report
6. **Results Visualization**: Plotting training/validation metrics and prediction examples
7. **Model Saving**: Saving the trained model for future use

## 2.3. exploratory_analysis.ipynb

This notebook contains additional exploratory data analysis:
- Basic environment verification
- Simple examples of Python data analysis

# 3. Running the Notebooks

To run these notebooks:

1. Ensure the Docker container is running:
   ```bash
   docker-compose up -d
   ```

2. Access Jupyter via browser:
   ```
   http://localhost:8888
   ```

3. Navigate to the `notebooks` directory and open the desired notebook

4. Execute the notebooks in order:
   - Start with `01_data_preparation.ipynb`
   - Then proceed to `02_model_training.ipynb`

Alternatively, you can connect VS Code to the running container and open the notebooks directly in VS Code's Jupyter extension.

# 4. Notebook Dependencies

The notebooks depend on the following Python libraries (installed in the Docker container):
- TensorFlow 2.15.0
- NumPy 1.26.0
- Pandas 2.1.3
- Matplotlib 3.8.0
- Scikit-learn 1.3.2
- Seaborn 0.13.0

# 5. Best Practices

When working with these notebooks:

- Execute cells in order to avoid dependency issues
- Use the kernel selector to ensure you're using the container's Python kernel
- Save your work regularly
- Consider creating copies before making significant changes
- For large changes, consider developing in a new branch 