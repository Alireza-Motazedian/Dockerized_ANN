<div align="center">
  <h1>MNIST Neural Network Models</h1>
</div>

# Table of Contents 
<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#1-overview"><i><b>1. Overview</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#2-model-files"><i><b>2. Model Files</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#3-model-architecture"><i><b>3. Model Architecture</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#4-training-configuration"><i><b>4. Training Configuration</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#5-performance"><i><b>5. Performance</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#6-loading-a-saved-model"><i><b>6. Loading a Saved Model</b></i></a>
</div>
&nbsp;

# 1. Overview

This directory stores the trained Artificial Neural Network (ANN) models for MNIST digit recognition. The models are saved in HDF5 format, which preserves the model architecture, weights, optimizer state, and training configuration.

# 2. Model Files

- **mnist_ann_best.h5**: The model with the best validation accuracy during training (saved via ModelCheckpoint callback).
- **mnist_ann_final.h5**: The final model after training is complete.

# 3. Model Architecture

The ANN architecture consists of:

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 512)               401920    
                                                                 
 dropout (Dropout)           (None, 512)               0         
                                                                 
 dense_1 (Dense)             (None, 256)               131328    
                                                                 
 dropout_1 (Dropout)         (None, 256)               0         
                                                                 
 dense_2 (Dense)             (None, 128)               32896     
                                                                 
 dropout_2 (Dropout)         (None, 128)               0         
                                                                 
 dense_3 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 567,434
Trainable params: 567,434
Non-trainable params: 0
_________________________________________________________________
```

Key components:
- **Input Layer**: 784 neurons (28×28 flattened images)
- **Hidden Layers**: 3 dense layers with 512, 256, and 128 neurons with ReLU activation
- **Dropout Layers**: For regularization (prevent overfitting)
- **Output Layer**: 10 neurons with softmax activation (one for each digit)

# 4. Training Configuration

The model was trained with:
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 128
- **Early Stopping**: Patience of 10 epochs monitoring validation loss
- **Validation Split**: 20% of training data
- **Metrics**: Accuracy

# 5. Performance

The model typically achieves:
- **Training Accuracy**: ~99%
- **Validation Accuracy**: ~98%
- **Test Accuracy**: ~98%

For detailed performance metrics, refer to the confusion matrix and classification report generated during evaluation.

# 6. Loading a Saved Model

To load and use a saved model in Python:

```python
from tensorflow.keras.models import load_model

# Load the model
model = load_model('models/mnist_ann_best.h5')

# Use the model for prediction
predictions = model.predict(x_test)
``` 