# Simple Neural Network Implementation

This project demonstrates a basic neural network implementation in Python using NumPy and pandas.

## Overview

The neural network consists of the following components:
1. **DataFrame Setup**: A pandas DataFrame is created containing columns `'cgpa'`, `'profile_score'`, and `'lpa'`.
2. **Network Functions**: Functions for initializing parameters, performing forward propagation, and applying activation functions are defined.
3. **Forward Propagation**: A simple forward pass through a network with multiple layers, including ReLU activations for hidden layers.

## Code Explanation

### 1. **Data Preparation**
A pandas DataFrame `df` is created with three columns:
- `'cgpa'`
- `'profile_score'`
- `'lpa'`

### 2. **Functions**

- **`initialize_parameters(layers_dims)`**: 
  - Initializes weights (`W`) and biases (`b`) for each layer based on the given `layers_dims`. 
  - Weights are set to small values (`0.1`), and biases are initialized to zero.

- **`linear_forward(A_prev, W, b)`**:
  - Computes the linear transformation \( Z = W^T A_{\text{prev}} + b \), where `A_prev` is the activations from the previous layer.

- **`relu(Z)`**:
  - Applies the ReLU activation function: sets all negative values in `Z` to zero.

- **`L_layer_forward(X, parameters)`**:
  - Performs forward propagation through all layers.
  - For each hidden layer, it computes \( Z \) and applies the ReLU activation.
  - For the output layer, only the linear transformation is applied (no activation).

### 3. **Example Use**

- The network has the following architecture: 
  - 3 input features
  - 2 hidden layers (with 4 and 2 neurons, respectively)
  - 1 output neuron
  
- The input data is defined as `X = [[8], [7], [6]]`, and the output of the forward pass is `AL = [[0.168]]`.

## How It Works

The code demonstrates the initialization of parameters, performing forward propagation through multiple layers, and calculating the output. The ReLU activation function ensures that only positive values pass through the network, while negative values are discarded.

## Requirements

- Python 3.x
- numpy
- pandas

## Usage

1. Install the required libraries:
   ```bash
   pip install numpy pandas
