# Heart Disease Classification with Feed-Forward Neural Network

This project applies a feed-forward neural network to predict heart disease based on clinical features. Using a supervised learning approach, we train the model on labeled input-output pairs to make predictions on heart disease diagnosis.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Setup Instructions](#setup-instructions)
5. [Code Structure and Usage](#code-structure-and-usage)
6. [Example Notebook](#example-notebook)
7. [Acknowledgements](#acknowledgements)

---

## Overview

The goal of this project is to build a binary classification model that predicts the presence of heart disease based on health indicators like age, cholesterol levels, and chest pain type. We implement a feed-forward neural network using PyTorch and train it on a labeled dataset of patient health records.

This solution includes:
- A modular neural network setup for easy data preparation, training, and evaluation.
- A Jupyter notebook (`main.ipynb`) demonstrating the code functionality interactively.
- A separate dataset file (`heart.csv`) for data loading.

## Dataset

The dataset contains clinical and demographic features related to heart health. The target variable, labeled as `target`, indicates the presence (`1`) or absence (`0`) of heart disease.

### Features
The dataset includes the following features:
- **age**: Age of the patient
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Chest pain type (categorical: 0, 1, 2, 3)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results (categorical: 0, 1, 2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise-induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment
- **ca**: Number of major vessels (0–3) colored by fluoroscopy
- **thal**: Thalassemia (categorical: 1, 2, 3)
- **target**: Diagnosis of heart disease (1 = disease, 0 = no disease)

## Model Architecture

The neural network is a simple feed-forward architecture:
1. **Input Layer**: Takes in 13 features.
2. **Hidden Layers**: Two fully connected hidden layers with ReLU activation functions.
3. **Output Layer**: A single output neuron with sigmoid activation for binary classification.

The neural network uses binary cross-entropy as the loss function and the Adam optimizer.

## Setup Instructions

### Prerequisites

Make sure you have the following libraries installed:
- **Python** (3.x)
- **PyTorch**: `torch`, `torchvision`
- **Pandas**: `pandas`
- **Scikit-Learn**: `scikit-learn`

To install the necessary libraries, run:
```bash
pip install torch pandas scikit-learn
```

### Files

- `neural_network_module.py`: Contains the main `NeuralNetwork` class and helper functions for data preparation, training, and evaluation.
- `main.ipynb`: A Jupyter notebook demonstrating how to use the module interactively.
- `heart.csv`: The heart disease dataset in CSV format.

## Code Structure and Usage

### 1. `neural_network_module.py`

This module provides a modular design to load, train, and evaluate the neural network model. Key components include:

- **`NeuralNetwork` class**: Defines the neural network architecture.
- **`BuildDataset` function**: Prepares the dataset by splitting it into training and test sets, scaling the features, and converting them to PyTorch tensors.
- **`train_one_by_one` function**: Trains the model one epoch at a time, printing the loss at each epoch.
- **`evaluate` function**: Evaluates the model on the test set and returns accuracy.

#### Example Usage

```python
from neural_network_module import NeuralNetwork, BuildDataset, train_one_by_one, evaluate
import pandas as pd

# Load dataset
data = pd.read_csv("heart.csv")

# Prepare dataset
X_train, X_test, y_train, y_test = BuildDataset(data)

# Initialize and train the model
input_size = X_train.shape[1]
model = NeuralNetwork(input_size=input_size)
train_one_by_one(model, X_train, y_train, epochs=100, learning_rate=0.001)

# Evaluate the model
evaluate(model, X_test, y_test)
```

### 2. `main.ipynb`

This notebook walks through loading the dataset, initializing and training the neural network, and evaluating the model. It is ideal for interactive exploration and understanding of the module.

## Example Notebook

To run the example notebook:

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `main.ipynb`.
3. Follow the steps to see data loading, model training, and evaluation in action.

## Acknowledgements

This solution and dataset are inspired by Stanford's CS229 course material, particularly on supervised learning and neural network applications for classification tasks.

