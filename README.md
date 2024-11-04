# Heart Disease Classification with Feed-Forward Neural Network

This project applies a feed-forward neural network to predict heart disease based on clinical features. Using a supervised learning approach, we leverage a labeled dataset with input-output pairs to train and evaluate the model. This README includes details on the model, dataset, setup instructions, and usage guidance.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Setup Instructions](#setup-instructions)
5. [Code Structure and Usage](#code-structure-and-usage)
6. [Example Notebook](#example-notebook)
7. [Acknowledgements]

---

## Overview

The goal of this project is to build a binary classification model that predicts the presence of heart disease based on various health indicators. A feed-forward neural network is implemented using PyTorch, trained on a labeled heart disease dataset.

This solution includes:
- A modular neural network implementation with separate functions for data preparation, training, and evaluation.
- A Jupyter notebook (`main.ipynb`) that demonstrates the code's functionality in an interactive format.
- A separate dataset file (`heart.csv`) for easy data loading and reuse.

## Dataset

The dataset contains various clinical and demographic features related to heart health. The `target` column is the label, indicating the presence (`1`) or absence (`0`) of heart disease.

### Features
The main features in the dataset include:
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

The neural network is a simple feed-forward architecture with:
1. **Input Layer**: Accepts input features (13 in total).
2. **Hidden Layers**: Two fully connected hidden layers with ReLU activation functions.
3. **Output Layer**: A single output neuron with sigmoid activation for binary classification.

The neural network uses binary cross-entropy as the loss function, optimized with the Adam optimizer.

## Setup Instructions

### Prerequisites

Ensure you have the following libraries installed:
- **Python** (3.x)
- **PyTorch**: `torch`, `torchvision`
- **Pandas**: `pandas`
- **Scikit-Learn**: `scikit-learn`

To install the necessary libraries, run:
```bash
pip install torch pandas scikit-learn


## Acknowledgements

This solution and dataset are inspired by Stanford's CS229 course material, particularly on supervised learning and neural network applications for classification tasks.