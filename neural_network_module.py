# neural_network_module.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.output = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

def BuildDataset(data, target_column="target", test_size=0.2):
    """
    Splits the dataset into train and test sets and applies scaling.
    """

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32)

    return X_train, X_test, y_train, y_test

def train_one_by_one(model, X_train, y_train, epochs=100, learning_rate=0.001):
    """
    Trains the neural network model one epoch at a time.
    """

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()

        # Forward pass
        y_pred = model(X_train).squeeze()
        loss = criterion(y_pred, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss for the current epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


def evaluate(model, X_test, y_test):
    """
    Evaluates the model on the test set.
    """

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze()
        predictions = (y_pred > 0.5).float()
        accuracy = (predictions == y_test).float().mean().item()
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy
