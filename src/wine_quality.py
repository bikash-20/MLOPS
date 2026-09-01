"""
Wine Quality Classifier: PyTorch Implementation

This is our second project, using PyTorch to classify wine quality.
We frame it as binary classification: "good wine" (quality >= 7) vs "not good wine".

Dataset: UCI Wine Quality Dataset
- ~1,600 samples (red + white wines)
- 11 chemical features
- Quality scores: 0-10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import urllib.request

from utils import banner, data_path, models_path, plots_path, set_seed


set_seed(42)


def download_wine_data():
    """
    Download the Wine Quality dataset from UCI ML Repository.

    Returns:
        pd.DataFrame: Wine quality dataset
    """
    # URL for white wine dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    # Download to local file (absolute path via utils)
    filepath = data_path("winequality-white.csv")

    if not os.path.exists(filepath):
        print(f"Downloading wine quality dataset from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded to: {filepath}")

    # Load data (semicolon-separated)
    df = pd.read_csv(filepath, sep=';')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


class WineNet(nn.Module):
    """
    Neural network for wine quality classification.

    Architecture:
        Input (11) → Hidden (64, ReLU) → Dropout → Hidden (32, ReLU) → Output (2)
    """

    def __init__(self, input_size=11, hidden_size=64, output_size=2):
        super(WineNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        return self.network(x)


def prepare_data(df, binary=True):
    """
    Prepare wine quality data for training.

    Args:
        df (pd.DataFrame): Raw wine quality dataset
        binary (bool): If True, convert to binary classification (good/not good)

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Separate features and target
    X = df.drop('quality', axis=1).values
    y = df['quality'].values

    # Convert to binary classification
    if binary:
        # Good wine: quality >= 7
        y = (y >= 7).astype(int)
        print(f"\nBinary Classification: Good (>=7) vs Not Good (<7)")
        print(f"Class distribution: {np.bincount(y)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Feature dimension: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


def train_model(model, train_loader, test_loader, epochs=100, lr=0.001):
    """
    Train the neural network.

    Args:
        model (nn.Module): Neural network
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        epochs (int): Number of training epochs
        lr (float): Learning rate

    Returns:
        dict: Training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    print("\n🚀 Training Wine Quality Classifier...\n")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Test Loss':<12} {'Test Acc':<12}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Evaluation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_acc = test_correct / test_total

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:<8} {train_loss:<12.4f} {train_acc:<12.4f} {test_loss:<12.4f} {test_acc:<12.4f}")

    return history


def plot_training_history(history, save_path=None):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['test_loss'], label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Wine Quality: Training & Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['test_acc'], label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Wine Quality: Training & Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\n✓ Training history plot saved to: {save_path}")
    plt.close()

    return save_path


def main():
    """Main training pipeline for wine quality classification."""
    banner("WINE QUALITY CLASSIFIER: PYTORCH IMPLEMENTATION")
    print("\n🍷 Task: Binary classification (good wine vs not good wine)")
    print("📚 Framework: PyTorch")
    print("🏗️  Architecture: Input(11) → 64 → 32 → Output(2)\n")

    # Download and load data
    print("📊 Downloading and loading wine quality dataset...")
    df = download_wine_data()

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df, binary=True)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # Create data loaders
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    print("\n🔧 Creating neural network...")
    model = WineNet(input_size=11, hidden_size=64, output_size=2)
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Train model
    history = train_model(model, train_loader, test_loader, epochs=100, lr=0.001)

    # Final evaluation
    banner("FINAL RESULTS")
    print(f"Final Test Accuracy: {history['test_acc'][-1] * 100:.2f}%")
    print(f"Final Test Loss: {history['test_loss'][-1]:.4f}")

    # Plot results
    plot_path = plot_training_history(history, plots_path("wine_training_history.png"))

    # Save model
    model_path = models_path("wine_quality_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to: {model_path}")

    banner("🎉 WINE QUALITY CLASSIFICATION COMPLETE!")


if __name__ == "__main__":
    main()