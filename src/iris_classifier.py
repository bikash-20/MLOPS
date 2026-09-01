"""
Iris Classifier: 2-Layer Neural Network from Scratch (NumPy Only)

This implementation builds a neural network using only NumPy to classify
the three species of Iris flowers. No deep learning frameworks are used.

Architecture:
    Input (4) → Hidden (10, ReLU) → Output (3, Softmax)

Mathematical Foundation:
    - Forward: z = Wx + b, a = activation(z)
    - Loss: Categorical Cross-Entropy
    - Backprop: Chain rule to compute gradients
    - Update: Gradient Descent
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

from utils import banner, models_path, plots_path, set_seed


set_seed(42)


class NeuralNetwork:
    """
    A 2-layer neural network for multi-class classification.

    Attributes:
        W1 (np.ndarray): Weight matrix for hidden layer, shape (hidden_size, input_size)
        b1 (np.ndarray): Bias vector for hidden layer, shape (hidden_size, 1)
        W2 (np.ndarray): Weight matrix for output layer, shape (output_size, hidden_size)
        b2 (np.ndarray): Bias vector for output layer, shape (output_size, 1)
    """

    def __init__(self, input_size=4, hidden_size=10, output_size=3, learning_rate=0.01):
        """
        Initialize the neural network with random weights (He initialization).

        Args:
            input_size (int): Number of input features (4 for Iris)
            hidden_size (int): Number of neurons in hidden layer
            output_size (int): Number of classes (3 for Iris)
            learning_rate (float): Learning rate for gradient descent
        """
        # He initialization for ReLU activation
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((output_size, 1))
        self.learning_rate = learning_rate

        # Store training history
        self.loss_history = []
        self.accuracy_history = []

    def relu(self, z):
        """ReLU activation function: max(0, z)"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """Derivative of ReLU: 1 if z > 0, else 0"""
        return (z > 0).astype(float)

    def softmax(self, z):
        """
        Softmax activation function.
        Numerically stable version (subtracts max for stability).

        Args:
            z (np.ndarray): Pre-activation values, shape (output_size, N)

        Returns:
            np.ndarray: Probability distribution, shape (output_size, N)
        """
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        """
        Categorical cross-entropy loss.

        L = -1/N * sum(y * log(y_pred))

        Args:
            y_pred (np.ndarray): Predicted probabilities, shape (output_size, N)
            y_true (np.ndarray): True labels (one-hot), shape (output_size, N)

        Returns:
            float: Average loss
        """
        N = y_true.shape[1]
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / N
        return loss

    def forward(self, X):
        """
        Forward propagation through the network.

        Args:
            X (np.ndarray): Input data, shape (input_size, N)

        Returns:
            tuple: (z1, a1, z2, a2) - intermediate values for backprop
        """
        # Layer 1: Input → Hidden
        z1 = np.dot(self.W1, X) + self.b1
        a1 = self.relu(z1)

        # Layer 2: Hidden → Output
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = self.softmax(z2)

        return z1, a1, z2, a2

    def backward(self, X, Y, z1, a1, z2, a2):
        """
        Backpropagation: compute gradients using chain rule.

        Args:
            X (np.ndarray): Input data, shape (input_size, N)
            Y (np.ndarray): True labels (one-hot), shape (output_size, N)
            z1, a1, z2, a2: Intermediate values from forward pass

        Returns:
            tuple: Gradients (dW1, db1, dW2, db2)
        """
        N = X.shape[1]

        # Output layer gradients
        # dL/dz2 = y_pred - y_true (beautiful simplification!)
        dz2 = a2 - Y
        dW2 = np.dot(dz2, a1.T) / N
        db2 = np.sum(dz2, axis=1, keepdims=True) / N

        # Hidden layer gradients
        da1 = np.dot(self.W2.T, dz2)
        dz1 = da1 * self.relu_derivative(z1)
        dW1 = np.dot(dz1, X.T) / N
        db1 = np.sum(dz1, axis=1, keepdims=True) / N

        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2):
        """
        Update weights and biases using gradient descent.

        W := W - alpha * dW
        b := b - alpha * db
        """
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X_train, Y_train, epochs=1000, verbose=True):
        """
        Train the neural network.

        Args:
            X_train (np.ndarray): Training data, shape (input_size, N)
            Y_train (np.ndarray): Training labels (one-hot), shape (output_size, N)
            epochs (int): Number of training iterations
            verbose (bool): Print progress every 100 epochs
        """
        for epoch in range(1, epochs + 1):
            # Forward pass
            z1, a1, z2, a2 = self.forward(X_train)

            # Compute loss
            loss = self.cross_entropy_loss(a2, Y_train)

            # Compute accuracy
            predictions = np.argmax(a2, axis=0)
            labels = np.argmax(Y_train, axis=0)
            accuracy = np.mean(predictions == labels)

            # Store history
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)

            # Backward pass
            dW1, db1, dW2, db2 = self.backward(X_train, Y_train, z1, a1, z2, a2)

            # Update parameters
            self.update_parameters(dW1, db1, dW2, db2)

            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X (np.ndarray): Input data, shape (input_size, N)

        Returns:
            np.ndarray: Predicted class labels, shape (N,)
        """
        _, _, _, a2 = self.forward(X)
        return np.argmax(a2, axis=0)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X (np.ndarray): Input data, shape (input_size, N)

        Returns:
            np.ndarray: Class probabilities, shape (output_size, N)
        """
        _, _, _, a2 = self.forward(X)
        return a2


def load_and_prepare_data():
    """
    Load Iris dataset and prepare it for training.

    Returns:
        tuple: (X_train, X_test, Y_train, Y_test, scaler)
    """
    # Load dataset
    iris = load_iris()
    X = iris.data  # Shape: (150, 4)
    y = iris.target  # Shape: (150,)

    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # One-hot encode labels
    # y=0 → [1,0,0], y=1 → [0,1,0], y=2 → [0,0,1]
    Y_train = np.eye(3)[y_train].T  # Shape: (3, N_train)
    Y_test = np.eye(3)[y_test].T    # Shape: (3, N_test)

    # Transpose X for our convention: (features, samples)
    X_train = X_train.T
    X_test = X_test.T

    print(f"Training set: {X_train.shape[1]} samples")
    print(f"Test set: {X_test.shape[1]} samples")
    print(f"Feature dimension: {X_train.shape[0]}")
    print(f"Classes: {iris.target_names}")

    return X_train, X_test, Y_train, Y_test, scaler


def plot_training_history(nn, save_path=None):
    """
    Plot training loss and accuracy over epochs.

    Args:
        nn (NeuralNetwork): Trained neural network
        save_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(nn.loss_history, color='#e74c3c', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=12)
    ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(nn.accuracy_history, color='#27ae60', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\n✓ Training history plot saved to: {save_path}")
    plt.close()

    return save_path


def evaluate_model(nn, X_test, Y_test, y_test):
    """
    Evaluate the trained model on test data.

    Args:
        nn (NeuralNetwork): Trained neural network
        X_test (np.ndarray): Test features
        Y_test (np.ndarray): Test labels (one-hot)
        y_test (np.ndarray): Test labels (integer)

    Returns:
        float: Test accuracy
    """
    # Make predictions
    predictions = nn.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 60)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Correct Predictions: {np.sum(predictions == y_test)}/{len(y_test)}")

    # Per-class accuracy
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    print("\nPer-Class Performance:")
    for i, name in enumerate(class_names):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(predictions[mask] == y_test[mask])
            print(f"  {name:12s}: {class_acc * 100:.2f}% ({np.sum(predictions[mask] == y_test[mask])}/{np.sum(mask)})")

    # Show some predictions
    print("\nSample Predictions (first 10 test samples):")
    print(f"{'True':<12} {'Predicted':<12} {'Confidence':<12}")
    print("-" * 36)
    probabilities = nn.predict_proba(X_test)
    for i in range(min(10, len(y_test))):
        true_class = class_names[y_test[i]]
        pred_class = class_names[predictions[i]]
        confidence = probabilities[predictions[i], i] * 100
        print(f"{true_class:<12} {pred_class:<12} {confidence:.2f}%")

    return accuracy


def main():
    """Main function to run the complete training pipeline."""
    banner("IRIS CLASSIFIER: 2-LAYER NEURAL NETWORK FROM SCRATCH")
    print("\n🧠 Architecture: Input(4) → Hidden(10, ReLU) → Output(3, Softmax)")
    print("📚 Implementation: Pure NumPy (no deep learning frameworks)")
    print("🎯 Task: Multi-class classification (3 species)\n")

    # Load and prepare data
    print("📊 Loading and preprocessing data...")
    X_train, X_test, Y_train, Y_test, scaler = load_and_prepare_data()

    # Create neural network
    print("\n🔧 Creating neural network...")
    nn = NeuralNetwork(
        input_size=4,
        hidden_size=10,
        output_size=3,
        learning_rate=0.1
    )

    # Train the network
    print("\n🚀 Training neural network...\n")
    nn.train(X_train, Y_train, epochs=1500, verbose=True)

    # Evaluate on test set
    y_test_labels = np.argmax(Y_test, axis=0)
    test_accuracy = evaluate_model(nn, X_test, Y_test, y_test_labels)

    # Plot training history
    print("\n📈 Generating visualizations...")
    plot_path = plot_training_history(nn, plots_path("training_history.png"))

    # Save the model parameters
    print("\n💾 Saving model parameters...")
    model_path = models_path("iris_model.npz")
    np.savez(
        model_path,
        W1=nn.W1, b1=nn.b1, W2=nn.W2, b2=nn.b2,
        loss_history=nn.loss_history,
        accuracy_history=nn.accuracy_history
    )
    print(f"✓ Model saved to: {model_path}")

    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Final Training Loss: {nn.loss_history[-1]:.4f}")
    print("\nNext steps:")
    print("  - Check plots/training_history.png for visualizations")
    print("  - Read docs/03_training_process.md for detailed analysis")
    print("  - Try wine_quality.py for the next project!")


if __name__ == "__main__":
    main()