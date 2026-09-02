"""2-layer NumPy neural network for the Iris dataset.

Architecture: Input (4) -> Hidden (10, ReLU) -> Output (3, Softmax).
Loss: categorical cross-entropy. Optimizer: vanilla gradient descent.
Built without deep-learning frameworks — math is explicit.
"""

from __future__ import annotations

import numpy as np


class NeuralNetwork:
    """A 2-layer neural network for multi-class classification.

    Attributes:
        W1 (np.ndarray): Weight matrix for hidden layer, shape ``(hidden, in)``.
        b1 (np.ndarray): Bias vector for hidden layer, shape ``(hidden, 1)``.
        W2 (np.ndarray): Weight matrix for output layer, shape ``(out, hidden)``.
        b2 (np.ndarray): Bias vector for output layer, shape ``(out, 1)``.
        loss_history (list[float]): Per-epoch training loss.
        accuracy_history (list[float]): Per-epoch training accuracy.
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 10,
        output_size: int = 3,
        learning_rate: float = 0.1,
    ) -> None:
        """Initialize with He-scaled random weights.

        Args:
            input_size: Number of input features (4 for Iris).
            hidden_size: Hidden-layer width.
            output_size: Number of output classes (3 for Iris).
            learning_rate: Step size for gradient descent.
        """
        rng = np.random.default_rng()
        self.W1 = rng.standard_normal((hidden_size, input_size)) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = rng.standard_normal((output_size, hidden_size)) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((output_size, 1))
        self.learning_rate = learning_rate

        self.loss_history: list[float] = []
        self.accuracy_history: list[float] = []

    # ---- Activation functions ---------------------------------------------

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """ReLU activation: ``max(0, z)``."""
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU: ``1`` if ``z > 0`` else ``0``."""
        return (z > 0).astype(float)

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax (subtracts max per column)."""
        z_shifted = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    # ---- Loss -------------------------------------------------------------

    @staticmethod
    def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Categorical cross-entropy with epsilon-clipping for stability."""
        n = y_true.shape[1]
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / n
        return float(loss)

    # ---- Forward / backward passes ---------------------------------------

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run forward propagation.

        Args:
            X: Input batch, shape ``(input_size, N)``.

        Returns:
            Tuple ``(z1, a1, z2, a2)`` of intermediate values for backprop.
        """
        z1 = np.dot(self.W1, X) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = self.softmax(z2)
        return z1, a1, z2, a2

    def backward(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        z1: np.ndarray,
        a1: np.ndarray,
        z2: np.ndarray,
        a2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run backpropagation to compute gradients.

        Returns:
            Tuple ``(dW1, db1, dW2, db2)``.
        """
        n = X.shape[1]

        # Output layer: beautiful simplification when cross-entropy + softmax.
        dz2 = a2 - Y
        dW2 = np.dot(dz2, a1.T) / n
        db2 = np.sum(dz2, axis=1, keepdims=True) / n

        # Hidden layer.
        da1 = np.dot(self.W2.T, dz2)
        dz1 = da1 * self.relu_derivative(z1)
        dW1 = np.dot(dz1, X.T) / n
        db1 = np.sum(dz1, axis=1, keepdims=True) / n

        return dW1, db1, dW2, db2

    def update_parameters(
        self,
        dW1: np.ndarray,
        db1: np.ndarray,
        dW2: np.ndarray,
        db2: np.ndarray,
    ) -> None:
        """Apply vanilla gradient-descent updates."""
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    # ---- Training / inference --------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        epochs: int = 1000,
        verbose: bool = True,
        log_every: int = 100,
    ) -> None:
        """Train for ``epochs`` full-batch gradient-descent steps."""
        for epoch in range(1, epochs + 1):
            z1, a1, z2, a2 = self.forward(X_train)
            loss = self.cross_entropy_loss(a2, Y_train)
            predictions = np.argmax(a2, axis=0)
            labels = np.argmax(Y_train, axis=0)
            accuracy = float(np.mean(predictions == labels))

            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)

            dW1, db1, dW2, db2 = self.backward(X_train, Y_train, z1, a1, z2, a2)
            self.update_parameters(dW1, db1, dW2, db2)

            if verbose and (epoch % log_every == 0 or epoch == 1):
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices for inputs ``X``."""
        _, _, _, a2 = self.forward(X)
        return np.argmax(a2, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for inputs ``X``."""
        _, _, _, a2 = self.forward(X)
        return a2


__all__ = ["NeuralNetwork"]