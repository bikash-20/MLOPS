# Neural Network Theory: Mathematical Foundations

## 🧠 What is a Neural Network?

A neural network is a computational model inspired by biological neurons. It consists of interconnected nodes (neurons) organized in layers that learn to map inputs to outputs through adjustable weights.

### Biological Inspiration
- **Dendrites** → Input connections
- **Cell body** → Summation and activation
- **Axon** → Output signal

## 🏗️ Architecture: 2-Layer Network for Iris

For our Iris classifier, we'll build a simple but powerful architecture:

```
Input Layer (4 neurons)    →    Hidden Layer (10 neurons)    →    Output Layer (3 neurons)
[sepal len, sepal wid,    →    [h1, h2, h3, ..., h10]      →    [P(setosa), P(versicolor),
 petal len, petal wid]                                                  P(virginica)]
```

### Layer Dimensions
- **Input layer**: 4 neurons (one per feature)
- **Hidden layer**: 10 neurons (with ReLU activation)
- **Output layer**: 3 neurons (with softmax activation)

## 📐 Mathematical Formulation

### 1. **Forward Propagation**

#### Layer 1 (Hidden Layer)
$$z^{(1)} = W^{(1)} x + b^{(1)}$$

Where:
- $x \in \mathbb{R}^{4}$ is the input vector
- $W^{(1)} \in \mathbb{R}^{10 \times 4}$ is the weight matrix
- $b^{(1)} \in \mathbb{R}^{10}$ is the bias vector
- $z^{(1)} \in \mathbb{R}^{10}$ is the pre-activation

**Activation (ReLU):**
$$a^{(1)}_i = \max(0, z^{(1)}_i) = \begin{cases} z^{(1)}_i & \text{if } z^{(1)}_i > 0 \\ 0 & \text{otherwise} \end{cases}$$

Why ReLU?
- Computationally simple
- Mitigates vanishing gradient problem
- Introduces non-linearity

#### Layer 2 (Output Layer)
$$z^{(2)} = W^{(2)} a^{(1)} + b^{(2)}$$

Where:
- $W^{(2)} \in \mathbb{R}^{3 \times 10}$
- $b^{(2)} \in \mathbb{R}^{3}$
- $z^{(2)} \in \mathbb{R}^{3}$

**Activation (Softmax):**
$$\hat{y}_k = \frac{e^{z^{(2)}_k}}{\sum_{j=1}^{3} e^{z^{(2)}_j}}$$

Properties:
- Outputs sum to 1: $\sum_k \hat{y}_k = 1$
- Each $\hat{y}_k \in [0, 1]$ (can be interpreted as probability)
- Differentiable, making backpropagation possible

## 🎯 Loss Function: Cross-Entropy

For multi-class classification, we use **categorical cross-entropy**:

$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$$

Where:
- $N$ = number of samples
- $K$ = number of classes (3 for Iris)
- $y_{ik}$ = 1 if sample $i$ belongs to class $k$, else 0
- $\hat{y}_{ik}$ = predicted probability of class $k$ for sample $i$

### Intuition
- If prediction is **correct and confident** (e.g., $\hat{y} = 0.99$, $y = 1$): $L \approx 0.01$ ✓
- If prediction is **wrong and confident** (e.g., $\hat{y} = 0.99$, $y = 0$): $L \approx 4.6$ ✗
- The loss **penalizes wrong confident predictions** heavily

## 🔄 Backpropagation: The Chain Rule

This is where the magic happens. We compute gradients using the chain rule of calculus.

### Output Layer Gradients

**Derivative of softmax + cross-entropy** (beautiful simplification):
$$\frac{\partial L}{\partial z^{(2)}} = \hat{y} - y$$

This is one of the most elegant results in deep learning!

**Weight gradient:**
$$\frac{\partial L}{\partial W^{(2)}} = \frac{1}{N} (\hat{y} - y) (a^{(1)})^T$$

**Bias gradient:**
$$\frac{\partial L}{\partial b^{(2)}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)$$

### Hidden Layer Gradients

**Error signal (delta):**
$$\delta^{(1)} = (W^{(2)})^T (\hat{y} - y) \odot f'(z^{(1)})$$

Where $\odot$ is element-wise multiplication and $f'$ is ReLU derivative:

$$f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{otherwise} \end{cases}$$

**Weight gradient:**
$$\frac{\partial L}{\partial W^{(1)}} = \frac{1}{N} \delta^{(1)} x^T$$

**Bias gradient:**
$$\frac{\partial L}{\partial b^{(1)}} = \frac{1}{N} \sum_{i=1}^{N} \delta^{(1)}_i$$

## 🚀 Gradient Descent Optimization

### Vanilla Update Rule
$$W^{(l)} := W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}$$
$$b^{(l)} := b^{(l)} - \alpha \frac{\partial L}{\partial b^{(l)}}$$

Where $\alpha$ is the **learning rate** (typically 0.01 to 0.1).

### What is the Learning Rate?
- **Too small**: Training is slow, might get stuck
- **Too large**: Training diverges, loss explodes
- **Just right**: Smooth convergence

## 🎲 Weight Initialization

We initialize weights randomly (not zeros!) to break symmetry. A good strategy is **He initialization** for ReLU:

$$W^{(l)}_{ij} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{l-1}}}\right)$$

Where $n_{l-1}$ is the number of neurons in the previous layer.

Biases are typically initialized to zero:
$$b^{(l)} = 0$$

## 🔁 Training Loop

```python
for epoch in range(num_epochs):
    # 1. Forward pass
    z1 = W1 @ x + b1
    a1 = relu(z1)
    z2 = W2 @ a1 + b2
    y_hat = softmax(z2)
    
    # 2. Compute loss
    loss = cross_entropy(y, y_hat)
    
    # 3. Backward pass (backpropagation)
    dz2 = y_hat - y
    dW2 = dz2 @ a1.T / N
    db2 = np.sum(dz2, axis=1, keepdims=True) / N
    
    da1 = W2.T @ dz2
    dz1 = da1 * relu_derivative(z1)
    dW1 = dz1 @ x.T / N
    db1 = np.sum(dz1, axis=1, keepdims=True) / N
    
    # 4. Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
```

## 📊 Key Concepts

### Epoch
One complete pass through the entire training dataset.

### Batch Size
Number of samples used in one forward/backward pass. We use **full-batch gradient descent** (all 120 samples).

### Learning Rate ($\alpha$)
Controls the step size during gradient descent.

## 🎯 Why This Works: Universal Approximation Theorem

A neural network with even **one hidden layer** can approximate any continuous function to arbitrary precision (given enough neurons). This is why neural networks are so powerful!

## 🧮 Total Parameters to Learn

For our architecture:
- $W^{(1)}$: 10 × 4 = 40 parameters
- $b^{(1)}$: 10 parameters
- $W^{(2)}$: 3 × 10 = 30 parameters
- $b^{(2)}$: 3 parameters
- **Total: 83 parameters**

With 120 training samples, we have more data than parameters, reducing overfitting risk.

## 📚 Next Steps

Now that we understand the theory, let's implement it! See [src/iris_classifier.py](../src/iris_classifier.py) for the complete NumPy implementation.

---

**Mathematical Notation:**
- $x$: Input vector
- $y$: True label (one-hot)
- $\hat{y}$: Predicted probability
- $W^{(l)}$: Weight matrix for layer $l$
- $b^{(l)}$: Bias vector for layer $l$
- $z^{(l)}$: Pre-activation
- $a^{(l)}$: Post-activation
- $\alpha$: Learning rate
- $N$: Number of samples
