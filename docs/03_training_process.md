# Training Process: From Data to Predictions

## 🔄 The Training Pipeline

Training a neural network is an iterative optimization process. We repeatedly:
1. Make predictions (forward pass)
2. Measure error (loss)
3. Compute gradients (backward pass)
4. Update parameters (gradient descent)

## 🎯 Hyperparameters for Iris Dataset

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| **Learning Rate ($\alpha$)** | 0.01 | Step size for gradient descent |
| **Epochs** | 1000 | Number of complete passes through training data |
| **Hidden Units** | 10 | Neurons in hidden layer |
| **Batch Size** | Full batch (120) | All samples used per update |
| **Weight Init** | He initialization | $\sigma = \sqrt{2/n_{in}}$ |

## 📊 Expected Training Behavior

### Loss Curve
```
Epoch    Loss    Accuracy
   0    1.0986     0.33
 100    0.8234     0.65
 200    0.5432     0.82
 300    0.3421     0.92
 500    0.1234     0.97
 750    0.0543     0.99
1000    0.0234     1.00
```

### What to Look For
- **Decreasing loss**: Model is learning ✓
- **Increasing accuracy**: Predictions improving ✓
- **Smooth curve**: Good learning rate ✓
- **Plateau**: Model converged ✓

## 🔍 Monitoring Training

### Signs of Good Training
✅ Loss decreases steadily
✅ Accuracy approaches 100%
✅ Both train and test performance are similar
✅ Convergence within reasonable epochs (<1000)

### Signs of Problems
❌ **Loss increases**: Learning rate too high
❌ **Loss doesn't decrease**: Learning rate too low or bug
❌ **Train accuracy 100%, test accuracy low**: Overfitting
❌ **Loss oscillates**: Learning rate too high

## 🎲 Stochasticity in Training

### Random Initialization
Each run starts with different random weights, so results vary slightly:
- Run 1: Final accuracy 96.67%
- Run 2: Final accuracy 93.33%
- Run 3: Final accuracy 100.00%

This is **normal**! To get reproducible results, set a random seed:
```python
np.random.seed(42)
```

## 📈 Visualizations Generated

### 1. Training History Plot
- **Loss curve**: Shows optimization progress
- **Accuracy curve**: Shows classification performance

Saved to: `plots/training_history.png`

## 🧪 Experiments to Try

### Experiment 1: Learning Rate
```python
learning_rates = [0.001, 0.01, 0.1, 1.0]
for lr in learning_rates:
    nn = NeuralNetwork(learning_rate=lr)
    nn.train(X_train, Y_train, epochs=500)
```

**Expected observations:**
- 0.001: Slow convergence
- 0.01: Smooth convergence ✓
- 0.1: Fast but might oscillate
- 1.0: Likely diverges

### Experiment 2: Hidden Layer Size
```python
hidden_sizes = [3, 5, 10, 20, 50]
for hs in hidden_sizes:
    nn = NeuralNetwork(hidden_size=hs)
    nn.train(X_train, Y_train, epochs=500)
```

**Expected observations:**
- Too small (3): Underfitting
- Moderate (10): Good balance ✓
- Large (50): Might overfit on this small dataset

### Experiment 3: Number of Epochs
```python
for epochs in [100, 500, 1000, 5000]:
    nn = NeuralNetwork()
    nn.train(X_train, Y_train, epochs=epochs)
```

**Expected observations:**
- 100: Underfitting
- 500-1000: Sweet spot ✓
- 5000: Overfitting risk

## 🎓 Key Takeaways

1. **Neural networks learn iteratively** - each epoch improves predictions
2. **Learning rate is critical** - too high diverges, too low is slow
3. **Monitor both loss and accuracy** - they tell different stories
4. **Random initialization matters** - try multiple seeds
5. **Visualization is essential** - numbers can hide problems

## 🔬 Understanding the Optimization

### Gradient Descent Visualization

```
Loss
 │  ╲
 │   ╲        ← Gradient descent steps
 │    ╲
 │     ╲╲
 │       ╲╲╲
 │          ╲╲╲╲___
 └──────────────────→ Epochs
```

We're trying to find the **lowest point** in this loss landscape.

### The Loss Landscape
- Each point represents a set of weights
- Height = loss value
- We want to reach the **global minimum**
- Gradient descent follows the steepest descent

## 📚 Mathematical Intuition: Why It Works

The chain rule lets us compute how each weight affects the final loss:

$$\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{(2)}} \cdot \frac{\partial z^{(2)}}{\partial a^{(1)}} \cdot \frac{\partial a^{(1)}}{\partial z^{(1)}} \cdot \frac{\partial z^{(1)}}{\partial W^{(1)}}$$

Each term in this chain has a simple form, making the computation tractable!

## 🎉 Success Criteria

Our model is successful if:
- ✅ Test accuracy > 90%
- ✅ Training converges in <1000 epochs
- ✅ Final loss < 0.1
- ✅ Consistent performance across runs

---

**Next:** See [04_results_analysis.md](04_results_analysis.md) for detailed results and interpretation.