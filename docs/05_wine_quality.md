# Wine Quality Dataset: From Simple to Realistic

## Dataset Overview

The Wine Quality dataset is from the **UCI Machine Learning Repository**, featuring wines from the Portuguese *Vinho Verde* region. It's our bridge from textbook-perfect (Iris) to real-world messy data.

### Dataset Statistics

| Property | Value |
|----------|-------|
| **Total Samples** | 4,898 (1,599 red + 3,899 white) |
| **Features** | 11 (all numeric, chemical properties) |
| **Target** | Quality score (0-10) |
| **Missing Values** | None |
| **Storage Size** | ~250 KB |
| **Source** | UCI ML Repository |

## Features (Chemical Properties)

1. **Fixed Acidity** - Tartaric acid content (g/dm³)
2. **Volatile Acidity** - Acetic acid content (g/dm³)
3. **Citric Acid** - Citric acid content (g/dm³)
4. **Residual Sugar** - Sugar remaining after fermentation (g/dm³)
5. **Chlorides** - Salt content (g/dm³)
6. **Free Sulfur Dioxide** - Free form of SO₂ (mg/dm³)
7. **Total Sulfur Dioxide** - Free + bound forms (mg/dm³)
8. **Density** - Wine density (g/cm³)
9. **pH** - Acidity level (0-14 scale)
10. **Sulphates** - Potassium sulphate content (g/dm³)
11. **Alcohol** - Alcohol percentage (% vol.)

## Target Variable

- **Quality**: Sensory score from 0 (very bad) to 10 (very excellent)
- Median score: 6
- Distribution is **imbalanced** (most wines are 5-6)

## Task Formulation

We have two options:

### Option 1: Binary Classification (Our Choice)
Classify wines as:
- **Good** (quality ≥ 7): 1,440 samples (29.4%)
- **Not Good** (quality < 7): 3,458 samples (70.6%)

### Option 2: Regression
Predict exact quality score (0-10)

## Why This Dataset is a Great Next Step

### 1. **More Realistic Challenges**
- **Imbalanced classes**: Good wines are minority
- **Messy patterns**: Wine quality is subjective
- **More features**: 11 vs 4 in Iris
- **Larger dataset**: Tests scalability

### 2. **Multiple Possible Approaches**
- Classification vs regression
- Binary vs multi-class
- Feature engineering opportunities

### 3. **Real-World Skills**
- Handling class imbalance
- Feature scaling
- Train/validation/test splits
- Preventing overfitting

## Our Architecture: Deeper Network

```
Input (11) → Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU) → Dropout(0.2) → Output(2)
```

### Key Differences from Iris Network:
1. **More hidden units**: 64 and 32 (vs 10)
2. **Multiple hidden layers**: 2 hidden layers (vs 1)
3. **Dropout regularization**: Prevents overfitting
4. **PyTorch framework**: Industry standard

## Key Concepts Introduced

### 1. Dropout Regularization
During training, randomly "drop" neurons with probability $p$:

$$\text{Dropout}(x_i) = \begin{cases} \frac{x_i}{1-p} & \text{with probability } 1-p \\ 0 & \text{with probability } p \end{cases}$$

**Why it works:**
- Prevents co-adaptation of neurons
- Acts as ensemble learning
- Reduces overfitting

### 2. Adam Optimizer
Instead of vanilla gradient descent, we use Adam (Adaptive Moment Estimation):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Advantages over SGD:**
- Adaptive learning rates per parameter
- Faster convergence
- Less sensitive to hyperparameters

### 3. Mini-Batch Training
Instead of full-batch, we process data in batches of 32:
- Faster per-epoch updates
- More stable than pure SGD
- Better generalization

## Expected Performance

| Metric | Expected Value |
|--------|---------------|
| **Test Accuracy** | 85-90% |
| **Precision (Good)** | 0.75-0.85 |
| **Recall (Good)** | 0.70-0.80 |
| **F1 Score** | 0.75-0.82 |

## Key Differences: Iris vs Wine Quality

| Aspect | Iris | Wine Quality |
|--------|------|--------------|
| **Size** | 150 | 4,898 |
| **Classes** | Balanced | Imbalanced |
| **Features** | Clear patterns | Subtle correlations |
| **Difficulty** | Easy (95-100%) | Harder (85-90%) |
| **Framework** | NumPy (learning) | PyTorch (production) |
| **Architecture** | Simple | Regularized |

## Experiments to Try

### Experiment 1: Architecture Depth
```python
# Shallow network
shallow = [11, 32, 2]

# Medium network
medium = [11, 64, 32, 2]

# Deep network
deep = [11, 128, 64, 32, 2]
```

### Experiment 2: Dropout Rate
```python
no_dropout = 0.0
light_dropout = 0.2
heavy_dropout = 0.5
```

### Experiment 3: Learning Rate
```python
learning_rates = [0.0001, 0.001, 0.01, 0.1]
```

### Experiment 4: Class Imbalance Handling
```python
# Option 1: Class weights in loss
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Option 2: Oversampling minority class
# Option 3: SMOTE (Synthetic Minority Over-sampling)
```

## Feature Importance

Based on typical wine chemistry knowledge:

| Feature | Importance | Reason |
|---------|-----------|--------|
| **Alcohol** | | Higher alcohol → better quality |
| **Volatile Acidity** | | Too much → vinegar taste |
| **Sulphates** | | Preservative, affects taste |
| **Citric Acid** | | Adds freshness |
| **Density** | | Correlated with sugar |
| **pH** | | Affects taste balance |

## Research Questions

### Q1: Can we predict wine quality from chemistry?
**A:** Partially. Wine quality is also influenced by subjective factors and vintage. We can predict with ~85% accuracy for binary classification.

### Q2: Why is this harder than Iris?
**A:** Multiple reasons:
- Class imbalance
- Subtle feature interactions
- Subjective quality scores
- Non-linear relationships

### Q3: How does dropout help?
**A:** It prevents the network from relying too heavily on specific neurons, forcing it to learn more robust features.

### Q4: When to use PyTorch vs NumPy?
**A:**
- **NumPy**: Learning, small datasets, full control
- **PyTorch**: Production, large datasets, GPU acceleration, complex models

## Advanced Topics to Explore

After mastering this dataset, consider:
1. **Regression** (predict exact quality score)
2. **Multi-class classification** (low/medium/high)
3. **Feature engineering** (ratios, interactions)
4. **Ensemble methods** (multiple models)
5. **Cross-validation** (k-fold for robust evaluation)

## Key Takeaways

1. **Real-world data is messier** than textbook datasets
2. **Regularization is crucial** for preventing overfitting
3. **Class imbalance requires special handling**
4. **PyTorch simplifies implementation** of complex models
5. **Domain knowledge helps** interpret results

---

**Conclusion:** The Wine Quality dataset teaches us that not all problems are as clean as Iris. We learn to handle class imbalance, prevent overfitting, and use industry-standard tools. These skills are essential for real-world ML!