# Neural Network Research Lab

> A researcher's notebook documenting the journey from mathematical foundations to practical implementation of neural networks.

## Project Overview

This project is a comprehensive exploration of neural networks, starting from first principles and progressing to real-world applications. We follow the scientific method: **observe, hypothesize, implement, evaluate, document**.

## Datasets

### 1. Iris Dataset (Phase 1)
- **Samples**: 150 flowers
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Task**: Multi-class classification
- **Implementation**: NumPy from scratch (no deep learning frameworks)

### 2. Wine Quality Dataset (Phase 2)
- **Samples**: 4,898 white wines
- **Features**: 11 chemical properties
- **Target**: Quality score (0-10), binarised to "good" (≥7) vs "not good"
- **Task**: Binary classification
- **Implementation**: PyTorch

## Mathematical Foundations

### Core Equations

**Forward Propagation:**
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = f(z^{(l)})$$

**Softmax Activation (output layer):**
$$\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Cross-Entropy Loss:**
$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$$

**Backpropagation (output layer):**
$$\frac{\partial L}{\partial z^{(L)}} = \hat{y} - y$$

**Gradient Updates:**
$$W^{(l)} := W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}$$
$$b^{(l)} := b^{(l)} - \alpha \frac{\partial L}{\partial b^{(l)}}$$

## Project Structure

```
neural-network/
├── README.md # This file - research overview
├── RUN_INSTRUCTIONS.md # Step-by-step run guide
├── requirements.txt # Python dependencies
├── iris_run.log # Captured Iris training output
├── wine_run.log # Captured Wine training output
├── docs/ # Documentation
│ ├── 01_iris_dataset.md # Dataset exploration
│ ├── 02_neural_network_theory.md # Math & architecture
│ ├── 03_training_process.md # Training loop details
│ ├── 04_results_analysis.md # Evaluation & insights
│ └── 05_wine_quality.md # Second project
├── models/ # Saved trained models
│ ├── iris_model.npz
│ └── wine_quality_model.pth
├── plots/ # Visualizations
│ ├── training_history.png
│ └── wine_training_history.png
├── data/ # Downloaded datasets
│ └── winequality-white.csv
└── src/ # Source code
 ├── iris_classifier.py # NumPy implementation
 ├── wine_quality.py # PyTorch implementation
 └── utils.py # Shared helpers (paths, seeding, banner)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train Iris classifier
cd src
python iris_classifier.py
```

## Results

| Dataset | Model | Test Accuracy | Final Loss | Epochs |
|---------|-------|---------------|-----------|--------|
| Iris | 2-layer NN (NumPy) | **96.67%** | 0.0414 | 1500 |
| Wine Quality | 64→32 MLP + Dropout (PyTorch) | **83.67%** | 0.3433 | 100 |

Run logs and per-class breakdowns are stored in `iris_run.log` and `wine_run.log`.

## Research Methodology

As researchers, we follow these principles:
1. **Document everything** - Every decision, every equation, every result
2. **Understand first** - Math before code, theory before implementation
3. **Experiment systematically** - Change one variable at a time
4. **Visualize results** - Loss curves, decision boundaries, confusion matrices
5. **Question assumptions** - Why does this work? What could go wrong?

## Learning Path

1. Phase 1: Build neural network from scratch with Iris dataset (96.67% test acc)
2. Phase 2: Apply to Wine Quality dataset with PyTorch (83.67% test acc)
3. ⏳ Phase 3: Advanced topics (regularization, optimization, architectures)

## Contributing

This is a personal research project, but insights and questions are welcome!

---

**Started**: 2026-09-02
**Researcher**: Bikash Talukder
**Institution**: Harvard University (fictional) → Google DeepMind (aspirational)
