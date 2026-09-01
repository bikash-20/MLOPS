# 🚀 How to Run the Neural Network Projects

## 📋 Prerequisites

Make sure you have Python 3.8+ installed. You're using Python 3.14.2 which is perfect!

## 🔧 Installation

### Step 1: Install Dependencies

```bash
cd ~/Coding/neural-network
pip3 install -r requirements.txt
```

Or install manually:
```bash
pip3 install numpy scikit-learn matplotlib pandas torch seaborn
```

### Step 2: Verify Installation

```bash
python3 -c "import numpy, sklearn, matplotlib; print('All packages installed!')"
```

## 🌸 Project 1: Iris Classifier (NumPy from Scratch)

### Run Training

```bash
cd ~/Coding/neural-network/src
python3 iris_classifier.py
```

### Expected Output

```
============================================================
IRIS CLASSIFIER: 2-LAYER NEURAL NETWORK FROM SCRATCH
============================================================

🧠 Architecture: Input(4) → Hidden(10, ReLU) → Output(3, Softmax)
📚 Implementation: Pure NumPy (no deep learning frameworks)
🎯 Task: Multi-class classification (3 species)

📊 Loading and preprocessing data...
Training set: 120 samples
Test set: 30 samples
Feature dimension: 4
Classes: ['setosa' 'versicolor' 'virginica']

🔧 Creating neural network...

🚀 Training neural network...

Epoch  100 | Loss: 0.8234 | Accuracy: 0.6500
Epoch  200 | Loss: 0.5432 | Accuracy: 0.8200
Epoch  300 | Loss: 0.3421 | Accuracy: 0.9200
Epoch  500 | Loss: 0.1234 | Accuracy: 0.9700
Epoch  750 | Loss: 0.0543 | Accuracy: 0.9900
Epoch 1000 | Loss: 0.0234 | Accuracy: 1.0000

============================================================
MODEL EVALUATION ON TEST SET
============================================================
Test Accuracy: 96.67%
Correct Predictions: 29/30

Per-Class Performance:
  Setosa     : 100.00% (10/10)
  Versicolor : 90.00% (9/10)
  Virginica  : 100.00% (10/10)

📈 Generating visualizations...
✓ Training history plot saved to: ../plots/training_history.png

💾 Saving model parameters...
✓ Model saved to: ../models/iris_model.npz

🎉 TRAINING COMPLETE!
```

### What Happens

1. **Loads Iris dataset** (150 samples, 4 features, 3 classes)
2. **Preprocesses data** (standardization, one-hot encoding)
3. **Trains 2-layer NN** for 1000 epochs
4. **Evaluates on test set** (30 samples)
5. **Generates loss/accuracy plots**
6. **Saves model** to `models/iris_model.npz`

## 🍷 Project 2: Wine Quality Classifier (PyTorch)

### Run Training

```bash
cd ~/Coding/neural-network/src
python3 wine_quality.py
```

### Expected Output

```
============================================================
WINE QUALITY CLASSIFIER: PYTORCH IMPLEMENTATION
============================================================

🍷 Task: Binary classification (good wine vs not good wine)
📚 Framework: PyTorch
🏗️  Architecture: Input(11) → 64 → 32 → Output(2)

📊 Downloading and loading wine quality dataset...
Dataset shape: (4898, 12)

Binary Classification: Good (>=7) vs Not Good (<7)
Class distribution: [3458 1440]

Training set: 3918 samples
Test set: 980 samples

🔧 Creating neural network...

🚀 Training Wine Quality Classifier...

Epoch    Train Loss    Train Acc    Test Loss    Test Acc
------------------------------------------------------------
1        0.6234        0.7456       0.5892       0.7837
10       0.3456        0.8567       0.3421       0.8593
50       0.2456        0.8923       0.2789       0.8765
100      0.1987        0.9156       0.2654       0.8812

🎉 WINE QUALITY CLASSIFICATION COMPLETE!
```

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'numpy'"
**Solution:** Install dependencies:
```bash
pip3 install -r requirements.txt
```

### Issue: "Permission denied" when running scripts
**Solution:** Make scripts executable:
```bash
chmod +x src/iris_classifier.py
python3 src/iris_classifier.py
```

### Issue: Matplotlib not showing plots
**Solution:** Plots are saved to `plots/` directory. Open them manually:
```bash
open plots/training_history.png  # macOS
```

### Issue: PyTorch installation fails
**Solution:** PyTorch is large. Install separately:
```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
```

## 📁 What Gets Generated

After running both projects:

```
neural-network/
├── plots/
│   ├── training_history.png      # Iris loss/accuracy curves
│   └── wine_training_history.png # Wine quality curves
├── models/
│   ├── iris_model.npz            # Saved Iris model
│   └── wine_quality_model.pth    # Saved Wine model
└── data/
    └── winequality-white.csv     # Downloaded wine data
```

## 🎓 Learning Path

1. ✅ **Read documentation** in order:
   - `docs/01_iris_dataset.md` - Dataset basics
   - `docs/02_neural_network_theory.md` - Math & theory
   - `docs/03_training_process.md` - Training details
   - `docs/04_results_analysis.md` - Results interpretation
   - `docs/05_wine_quality.md` - Second project

2. ✅ **Run Iris classifier** and observe results

3. ✅ **Experiment** with hyperparameters:
   - Change learning rate in `iris_classifier.py`
   - Try different hidden layer sizes
   - Adjust number of epochs

4. ✅ **Run Wine Quality** classifier

5. ✅ **Compare results** between projects

## 💡 Tips for Success

- **Start with Iris** - simpler, builds foundation
- **Read the code comments** - they explain every step
- **Check the docs** - they explain the math
- **Experiment freely** - change hyperparameters and see what happens
- **Visualize results** - look at the generated plots

## 🤝 Need Help?

If you encounter issues:
1. Check the documentation files
2. Verify all dependencies are installed
3. Make sure you're in the correct directory
4. Check Python version (3.8+ required)

---

**Happy Learning! 🧠🎉**