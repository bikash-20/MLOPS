# Datasets

This directory is **gitignored** (see `.gitignore`). Every dataset the
project uses is auto-downloaded on first run; nothing is committed.

| Subdir | Contents | Source | Auto-downloaded by |
|---|---|---|---|
| `raw/winequality-white.csv` | UCI white-wine CSV (4,898 rows) | [UCI ML repo](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv) | `src/data/wine_dataset.py` |
| `raw/MNIST/` | MNIST handwritten digits (60k train + 10k test) | torchvision | `src/data/mnist_dataset.py` |
| `raw/cifar-10-batches-py/` | CIFAR-10 (50k train + 10k test, 32x32 RGB) | torchvision | `src/data/cifar_dataset.py` |
| `processed/` | Output of preprocessing (scalers, tensor snapshots) | generated | training entrypoints |
| `external/` | Place for third-party / manually-added data | manual | — |

## Re-creating the data

To re-fetch everything from scratch:

```bash
rm -rf data/raw/MNIST data/raw/cifar-10-batches-py data/raw/winequality-white.csv
make train-iris      # no download
make train-wine      # downloads winequality-white.csv
make train-mnist     # downloads MNIST
make train-cifar     # downloads CIFAR-10
```

The first time you run a trainer for a given dataset it will hit the
network; subsequent runs reuse the cached files.
