# Neural Network Research Lab

> A production-ready ML engineering project documenting the journey from
> mathematical foundations to deployable neural-network systems.

## What's in here

Two neural-network implementations built from first principles, plus the
engineering scaffolding expected of any modern ML system:

| Project | Stack | Task | Test accuracy |
|---|---|---|---|
| **Iris** | NumPy (no frameworks) | Multi-class classification (3 species) | ~97% |
| **Wine Quality** | PyTorch MLP | Binary classification (good vs not good) | ~83% |

Engineering layers: Hydra configs, MLflow tracking, pytest suite, FastAPI
service, Docker + docker-compose, GitHub Actions CI, structured logging,
versioned model registry.

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run the test suite (31 tests, ~2s)
make test

# 3. Train the Iris classifier
make train-iris

# 4. Train the Wine Quality classifier (100 epochs ~30s on CPU)
make train-wine

# 5. Inspect runs in the MLflow UI
make mlflow-ui
# open http://localhost:5000

# 6. Serve the Wine model as a REST API
make api
# open http://localhost:8000/docs
```

## Project Structure

```
neural-network/
├── README.md                 # this file
├── RUN_INSTRUCTIONS.md       # detailed run guide
├── Makefile                  # one-liners for common tasks
├── Dockerfile                # multi-stage container image
├── docker-compose.yml        # api + mlflow services
├── requirements.txt          # pinned dependencies
├── pytest.ini                # pytest configuration
│
├── configs/                  # Hydra YAML configs
│   ├── iris.yaml             # default Iris training config
│   └── wine.yaml             # default Wine training config
│
├── src/
│   ├── api/                  # FastAPI service
│   ├── configs/              # typed dataclasses
│   ├── data/                 # dataset loaders
│   ├── evaluation/           # metrics
│   ├── models/               # NeuralNetwork (NumPy) + WineNet (PyTorch)
│   ├── tracking/             # MLflow wrapper
│   ├── training/             # Hydra-decorated entrypoints
│   ├── utils/                # paths, logging, reproducibility, display
│   └── visualization/        # plot helpers
│
├── tests/
│   ├── unit/                 # activations, loss, gradients, models
│   ├── integration/          # training pipelines (smoke tests)
│   └── api/                  # FastAPI TestClient
│
├── docs/
│   ├── 01_iris_dataset.md
│   ├── 02_neural_network_theory.md
│   ├── 03_training_process.md
│   ├── 04_results_analysis.md
│   ├── 05_wine_quality.md
│   ├── 06_configuration.md   # ← Hydra usage
│   ├── 07_experiment_tracking.md   # ← MLflow usage
│   ├── 08_testing.md         # ← pytest guide
│   ├── 09_api.md             # ← FastAPI endpoints
│   └── 10_deployment.md      # ← Docker + cloud
│
├── data/
│   ├── raw/                  # downloaded UCI wine CSV
│   ├── processed/            # scalers, tensor snapshots
│   └── external/             # third-party data
│
├── models/                   # versioned model registry
│   ├── iris/v1/{model.npz,scaler.joblib,config.yaml,metrics.json}
│   └── wine_quality/v1/{model.pth,scaler.joblib,config.yaml,metrics.json}
│
├── plots/                    # training-curve PNGs
├── mlruns/                   # MLflow SQLite + artifacts (gitignored)
└── reports/                  # generated analysis (gitignored)
```

## Datasets

### Iris (Phase 1)
- 150 flowers, 4 features, 3 species
- Loaded from scikit-learn, no download required
- See [docs/01_iris_dataset.md](docs/01_iris_dataset.md)

### Wine Quality (Phase 2)
- 4,898 white wines, 11 chemical features, binarised as good (≥7) vs not good
- Auto-downloaded from UCI on first run to `data/raw/winequality-white.csv`
- See [docs/05_wine_quality.md](docs/05_wine_quality.md)

## Mathematical Foundations

**Forward:**
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad a^{(l)} = f(z^{(l)})$$

**Softmax:**
$$\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Cross-Entropy Loss:**
$$L = -\frac{1}{N} \sum_i \sum_k y_{ik} \log(\hat{y}_{ik})$$

**Backprop (output):**
$$\frac{\partial L}{\partial z^{(L)}} = \hat{y} - y$$

**Gradient updates:**
$$W^{(l)} := W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}$$

## Results

| Dataset | Model | Test Accuracy | Final Loss | Epochs |
|---|---|---|---|---|
| Iris | 2-layer NumPy NN (4→10→3) | ~97% | ~0.04 | 1500 |
| Wine Quality | 11→64→32→2 MLP + Dropout | ~83% | ~0.34 | 100 |

## Common Tasks

```bash
# Override hyperparameters from the CLI
python -m src.training.train_wine model.hidden_sizes=[128,64] train.epochs=50

# Run only unit tests
make test-unit

# Lint
make lint

# Format
make format

# Build and run the container
make docker-build
make docker-run

# Start api + mlflow together
make docker-up
```

## Research Methodology

1. **Document everything** — every decision, equation, result
2. **Understand first** — math before code, theory before implementation
3. **Experiment systematically** — change one variable at a time
4. **Visualize** — loss curves, confusion matrices
5. **Question assumptions** — why does this work? What could go wrong?

## Contributing

Personal research project — insights and questions welcome!

---

**Started**: 2026-09-02
**Researcher**: Bikash Talukder
**Status**: Production-ready scaffold complete
