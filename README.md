# Neural Network Research Lab

> A production-ready ML engineering project documenting the journey from
> mathematical foundations to deployable neural-network systems.

## What's in here

Four neural-network implementations built from first principles, plus
the engineering scaffolding expected of any modern ML system:

| Project | Stack | Task | Test accuracy |
|---|---|---|---|
| **Iris** | NumPy (no frameworks) | Multi-class classification (3 species) | ~97% |
| **Wine Quality** | PyTorch MLP | Binary classification (good vs not good) | ~83% |
| **MNIST** | PyTorch CNN (3-layer) | Multi-class classification (10 digits) | ~99% |
| **CIFAR-10** | PyTorch ResNet-18 (CIFAR-style, from scratch) | Multi-class classification (10 classes) | ~93% |

Engineering layers: Hydra configs, MLflow tracking, pytest suite, FastAPI
service, Docker + docker-compose, Modal serverless deploy, GitHub Actions
CI (with auto-deploy to Modal on `main`), structured logging, versioned
model registry with promotion gating.

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run the test suite (89 tests, ~3s)
make test

# 3. Train the Iris classifier
make train-iris

# 4. Train the Wine Quality classifier (100 epochs ~30s on CPU)
make train-wine

# 5. Train the MNIST CNN (downloads MNIST on first run, ~90s on CPU)
make train-mnist

# 6. Train the CIFAR-10 ResNet (downloads CIFAR on first run)
make train-cifar

# 7. Inspect runs in the MLflow UI
make mlflow-ui
# open http://localhost:5000

# 8. Serve all models as a REST API (wine + MNIST + CIFAR-10)
make api
# open http://localhost:8000/docs
```

## Live Demo

The same API is also deployed to Modal (serverless, free tier) — one
URL serves all three models. Every merge to `main` auto-deploys via
GitHub Actions:

```bash
make deploy      # one-time: builds image and pushes to Modal
                 # prints a URL like https://<you>--neural-network-api-fastapi-app.modal.run
```

See [docs/10_deployment.md](docs/10_deployment.md#serverless-modal) for
the full setup, and [docs/13_model_registry.md](docs/13_model_registry.md)
for the registry versioning + promotion gate.

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
│   ├── wine.yaml             # default Wine training config
│   ├── mnist.yaml            # default MNIST training config
│   └── cifar.yaml            # default CIFAR-10 training config
│
├── src/
│   ├── api/                  # FastAPI service (wine + MNIST + CIFAR-10)
│   ├── configs/              # typed dataclasses
│   ├── data/                 # dataset loaders
│   ├── evaluation/           # metrics
│   ├── models/               # NeuralNetwork (NumPy) + WineNet + SimpleCNN + CifarResNet
│   ├── tracking/             # MLflow wrapper
│   ├── training/             # Hydra-decorated entrypoints
│   ├── utils/                # paths, logging, reproducibility, display, registry versioning
│   └── visualization/        # plot helpers
│
├── deploy/
│   └── modal_app.py          # one-command Modal deploy
│
├── tests/
│   ├── unit/                 # activations, loss, gradients, models, versioning
│   ├── integration/          # training pipelines (smoke tests)
│   └── api/                  # FastAPI TestClient (incl. /models listing)
│
├── docs/
│   ├── 01_iris_dataset.md
│   ├── 02_neural_network_theory.md
│   ├── 03_training_process.md
│   ├── 04_results_analysis.md
│   ├── 05_wine_quality.md
│   ├── 06_configuration.md            # ← Hydra usage
│   ├── 07_experiment_tracking.md      # ← MLflow usage
│   ├── 08_testing.md                  # ← pytest guide
│   ├── 09_api.md                      # ← FastAPI endpoints (all 3 models)
│   ├── 10_deployment.md               # ← Docker + Modal + cloud
│   ├── 11_mnist.md                    # ← Third project: CNN on MNIST
│   ├── 12_cifar10.md                  # ← Fourth project: ResNet on CIFAR-10
│   └── 13_model_registry.md           # ← Versioning, promotion gate, CI deploy
│
├── data/
│   ├── raw/                  # downloaded UCI wine CSV, MNIST, CIFAR-10
│   ├── processed/            # scalers, tensor snapshots
│   └── external/             # third-party data
│
├── models/                   # versioned model registry (v1, v2, ...)
│   ├── iris/v1/{model.npz,scaler.joblib,config.yaml,metrics.json}
│   ├── wine_quality/vN/{model.pth,scaler.joblib,config.yaml,metrics.json}
│   ├── mnist/vN/{model.pth,model_arch.json,config.yaml,metrics.json,class_names.json}
│   └── cifar/vN/{best.pt,last.pt,model_arch.json,config.yaml,metrics.json,class_names.json}
│
├── plots/                    # training-curve + confusion-matrix PNGs
├── mlruns/                   # MLflow SQLite + artifacts (gitignored)
└── reports/                  # generated analysis (gitignored)
```

## Datasets

### Iris (Phase 1)
- 150 flowers, 4 features, 3 species
- Loaded from scikit-learn, no download required
- See [docs/01_iris_dataset.md](docs/01_iris_dataset.md)

### Wine Quality (Phase 2)
- 4,898 white wines, 11 chemical features, binarised as good (>=7) vs not good
- Auto-downloaded from UCI on first run to `data/raw/winequality-white.csv`
- See [docs/05_wine_quality.md](docs/05_wine_quality.md)

### MNIST (Phase 3)
- 60,000 training + 10,000 test images of handwritten digits (0-9)
- 28x28 grayscale; auto-downloaded by torchvision into `data/raw/MNIST/`
- See [docs/11_mnist.md](docs/11_mnist.md)

### CIFAR-10 (Phase 4)
- 60,000 RGB images (50k train + 10k test), 32x32, 10 classes
- Auto-downloaded by torchvision into `data/raw/cifar-10-batches-py/`
- Canonical mean/std normalisation, stratified 45k/5k/10k split
- See [docs/12_cifar10.md](docs/12_cifar10.md)

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
| Iris | 2-layer NumPy NN (4->10->3) | ~97% | ~0.04 | 1500 |
| Wine Quality | 11->64->32->2 MLP + Dropout | ~83% | ~0.34 | 100 |
| MNIST | 3-layer CNN (Conv*2 -> FC -> 10) | ~99% | ~0.03 | 5 |
| CIFAR-10 | CIFAR-style ResNet-18 (3x3 stem, 4 stages) | ~93% | ~0.25 | 20 |

## API Endpoints

The FastAPI service (`src/api/main.py`) exposes every trained model
under a consistent contract:

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service info + endpoint listing |
| GET | `/health` | Liveness probe (reports overall health) |
| GET | `/models` | Registry listing (every project + every version + metrics) |
| GET | `/model/info` | Wine model metadata (version, metrics, features) |
| GET | `/model/info/mnist` | MNIST model metadata |
| GET | `/model/info/cifar` | CIFAR-10 model metadata |
| POST | `/predict` | Wine quality (JSON body) |
| POST | `/predict/mnist` | MNIST digit (multipart PNG, 28x28 grayscale) |
| POST | `/predict/cifar` | CIFAR-10 class (multipart PNG/JPEG, 32x32 RGB) + top-5 |

Interactive docs at `http://localhost:8000/docs`.

## Registry Versioning & Promotion Gate

Every training run writes to `models/<project>/vN/` where N is
auto-incremented (1, 2, 3, ...). A run is "promoted" (becomes the new
production candidate) only when its test accuracy beats the previous
version's accuracy by at least `train.min_acc_delta`:

```bash
# Default: accept any improvement or tie
make train-cifar

# Require at least +1% accuracy gain over the previous best
python -m src.training.train_cifar train.min_acc_delta=0.01

# List every on-disk version
make models

# Inspect live registry at runtime
curl http://localhost:8000/models
```

The API picks the highest version by default, but operators can pin a
specific version per deployment via the `MODEL_VERSION` env var:

```bash
MODEL_VERSION=v2 make api          # pin wine + MNIST + CIFAR to v2
MODEL_VERSION=v2 docker compose up # same in docker-compose
```

Full details in [docs/13_model_registry.md](docs/13_model_registry.md).

## CI / CD

GitHub Actions (`.github/workflows/ci.yml`) has three jobs:

1. **lint-and-test** — ruff + full pytest suite (89 tests) + Codecov.
2. **docker-build** — multi-stage container build + `/health` smoke test.
3. **deploy-modal** — gated on `push` to `main`. Builds the Modal
   image, deploys, and smoke-tests the live URL. Requires two GitHub
   repo secrets: `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (one-time
   setup via `modal token new`).

Manual deploy: `make deploy` (interactive).

## Common Tasks

```bash
# Override hyperparameters from the CLI
python -m src.training.train_wine model.hidden_sizes=[128,64] train.epochs=50
python -m src.training.train_mnist train.epochs=10 model.dropout=0.5
python -m src.training.train_cifar train.epochs=5 model.base_channels=16

# Run only unit tests / integration tests / API tests
make test-unit
make test-integration
make test-api

# Lint
make lint

# Format
make format

# Build and run the container
make docker-build
make docker-run

# Start api + mlflow together
make docker-up

# List every on-disk model version
make models

# Deploy the unified API to Modal (serverless, public URL)
make deploy

# Pin a specific registry version at runtime
MODEL_VERSION=v2 make api
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
