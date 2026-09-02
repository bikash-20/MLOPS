# How to Run the Neural Network Projects

## Prerequisites

Python 3.8+. You're on Python 3.14 — works thanks to a tiny compat shim.

## Install

```bash
pip install -r requirements.txt
```

(Or `make install`.)

## Run

All commands use the project root as CWD.

### Iris classifier (NumPy from scratch)

```bash
# via Make
make train-iris

# or directly
python -m src.training.train_iris

# override hyperparameters
python -m src.training.train_iris train.epochs=500 model.hidden_size=20
```

### Wine Quality classifier (PyTorch)

```bash
make train-wine
# or
python -m src.training.train_wine

# override
python -m src.training.train_wine model.hidden_sizes=[128,64] train.epochs=50
```

### MNIST CNN (PyTorch)

```bash
make train-mnist          # ~90s on CPU, downloads MNIST on first run
# or
python -m src.training.train_mnist

# override
python -m src.training.train_mnist train.epochs=10 model.dropout=0.5
```

### Inspect runs

```bash
make mlflow-ui
# open http://localhost:5000
```

### Serve the Wine + MNIST models as an API

```bash
make api
# open http://localhost:8000/docs

# test the wine endpoint
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"fixed_acidity":7.0,"volatile_acidity":0.27,"citric_acid":0.36,
       "residual_sugar":20.7,"chlorides":0.045,"free_sulfur_dioxide":45.0,
       "total_sulfur_dioxide":170.0,"density":1.001,"ph":3.0,
       "sulphates":0.45,"alcohol":8.8}'

# test the MNIST endpoint (multipart upload of a 28x28 PNG)
curl -X POST http://localhost:8000/predict/mnist -F "file=@tests/fixtures/sample_digit.png"
```

### Run the test suite

```bash
make test            # all tests
make test-unit       # unit only
make test-api        # API only
```

### Docker

```bash
make docker-build    # build image
make docker-run      # run API on port 8000
make docker-up       # api + mlflow together
make docker-down     # stop and remove
```

### Deploy to Modal (serverless, public URL)

```bash
pip install modal       # one-time
modal setup             # one-time: link your Modal account
make deploy             # builds image, deploys to Modal, prints public URL

# Or live-reload dev mode (URL is the same; code reloads):
make deploy-serve
```

## Old Entry Points (deprecated, still work)

The original scripts are now thin shims that print a deprecation warning
and forward to the new entrypoints. Prefer the new commands above.

```bash
python src/iris_classifier.py   # → python -m src.training.train_iris
python src/wine_quality.py      # → python -m src.training.train_wine
```

## Outputs

After training, you'll find:

```
models/
├── iris/v1/
│   ├── model.npz           # NumPy weights + history
│   ├── scaler.joblib       # StandardScaler
│   ├── config.yaml         # frozen Hydra config
│   └── metrics.json        # accuracy, precision, recall, f1
├── wine_quality/v1/
│   ├── model.pth           # PyTorch state_dict
│   ├── scaler.joblib
│   ├── config.yaml
│   ├── metrics.json
│   └── feature_names.json
└── mnist/v1/
    ├── model.pth           # PyTorch CNN state_dict
    ├── model_arch.json     # architecture (so API can rebuild)
    ├── config.yaml
    ├── metrics.json
    └── class_names.json    # ["0", "1", ..., "9"]

plots/
├── training_history.png
├── wine_training_history.png
└── mnist_training_history.png

mlruns/
└── mlflow.db               # MLflow tracking database
```

## Troubleshooting

**Hydra complains about struct mode**: This repo includes
`src/utils/hydra_compat.py` which monkey-patches argparse for
Hydra 1.3.x + Python 3.14 compatibility. If you see the error, make sure
your entrypoint imports `src.utils.hydra_compat` BEFORE `hydra.main` is
called. Both training scripts already do this.

**`mlflow.exceptions.MlflowException: filesystem tracking backend`**: MLflow
3.x removed file-based tracking. The `MlflowLogger` defaults to a local
SQLite database at `mlruns/mlflow.db`. To use a custom URI, pass
`tracking_uri` when constructing it.

**API returns 503**: Model not trained yet. Run `make train-wine` first.

**Tests fail with import errors**: Ensure `PYTHONPATH=.` is set when
running pytest directly, or use `make test`.

## Need Help?

- Read the docs in `docs/` — start with `06_configuration.md` and `09_api.md`.
- Run `make help` for a list of targets.
- Open an issue.
