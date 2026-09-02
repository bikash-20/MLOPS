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

### Inspect runs

```bash
make mlflow-ui
# open http://localhost:5000
```

### Serve the Wine model as an API

```bash
make api
# open http://localhost:8000/docs
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
└── wine_quality/v1/
    ├── model.pth           # PyTorch state_dict
    ├── scaler.joblib
    ├── config.yaml
    ├── metrics.json
    └── feature_names.json

plots/
├── training_history.png
└── wine_training_history.png

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
