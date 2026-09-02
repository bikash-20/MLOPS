# Testing

The project uses [pytest](https://docs.pytest.org/) with three test
categories:

```
tests/
├── unit/         # fast, no I/O
│   ├── test_activations.py     # softmax, ReLU, numerical stability
│   ├── test_loss.py            # cross-entropy, clipping
│   ├── test_gradients.py       # shape + numerical gradient check
│   ├── test_iris_nn.py         # NumPy network forward/backward
│   ├── test_wine_nn.py         # PyTorch model shapes/dropout
│   └── test_metrics.py         # accuracy / F1 helpers
├── integration/  # touches data + training loops
│   ├── test_train_iris.py      # 500 epochs on real Iris, ≥85% acc
│   └── test_train_wine.py      # 5-epoch smoke on real Wine
└── api/          # FastAPI TestClient
    └── test_api.py             # endpoints + 422 validation
```

## Run

```bash
make test            # all tests
make test-unit       # unit only (~1s)
make test-integration
make test-api
```

Or directly:

```bash
pytest                        # all
pytest tests/unit -v          # verbose unit tests
pytest -k softmax             # by name
pytest --cov=src              # with coverage
```

## Markers

```python
@pytest.mark.integration   # full-pipeline test
@pytest.mark.api           # API endpoint test
@pytest.mark.slow          # >5s (add to pytest.ini as needed)
```

## Coverage

Target: 70%+ on `src/`. Some modules (`configs/schemas.py`,
`tracking/mlflow_logger.py`, `training/*.py`) are exercised through
integration tests rather than unit tests.

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## CI

`.github/workflows/ci.yml` runs `pytest` on every push and PR. See the
[deployment docs](10_deployment.md) for more.
