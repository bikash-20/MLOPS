# Experiment Tracking (MLflow)

This project uses [MLflow](https://mlflow.org/) for experiment tracking.
Every training run logs hyperparameters, per-epoch metrics, the trained
model artifact, and the frozen config.

## Storage

MLflow 3.x removed file-based tracking. The `MlflowLogger` defaults to a
local SQLite database:

```
mlruns/mlflow.db    # tracking database
```

For production, point MLflow at a real server:

```bash
# Set the tracking URI via env or config
export MLFLOW_TRACKING_URI=http://mlflow.example.com:5000
```

```python
MlflowLogger(tracking_uri="http://mlflow.example.com:5000")
```

## What gets logged

For each training run:

- **Params** — every key in the Hydra config (learning rate, hidden sizes,
  dropout, etc.).
- **Metrics** — `train_loss`, `train_acc`, `test_loss`, `test_acc` per
  epoch; final precision/recall/F1.
- **Artifacts** — `model.pth` / `model.npz`, `scaler.joblib`,
  `config.yaml`, `metrics.json`, and the training-curve PNG.
- **Tags** — Python version, platform, hostname, git commit hash.

## Inspecting runs

```bash
# Start the MLflow UI
make mlflow-ui
# open http://localhost:5000
```

From the UI you can:
- Compare runs side-by-side
- Filter by hyperparameters
- See metric curves over time
- Download registered artifacts

## Programmatic access

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
runs = mlflow.search_runs(experiment_names=["wine-quality"])
best = runs.sort_values("metrics.test_acc", ascending=False).iloc[0]
print(best[["run_id", "metrics.test_acc", "params.model.hidden_sizes"]])
```

## Docker Compose

`make docker-up` brings up both the API and an MLflow server:

```yaml
services:
  api:
    image: neural-network:latest
    ports: ["8000:8000"]
    depends_on: [mlflow]
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.16.0
    ports: ["5000:5000"]
    volumes: ["./mlruns:/mlflow/mlruns"]
```

To point the API at the composed MLflow server, set
`MLFLOW_TRACKING_URI=http://mlflow:5000` in the `api` service environment.

## Without MLflow

If MLflow is not installed, `MlflowLogger` silently no-ops. Training still
works — you just lose tracking. Install with `pip install mlflow`.
