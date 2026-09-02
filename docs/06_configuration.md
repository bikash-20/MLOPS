# Configuration (Hydra)

This project uses [Hydra](https://hydra.cc/) by Facebook for configuration
management. Configs live in `configs/` and are loaded by the training
entrypoints at `src/training/train_iris.py` and `src/training/train_wine.py`.

## Files

```
configs/
├── iris.yaml       # default Iris training config (single file, all keys)
└── wine.yaml       # default Wine training config
```

Both files are self-contained — they declare `data`, `model`, and `train`
groups directly. No nested sub-configs.

## Schema

The dataclasses that mirror the YAML structure live in
`src/configs/schemas.py`. The training entrypoints register them with
Hydra's `ConfigStore` for type checking and CLI completion.

```python
@dataclass
class IrisTrainConfig:
    learning_rate: float = 0.1
    epochs: int = 1500
    batch_size: int = 1
    log_every: int = 100
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: str = "iris-classifier"
```

## Override from the CLI

Hydra's killer feature — change any parameter without editing files:

```bash
# Train Iris for 500 epochs instead of 1500
python -m src.training.train_iris train.epochs=500

# Train Wine with a deeper architecture
python -m src.training.train_wine model.hidden_sizes=[128,64,32]

# Combine multiple overrides
python -m src.training.train_wine \
  model.dropout=0.3 \
  train.learning_rate=0.0005 \
  train.epochs=50

# Run multiple experiments (sweep) over learning rates
python -m src.training.train_wine --multirun train.learning_rate=0.01,0.001,0.0001
```

## Programmatically

You can also instantiate configs in tests or scripts:

```python
from src.configs.schemas import IrisConfig, WineConfig

cfg = IrisConfig()
print(cfg.train.epochs)  # 1500

cfg.train.epochs = 500
print(cfg.train.epochs)  # 500
```

## Frozen snapshot in the registry

Every training run saves the **exact** resolved config to the model
registry:

```
models/iris/v1/config.yaml
models/wine_quality/v1/config.yaml
```

This guarantees reproducibility: loading a model from disk reads its
config to know architecture and dropout, so inference uses the same
shapes as training.
