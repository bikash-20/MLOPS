"""Typed config dataclasses wired into Hydra's config store.

Training entrypoints decorate ``main()`` with ``@hydra.main(...)`` and
receive an instance of ``DictConfig`` populated from the YAML files under
``configs/``. Override values from the CLI:

    python -m src.training.train_wine model.hidden_sizes=[128,64] train.epochs=50
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Generic configs -------------------------------------------------------


@dataclass
class DataConfig:
    """Dataset parameters shared across projects."""

    test_size: float = 0.2
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Generic model hyperparameters."""

    name: str = "default"


@dataclass
class TrainConfig:
    """Training-loop parameters shared across projects."""

    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    log_every: int = 100
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "neural-network"


# --- Iris-specific configs (no inheritance — Hydra struct-friendly) -------


@dataclass
class IrisDataConfig:
    test_size: float = 0.2
    random_seed: int = 42


@dataclass
class IrisModelConfig:
    name: str = "iris_2layer"
    input_size: int = 4
    hidden_size: int = 10
    output_size: int = 3
    learning_rate: float = 0.1


@dataclass
class IrisTrainConfig:
    learning_rate: float = 0.1
    epochs: int = 1500
    batch_size: int = 1  # Iris uses full-batch gradient descent
    log_every: int = 100
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "iris-classifier"


# --- Wine-specific configs (no inheritance — Hydra struct-friendly) --------


@dataclass
class WineDataConfig:
    test_size: float = 0.2
    random_seed: int = 42
    binary: bool = True
    good_threshold: int = 7


@dataclass
class WineModelConfig:
    name: str = "wine_mlp"
    input_size: int = 11
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 32])
    output_size: int = 2
    dropout: float = 0.2


@dataclass
class WineTrainConfig:
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    log_every: int = 10
    # Minimum accuracy improvement over previous version to be promoted
    # to a new ``vN+1`` in the on-disk registry. See mnist config.
    min_acc_delta: float = 0.0
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "wine-quality"


# --- MNIST-specific configs (no inheritance — Hydra struct-friendly) -------


@dataclass
class MnistDataConfig:
    test_size: float = 0.0  # MNIST uses fixed 60k/10k split
    random_seed: int = 42


@dataclass
class MnistModelConfig:
    name: str = "mnist_cnn"
    in_channels: int = 1
    conv_channels: tuple[int, ...] = (32, 64)
    fc_hidden: int = 128
    num_classes: int = 10
    dropout: float = 0.25


@dataclass
class MnistTrainConfig:
    learning_rate: float = 0.001
    epochs: int = 5
    batch_size: int = 128
    log_every: int = 1
    # Minimum accuracy improvement over previous version to be promoted
    # to a new ``vN+1`` in the on-disk registry. Set to 0 to accept any
    # improvement (or tie). Set higher to require meaningful gains.
    min_acc_delta: float = 0.0
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "mnist-cnn"


# --- CIFAR-10-specific configs (no inheritance — Hydra struct-friendly) ----


@dataclass
class CifarDataConfig:
    """CIFAR-10 uses the canonical 50k/10k split; we further carve 5k for val."""

    val_size: float = 0.1
    random_seed: int = 42


@dataclass
class CifarModelConfig:
    """CIFAR-style ResNet (3x3 stem, no maxpool) trained from scratch."""

    name: str = "cifar_resnet18"
    in_channels: int = 3
    num_classes: int = 10
    # Backbone width (CIFAR ResNet paper keeps this small: 16 -> 32 -> 64).
    base_channels: int = 64
    num_blocks_per_stage: int = 2  # 2 blocks per stage -> ~11M params
    dropout: float = 0.2


@dataclass
class CifarTrainConfig:
    """Training loop with OneCycle LR + early stopping + best checkpoint."""

    learning_rate: float = 0.1  # OneCycle max_lr; SGD needs higher than Adam
    epochs: int = 20
    batch_size: int = 128
    weight_decay: float = 5e-4
    momentum: float = 0.9
    log_every: int = 1
    # OneCycle scheduler
    scheduler: str = "one_cycle"
    pct_start: float = 0.1
    # Early stopping on val accuracy
    early_stop_patience: int = 5
    # Minimum accuracy improvement over previous version to be promoted
    # to a new ``vN+1`` in the on-disk registry.
    min_acc_delta: float = 0.0
    # Number of dataloader workers (0 is safe in tests/CI; >0 on real hosts).
    num_workers: int = 0
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "cifar10-resnet"


# --- Top-level Configs (kept for direct programmatic use) ------------------


@dataclass
class Config:
    """Root config composed from the project-specific groups."""

    project: str = "iris"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class IrisConfig:
    """Composed config used by the Iris training entrypoint."""

    project: str = "iris"
    data: IrisDataConfig = field(default_factory=IrisDataConfig)
    model: IrisModelConfig = field(default_factory=IrisModelConfig)
    train: IrisTrainConfig = field(default_factory=IrisTrainConfig)


@dataclass
class WineConfig:
    """Composed config used by the Wine training entrypoint."""

    project: str = "wine"
    data: WineDataConfig = field(default_factory=WineDataConfig)
    model: WineModelConfig = field(default_factory=WineModelConfig)
    train: WineTrainConfig = field(default_factory=WineTrainConfig)