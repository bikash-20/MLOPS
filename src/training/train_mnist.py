"""MNIST training entrypoint — Hydra-configured PyTorch CNN.

Usage:
    python -m src.training.train_mnist
    python -m src.training.train_mnist train.epochs=10 model.dropout=0.5
"""

from __future__ import annotations

import json
import os

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from torch import nn as nn_torch
from torch.utils.data import DataLoader, TensorDataset

# Hydra <1.4 + Python 3.14 compat shim — must come BEFORE ``import hydra``.
import src.utils.hydra_compat  # noqa: F401
from src.configs.schemas import MnistDataConfig, MnistModelConfig, MnistTrainConfig
from src.data.mnist_dataset import load_mnist_dataset
from src.evaluation.evaluate import evaluate_classifier
from src.models.mnist_cnn import SimpleCNN
from src.tracking.mlflow_logger import MlflowLogger
from src.utils import (
    banner,
    get_logger,
    models_path,
    plots_path,
    resolve_promotion,
    set_seed,
)
from src.visualization.plot_training import plot_training_history

logger = get_logger(__name__)


cs = ConfigStore.instance()
cs.store(name="data", node=MnistDataConfig, package="data")
cs.store(name="model", node=MnistModelConfig, package="model")
cs.store(name="train", node=MnistTrainConfig, package="train")


@hydra.main(config_path="../../configs", config_name="mnist", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train the MNIST CNN with PyTorch + MLflow tracking."""
    set_seed(int(cfg.data.random_seed))

    train_cfg = cfg.train
    model_cfg = cfg.model

    banner("MNIST CLASSIFIER: 3-LAYER CNN")
    logger.info(
        "Architecture: Conv(%d->%d) -> Conv(%d->%d) -> FC(%d) -> %d",
        model_cfg.in_channels, model_cfg.conv_channels[0],
        model_cfg.conv_channels[0], model_cfg.conv_channels[1],
        model_cfg.fc_hidden, model_cfg.num_classes,
    )
    logger.info(
        "lr=%s epochs=%s batch_size=%s",
        train_cfg.learning_rate, train_cfg.epochs, train_cfg.batch_size,
    )

    # Data
    data = load_mnist_dataset(
        test_size=float(cfg.data.test_size),
        random_seed=int(cfg.data.random_seed),
    )

    train_loader = DataLoader(
        TensorDataset(data.X_train, data.y_train),
        batch_size=int(train_cfg.batch_size), shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(data.X_test, data.y_test),
        batch_size=int(train_cfg.batch_size), shuffle=False,
    )

    # Model
    model = SimpleCNN(
        in_channels=int(model_cfg.in_channels),
        conv_channels=tuple(model_cfg.conv_channels),
        fc_hidden=int(model_cfg.fc_hidden),
        num_classes=int(model_cfg.num_classes),
        dropout=float(model_cfg.dropout),
    )
    logger.info("Model: %s", model)
    logger.info("Total parameters: %d", sum(p.numel() for p in model.parameters()))

    criterion = nn_torch.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.learning_rate))

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],
    }

    mlf = MlflowLogger(experiment_name=str(train_cfg.mlflow_experiment))
    with mlf.start_run(
        run_name=f"mnist-cnn-dr{int(model_cfg.dropout*100)}-lr{train_cfg.learning_rate}"
    ):
        mlf.log_params(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

        log_every = max(1, int(train_cfg.log_every))
        for epoch in range(1, int(train_cfg.epochs) + 1):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += int((predicted == labels).sum().item())
            train_loss /= max(1, len(train_loader))
            train_acc = train_correct / max(1, train_total)

            # Eval
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += int((predicted == labels).sum().item())
            test_loss /= max(1, len(test_loader))
            test_acc = test_correct / max(1, test_total)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            mlf.log_metrics(
                {"train_loss": train_loss, "train_acc": train_acc,
                 "test_loss": test_loss, "test_acc": test_acc},
                step=epoch,
            )

            if epoch % log_every == 0 or epoch == 1:
                logger.info(
                    "Epoch %4d | train_loss=%.4f train_acc=%.4f test_loss=%.4f test_acc=%.4f",
                    epoch, train_loss, train_acc, test_loss, test_acc,
                )

        # Final evaluation
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(data.X_test), dim=1).cpu().numpy()
        class_names = tuple(str(i) for i in range(int(model_cfg.num_classes)))
        metrics = evaluate_classifier(
            data.y_test.cpu().numpy(), preds, class_names=class_names, label="test",
        )
        mlf.log_metrics(
            {f"final_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
        )

        # Plot
        plot_path = plot_training_history(
            history, save_path=plots_path("mnist_training_history.png"), title="MNIST",
        )
        mlf.log_artifact(plot_path)

        # Save registry
        # Resolve the next version + promotion decision. A run is
        # "promoted" when its test accuracy is at least
        # ``min_acc_delta`` better than the previous version's accuracy.
        resolution = resolve_promotion(
            project="mnist",
            candidate_accuracy=float(metrics["accuracy"]),
            metric_key="accuracy",
            min_acc_delta=float(train_cfg.get("min_acc_delta", 0.0)),
            require_metrics=True,
        )
        version = resolution.new_version
        mlf.log_metrics({"registry_promoted": int(resolution.promoted)})
        mlf.log_params({
            "registry_version": version,
            "registry_previous_version": str(resolution.previous_version or "none"),
        })
        logger.info(
            "Registry resolution: version=%s previous=%s promoted=%s — %s",
            version,
            resolution.previous_version,
            resolution.promoted,
            resolution.reason,
        )

        registry_dir = models_path(f"mnist/{version}")
        os.makedirs(registry_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(registry_dir, "model.pth"))
        # Save the model architecture config so the API can rebuild it.
        model_arch = {
            "in_channels": int(model_cfg.in_channels),
            "conv_channels": list(model_cfg.conv_channels),
            "fc_hidden": int(model_cfg.fc_hidden),
            "num_classes": int(model_cfg.num_classes),
            "dropout": float(model_cfg.dropout),
        }
        with open(os.path.join(registry_dir, "model_arch.json"), "w") as f:
            json.dump(model_arch, f, indent=2)
        with open(os.path.join(registry_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)
        with open(os.path.join(registry_dir, "metrics.json"), "w") as f:
            payload = {
                k: v for k, v in metrics.items() if isinstance(v, (int, float, str))
            }
            payload["registry"] = {
                "version": version,
                "previous_version": resolution.previous_version,
                "promoted": resolution.promoted,
                "reason": resolution.reason,
            }
            json.dump(payload, f, indent=2)
        with open(os.path.join(registry_dir, "class_names.json"), "w") as f:
            json.dump(list(class_names), f, indent=2)
        logger.info("Saved registry artifacts to %s", registry_dir)
        mlf.log_artifact(os.path.join(registry_dir, "metrics.json"))
        mlf.log_artifact(os.path.join(registry_dir, "config.yaml"))

        if not resolution.promoted and resolution.previous_version is not None:
            logger.info(
                "Run did not pass promotion gate; v%s not promoted "
                "(previous=%s remains the production candidate)",
                version, resolution.previous_version,
            )

    banner("MNIST TRAINING COMPLETE")
    logger.info("Final test accuracy: %.4f", metrics["accuracy"])


if __name__ == "__main__":
    main()
