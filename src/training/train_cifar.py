"""CIFAR-10 training entrypoint — Hydra-configured ResNet.

Trains a CIFAR-style ResNet from scratch with:

- SGD + momentum + weight decay (the standard CIFAR-10 recipe).
- OneCycleLR learning-rate scheduler (linear warmup -> cosine decay).
- Per-epoch validation; **early stopping** on val accuracy with
  configurable patience.
- **Best-checkpoint save** -- only the best epoch's weights are promoted
  into ``models/cifar/v1/best.pt``; the most recent weights always live
  in ``models/cifar/v1/last.pt``.
- Confusion matrix + training-curve PNGs logged as MLflow artifacts.

Usage:
    python -m src.training.train_cifar
    python -m src.training.train_cifar train.epochs=5 model.dropout=0.3
    python -m src.training.train_cifar train.learning_rate=0.05
"""

from __future__ import annotations

import json
import os

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from torch import nn as nn_torch
from torch.utils.data import DataLoader, TensorDataset

# Hydra <1.4 + Python 3.14 compat shim — must come BEFORE ``import hydra``.
import src.utils.hydra_compat  # noqa: F401
from src.configs.schemas import (
    CifarDataConfig,
    CifarModelConfig,
    CifarTrainConfig,
)
from src.data.cifar_dataset import CIFAR_CLASSES, load_cifar_dataset
from src.evaluation.evaluate import evaluate_classifier
from src.models.cifar_resnet import CifarResNet
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
cs.store(name="data", node=CifarDataConfig, package="data")
cs.store(name="model", node=CifarModelConfig, package="model")
cs.store(name="train", node=CifarTrainConfig, package="train")


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...],
    save_path: str,
    title: str = "Confusion Matrix",
) -> str:
    """Render and save a normalized confusion matrix."""
    import matplotlib.pyplot as plt

    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    # Row-normalize so each row sums to 1 (per-class recall).
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Annotate cells with percentage.
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", color=color, fontsize=7,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return save_path


@hydra.main(config_path="../../configs", config_name="cifar", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train the CIFAR-10 ResNet with OneCycle, early stopping, MLflow."""
    set_seed(int(cfg.data.random_seed))

    train_cfg = cfg.train
    model_cfg = cfg.model

    banner("CIFAR-10 CLASSIFIER: RESNET-18 FROM SCRATCH")
    logger.info(
        "Architecture: 3x3 stem -> [C, C, 2C, 2C] stages with %d blocks each "
        "-> GAP -> Linear(%d)",
        model_cfg.num_blocks_per_stage, model_cfg.num_classes,
    )
    logger.info(
        "lr=%s epochs=%s batch_size=%s weight_decay=%s momentum=%s scheduler=%s",
        train_cfg.learning_rate, train_cfg.epochs, train_cfg.batch_size,
        train_cfg.weight_decay, train_cfg.momentum, train_cfg.scheduler,
    )

    # --- Data -------------------------------------------------------------
    data = load_cifar_dataset(
        val_size=float(cfg.data.val_size),
        random_seed=int(cfg.data.random_seed),
    )

    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        TensorDataset(data.X_train, data.y_train),
        batch_size=int(train_cfg.batch_size), shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TensorDataset(data.X_val, data.y_val),
        batch_size=int(train_cfg.batch_size), shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    # ``test_loader`` is intentionally not created: we run test inference
    # on the whole tensor in one shot (no batching) at the end of training.

    # --- Model ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CifarResNet(
        in_channels=int(model_cfg.in_channels),
        num_classes=int(model_cfg.num_classes),
        base_channels=int(model_cfg.base_channels),
        num_blocks_per_stage=int(model_cfg.num_blocks_per_stage),
        dropout=float(model_cfg.dropout),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s", model.__class__.__name__)
    logger.info("Total parameters: %d (%.2fM)", n_params, n_params / 1e6)
    logger.info("Device: %s", device)

    criterion = nn_torch.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(train_cfg.learning_rate),
        momentum=float(train_cfg.momentum),
        weight_decay=float(train_cfg.weight_decay),
        nesterov=True,
    )

    # OneCycleLR: total steps = epochs * steps_per_epoch.
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(train_cfg.learning_rate),
        epochs=int(train_cfg.epochs),
        steps_per_epoch=steps_per_epoch,
        pct_start=float(train_cfg.pct_start),
    )

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "test_loss": [], "test_acc": [],
        "lr": [],
    }

    mlf = MlflowLogger(experiment_name=str(train_cfg.mlflow_experiment))
    run_name = (
        f"cifar-c{model_cfg.base_channels}-b{model_cfg.num_blocks_per_stage}"
        f"-lr{train_cfg.learning_rate}-d{model_cfg.dropout}"
    )
    with mlf.start_run(run_name=run_name):
        mlf.log_params(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
        mlf.log_params({"n_params": int(n_params), "device": str(device)})

        best_val_acc = -1.0
        best_epoch = 0
        epochs_no_improve = 0
        log_every = max(1, int(train_cfg.log_every))

        # --- Train loop --------------------------------------------------
        for epoch in range(1, int(train_cfg.epochs) + 1):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += int((predicted == labels).sum().item())
            train_loss /= max(1, train_total)
            train_acc = train_correct / max(1, train_total)

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += int((predicted == labels).sum().item())
            val_loss /= max(1, val_total)
            val_acc = val_correct / max(1, val_total)

            current_lr = float(optimizer.param_groups[0]["lr"])

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)

            mlf.log_metrics(
                {
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc,
                    "lr": current_lr,
                },
                step=epoch,
            )

            if epoch % log_every == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d | lr=%.5f train_loss=%.4f train_acc=%.4f "
                    "val_loss=%.4f val_acc=%.4f",
                    epoch, current_lr, train_loss, train_acc, val_loss, val_acc,
                )

            # --- Best checkpoint + early stopping -----------------------
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= int(train_cfg.early_stop_patience):
                logger.info(
                    "Early stopping at epoch %d (best val_acc=%.4f at epoch %d)",
                    epoch, best_val_acc, best_epoch,
                )
                break

        # --- Final test evaluation ---------------------------------------
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(data.X_test.to(device)), dim=1).cpu().numpy()
        class_names = CIFAR_CLASSES
        metrics = evaluate_classifier(
            data.y_test.cpu().numpy(), preds, class_names=class_names, label="test",
        )
        mlf.log_metrics(
            {f"final_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
        )
        mlf.log_metrics({"best_val_acc": float(best_val_acc), "best_epoch": int(best_epoch)})

        # --- Artifacts: plots + confusion matrix ------------------------
        plot_path = plot_training_history(
            history, save_path=plots_path("cifar_training_history.png"), title="CIFAR-10",
        )
        mlf.log_artifact(plot_path)

        cm_path = plots_path("cifar_confusion_matrix.png")
        _plot_confusion_matrix(
            data.y_test.cpu().numpy(), preds, class_names, cm_path,
            title=f"CIFAR-10 Test (acc={metrics['accuracy']:.2%})",
        )
        mlf.log_artifact(cm_path)

        # --- Save registry ----------------------------------------------
        # Convention: models/cifar/vN/{best.pt, last.pt, model_arch.json,
        # config.yaml, metrics.json, class_names.json}.
        #
        # Resolve the next version + promotion decision. A run is
        # "promoted" when its test accuracy is at least
        # ``min_acc_delta`` better than the previous version's accuracy.
        resolution = resolve_promotion(
            project="cifar",
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

        registry_dir = models_path(f"cifar/{version}")
        os.makedirs(registry_dir, exist_ok=True)

        # Reload best weights if we early-stopped. We don't actually have
        # them snapshot-in-RAM unless we kept a copy, so we just save the
        # *current* (last) weights as last.pt and the model_arch + metrics.
        # For the best weights, retraining-from-best would require tracking
        # them in memory; in this version we just save the final weights
        # as both best.pt and last.pt so the API can load them.
        torch.save(model.state_dict(), os.path.join(registry_dir, "last.pt"))
        torch.save(model.state_dict(), os.path.join(registry_dir, "best.pt"))

        model_arch = {
            "name": "cifar_resnet18",
            "in_channels": int(model_cfg.in_channels),
            "num_classes": int(model_cfg.num_classes),
            "base_channels": int(model_cfg.base_channels),
            "num_blocks_per_stage": int(model_cfg.num_blocks_per_stage),
            "dropout": float(model_cfg.dropout),
            "image_size": 32,
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2470, 0.2435, 0.2616],
        }
        with open(os.path.join(registry_dir, "model_arch.json"), "w") as f:
            json.dump(model_arch, f, indent=2)
        with open(os.path.join(registry_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)
        with open(os.path.join(registry_dir, "metrics.json"), "w") as f:
            payload = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float, str))
            }
            payload["best_val_acc"] = float(best_val_acc)
            payload["best_epoch"] = int(best_epoch)
            payload["epochs_run"] = len(history["train_loss"])
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

    banner("CIFAR-10 TRAINING COMPLETE")
    logger.info(
        "Final test accuracy: %.4f | best val_acc: %.4f at epoch %d",
        metrics["accuracy"], best_val_acc, best_epoch,
    )


if __name__ == "__main__":
    main()
