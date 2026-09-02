"""Iris training entrypoint — Hydra-configured.

Usage:
    python -m src.training.train_iris
    python -m src.training.train_iris train.epochs=500 model.hidden_size=20
"""

from __future__ import annotations

import json
import os

import hydra
import joblib
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# Hydra <1.4 + Python 3.14 compat shim — must come BEFORE ``import hydra``.
import src.utils.hydra_compat  # noqa: F401
from src.configs.schemas import IrisDataConfig, IrisModelConfig, IrisTrainConfig
from src.data.iris_dataset import load_iris_dataset
from src.evaluation.evaluate import evaluate_classifier
from src.models.iris_nn import NeuralNetwork
from src.tracking.mlflow_logger import MlflowLogger
from src.utils import (
    banner,
    get_logger,
    models_path,
    plots_path,
    set_seed,
)
from src.visualization.plot_training import plot_training_history

logger = get_logger(__name__)


cs = ConfigStore.instance()
cs.store(name="data", node=IrisDataConfig, package="data")
cs.store(name="model", node=IrisModelConfig, package="model")
cs.store(name="train", node=IrisTrainConfig, package="train")


@hydra.main(config_path="../../configs", config_name="iris", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train the Iris NumPy NN with logging, MLflow, and registry saving."""
    set_seed(int(cfg.data.random_seed))

    train_cfg = cfg.train
    model_cfg = cfg.model

    banner("IRIS CLASSIFIER: 2-LAYER NEURAL NETWORK FROM SCRATCH")
    logger.info(
        "Architecture: Input(%d) -> Hidden(%d, ReLU) -> Output(%d, Softmax)",
        model_cfg.input_size, model_cfg.hidden_size, model_cfg.output_size,
    )
    logger.info("Learning rate=%s epochs=%s", model_cfg.learning_rate, train_cfg.epochs)

    # Data
    data = load_iris_dataset(
        test_size=float(cfg.data.test_size),
        random_state=int(cfg.data.random_seed),
    )

    # Model
    nn = NeuralNetwork(
        input_size=int(model_cfg.input_size),
        hidden_size=int(model_cfg.hidden_size),
        output_size=int(model_cfg.output_size),
        learning_rate=float(model_cfg.learning_rate),
    )

    # MLflow
    mlf = MlflowLogger(
        experiment_name=str(train_cfg.mlflow_experiment),
    )

    with mlf.start_run(run_name=f"iris-h{model_cfg.hidden_size}-lr{model_cfg.learning_rate}"):
        mlf.log_params(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

        # Train
        nn.train(
            data.X_train,
            data.Y_train,
            epochs=int(train_cfg.epochs),
            verbose=True,
            log_every=int(train_cfg.log_every),
        )

        # Per-epoch metrics for MLflow
        for epoch, (loss, acc) in enumerate(zip(nn.loss_history, nn.accuracy_history), start=1):
            mlf.log_metrics({"train_loss": loss, "train_acc": acc}, step=epoch)

        # Evaluate
        y_pred = nn.predict(data.X_test)
        metrics = evaluate_classifier(data.y_test, y_pred, class_names=data.class_names, label="test")
        mlf.log_metrics(
            {f"test_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
        )

        # Plot
        plot_path = plot_training_history(
            {"train_loss": nn.loss_history, "train_acc": nn.accuracy_history},
            save_path=plots_path("training_history.png"),
            title="Iris",
        )
        mlf.log_artifact(plot_path)

        # Save model + scaler + metadata under versioned registry
        version = "v1"
        registry_dir = models_path(f"iris/{version}")
        os.makedirs(registry_dir, exist_ok=True)

        np.savez(
            os.path.join(registry_dir, "model.npz"),
            W1=nn.W1, b1=nn.b1, W2=nn.W2, b2=nn.b2,
            loss_history=np.array(nn.loss_history),
            accuracy_history=np.array(nn.accuracy_history),
        )
        joblib.dump(data.scaler, os.path.join(registry_dir, "scaler.joblib"))
        with open(os.path.join(registry_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)
        with open(os.path.join(registry_dir, "metrics.json"), "w") as f:
            json.dump(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
                f, indent=2,
            )
        logger.info("Saved registry artifacts to %s", registry_dir)
        mlf.log_artifact(os.path.join(registry_dir, "metrics.json"))
        mlf.log_artifact(os.path.join(registry_dir, "config.yaml"))

    banner("TRAINING COMPLETE")
    logger.info("Final test accuracy: %.4f", metrics["accuracy"])


if __name__ == "__main__":
    main()
