"""Generic training-curve plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_training_history(
    history: dict[str, list[float]],
    save_path: str,
    title: str = "Training History",
) -> str:
    """Plot train/test loss and accuracy curves side-by-side.

    Args:
        history: Dict with keys ``train_loss``, ``test_loss``, ``train_acc``,
            ``test_acc`` (any subset is plotted when present).
        save_path: Destination file path.
        title: Figure title.

    Returns:
        The ``save_path`` for chaining.
    """
    has_loss = "train_loss" in history
    has_acc = "train_acc" in history
    n_panels = int(has_loss) + int(has_acc)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    idx = 0
    if has_loss:
        ax = axes[idx]
        ax.plot(history["train_loss"], label="Train Loss", linewidth=2)
        if "test_loss" in history:
            ax.plot(history["test_loss"], label="Test Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{title}: Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        idx += 1
    if has_acc:
        ax = axes[idx]
        ax.plot(history["train_acc"], label="Train Acc", linewidth=2)
        if "test_acc" in history:
            ax.plot(history["test_acc"], label="Test Acc", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title}: Accuracy")
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return save_path


__all__ = ["plot_training_history"]
