"""PyTorch MLP for the Wine Quality binary classification task."""

from __future__ import annotations

from torch import nn


class WineNet(nn.Module):
    """MLP for wine-quality classification.

    Architecture: ``Input (in) -> Hidden (h, ReLU) -> Dropout ->
    Hidden (h/2, ReLU) -> Dropout -> Output (out)``.
    """

    def __init__(
        self,
        input_size: int = 11,
        hidden_sizes: list[int] | None = None,  # type: ignore[assignment]
        output_size: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Initialize layers.

        Args:
            input_size: Number of input features (11 for wine quality).
            hidden_sizes: Widths of the hidden layers. Defaults to ``[64, 32]``.
            output_size: Number of output classes.
            dropout: Dropout probability applied after each hidden layer.
        """
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        layers: list[nn.Module] = []
        prev = input_size
        for width in hidden_sizes:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


__all__ = ["WineNet"]