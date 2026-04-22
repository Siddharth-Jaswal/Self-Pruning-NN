"""Feed-forward CIFAR-10 classifier built from prunable linear layers."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.layers.prunable_linear import PrunableLinear

DEFAULT_HIDDEN_DIMS = [512, 256]
INPUT_FEATURES = 32 * 32 * 3
DEFAULT_SPARSITY_THRESHOLD = 1e-2


class PrunableNetwork(nn.Module):
    """Multi-layer perceptron for CIFAR-10 with learnable connection gates.

    Args:
        hidden_dims: Hidden layer sizes used to construct the network.
        num_classes: Number of output classes.
        dropout_rate: Dropout probability between hidden layers.
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        num_classes: int = 10,
        dropout_rate: float = 0.3,
    ) -> None:
        """Initialize the prunable feed-forward network."""
        super().__init__()
        resolved_hidden_dims = hidden_dims or list(DEFAULT_HIDDEN_DIMS)
        if not resolved_hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer.")

        layers: list[nn.Module] = []
        in_features = INPUT_FEATURES
        for hidden_dim in resolved_hidden_dims:
            layers.extend(
                [
                    PrunableLinear(in_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            in_features = hidden_dim

        layers.append(PrunableLinear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model on an image batch.

        Args:
            x: Input tensor of shape ``(batch_size, 3, 32, 32)``.

        Returns:
            Raw logits of shape ``(batch_size, num_classes)``.
        """
        x = x.view(x.size(0), -1)
        return self.network(x)

    def get_all_prunable_layers(self) -> list[PrunableLinear]:
        """Return all prunable layers in the network."""
        return [
            module for module in self.modules() if isinstance(module, PrunableLinear)
        ]

    def get_total_sparsity(
        self, threshold: float = DEFAULT_SPARSITY_THRESHOLD
    ) -> dict[str, Any]:
        """Compute per-layer and overall sparsity.

        Args:
            threshold: Evaluation-time pruning threshold.

        Returns:
            Dictionary containing overall sparsity, per-layer sparsity, and the
            number of prunable layers.
        """
        per_layer = [
            layer.sparsity(threshold) for layer in self.get_all_prunable_layers()
        ]
        overall = sum(per_layer) / len(per_layer) if per_layer else 0.0
        return {
            "overall": overall,
            "per_layer": per_layer,
            "num_prunable_layers": len(per_layer),
        }
