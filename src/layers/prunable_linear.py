"""Prunable linear layer with learnable sigmoid gates."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_SPARSITY_THRESHOLD = 1e-2


class PrunableLinear(nn.Module):
    """Linear layer whose weights are modulated by learnable gates.

    Each weight is multiplied by a gate computed as the sigmoid of a learnable
    gate score. This allows the network to suppress individual connections
    during training while remaining fully differentiable.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to include a learnable bias term.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """Initialize the prunable linear layer."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated linear transformation.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return detached gate values for inspection."""
        return torch.sigmoid(self.gate_scores).detach().clone()

    def sparsity(self, threshold: float = DEFAULT_SPARSITY_THRESHOLD) -> float:
        """Compute the fraction of gates below the pruning threshold.

        Args:
            threshold: Evaluation-time threshold used to count a gate as pruned.

        Returns:
            Fraction of gates below ``threshold``.
        """
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()

    def extra_repr(self) -> str:
        """Return an informative module representation."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
