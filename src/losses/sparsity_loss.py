"""Sparsity regularization helpers for gated prunable models."""

from __future__ import annotations

import torch
from torch import nn

from src.layers.prunable_linear import PrunableLinear

DEFAULT_SPARSITY_THRESHOLD = 1e-2


def compute_sparsity_loss(model: nn.Module) -> torch.Tensor:
    """Compute the L1-style sparsity regularization term over gate values.

    Args:
        model: Model containing one or more ``PrunableLinear`` layers.

    Returns:
        Tensor equal to the sum of all gate values across prunable parameters.
    """
    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        raise ValueError("Model must contain at least one parameter.")

    total_gate_sum = torch.zeros((), device=first_parameter.device)
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total_gate_sum = total_gate_sum + gates.sum()
    return total_gate_sum


def compute_total_loss(
    classification_loss: torch.Tensor,
    model: nn.Module,
    lambda_sparsity: float,
) -> torch.Tensor:
    """Combine classification and sparsity losses.

    Args:
        classification_loss: Base classification loss, typically cross entropy.
        model: Model containing prunable layers.
        lambda_sparsity: Weight for the sparsity regularization term.

    Returns:
        Total differentiable loss value.
    """
    sparsity_loss = compute_sparsity_loss(model)
    return classification_loss + (lambda_sparsity * sparsity_loss)


def compute_sparsity_metric(
    model: nn.Module, threshold: float = DEFAULT_SPARSITY_THRESHOLD
) -> float:
    """Compute the percentage of gates below the evaluation threshold.

    Args:
        model: Model containing prunable layers.
        threshold: Gate threshold used to count a connection as pruned.

    Returns:
        Percentage of gates with value below ``threshold``.
    """
    total_gates = 0
    pruned_gates = 0

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                total_gates += gates.numel()
                pruned_gates += (gates < threshold).sum().item()

    if total_gates == 0:
        return 0.0
    return (pruned_gates / total_gates) * 100.0


def compute_mean_gate_value(model: nn.Module) -> float:
    """Compute the mean gate value across all prunable parameters.

    Args:
        model: Model containing prunable layers.

    Returns:
        Mean sigmoid gate value across the whole model.
    """
    total_gate_value = 0.0
    total_gates = 0

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                total_gate_value += gates.sum().item()
                total_gates += gates.numel()

    if total_gates == 0:
        return 0.0
    return total_gate_value / total_gates
