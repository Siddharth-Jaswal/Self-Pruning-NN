"""Sparsity regularization helpers for gated prunable models."""

from __future__ import annotations

import torch
from torch import nn

from src.layers.prunable_linear import PrunableLinear

DEFAULT_SPARSITY_THRESHOLD = 1e-2
DEFAULT_TARGET_MEAN_GATE = 0.2
DEFAULT_BINARIZATION_BETA = 0.1


def _collect_all_gates(model: nn.Module) -> torch.Tensor:
    """Collect all differentiable gate values from prunable layers.

    Args:
        model: Model containing one or more ``PrunableLinear`` layers.

    Returns:
        Flattened tensor of all sigmoid gate values.
    """
    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        raise ValueError("Model must contain at least one parameter.")

    all_gates: list[torch.Tensor] = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            all_gates.append(gates.reshape(-1))

    if not all_gates:
        return torch.zeros(1, device=first_parameter.device)
    return torch.cat(all_gates)


def compute_sparsity_loss(
    model: nn.Module,
    target_mean_gate: float = DEFAULT_TARGET_MEAN_GATE,
    binarization_beta: float = DEFAULT_BINARIZATION_BETA,
) -> torch.Tensor:
    """Compute sparsity regularization with one lambda-controlled objective.

    Regularizer:
    ``(mean(gates) - target_mean_gate)^2 + beta * mean(gates * (1 - gates))``

    Args:
        model: Model containing one or more ``PrunableLinear`` layers.
        target_mean_gate: Desired mean gate value.
        binarization_beta: Weight for the binarization-promoting term.

    Returns:
        Scalar tensor for sparsity regularization.
    """
    gates = _collect_all_gates(model)
    mean_gate = gates.mean()
    target_term = (mean_gate - target_mean_gate) ** 2
    binarization_term = (gates * (1.0 - gates)).mean()
    return target_term + (binarization_beta * binarization_term)


def compute_total_loss(
    classification_loss: torch.Tensor,
    model: nn.Module,
    lambda_sparsity: float,
    target_mean_gate: float = DEFAULT_TARGET_MEAN_GATE,
    binarization_beta: float = DEFAULT_BINARIZATION_BETA,
) -> torch.Tensor:
    """Combine classification and sparsity losses.

    Args:
        classification_loss: Base classification loss, typically cross entropy.
        model: Model containing prunable layers.
        lambda_sparsity: Weight for the sparsity regularization term.
        target_mean_gate: Desired mean gate value.
        binarization_beta: Weight for the binarization-promoting term.

    Returns:
        Total differentiable loss value.
    """
    sparsity_loss = compute_sparsity_loss(
        model,
        target_mean_gate=target_mean_gate,
        binarization_beta=binarization_beta,
    )
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
