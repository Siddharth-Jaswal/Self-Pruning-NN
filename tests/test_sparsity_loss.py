"""Unit tests for sparsity loss helpers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.losses import (
    compute_sparsity_loss,
    compute_sparsity_metric,
    compute_total_loss,
)


def test_loss_is_positive(small_model: nn.Module) -> None:
    """Sparsity loss should be non-negative."""
    loss = compute_sparsity_loss(small_model)
    assert loss.item() >= 0.0


def test_loss_has_grad(small_model: nn.Module) -> None:
    """Sparsity loss should remain on the computation graph."""
    loss = compute_sparsity_loss(small_model)
    assert loss.requires_grad is True


def test_total_loss_formula(
    small_model: nn.Module, dummy_batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Total loss should equal CE plus lambda times sparsity loss."""
    inputs, targets = dummy_batch
    criterion = nn.CrossEntropyLoss()
    logits = small_model(inputs)
    ce_loss = criterion(logits, targets)
    sparsity_loss = compute_sparsity_loss(small_model)
    lambda_sparsity = 0.1

    total_loss = compute_total_loss(ce_loss, small_model, lambda_sparsity)
    expected = ce_loss + (lambda_sparsity * sparsity_loss)
    assert torch.allclose(total_loss, expected)


def test_metric_range(small_model: nn.Module) -> None:
    """The sparsity metric should return a percentage in [0, 100]."""
    metric = compute_sparsity_metric(small_model)
    assert 0.0 <= metric <= 100.0


def test_high_lambda_gives_higher_loss(
    small_model: nn.Module, dummy_batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Higher lambda should yield a larger total loss for the same model."""
    inputs, targets = dummy_batch
    criterion = nn.CrossEntropyLoss()
    logits = small_model(inputs)
    ce_loss = criterion(logits, targets)

    low_lambda_loss = compute_total_loss(ce_loss, small_model, 0.001)
    high_lambda_loss = compute_total_loss(ce_loss, small_model, 1.0)
    assert high_lambda_loss.item() > low_lambda_loss.item()
