"""Shared pytest fixtures for the Self-Pruning Neural Network project."""

from __future__ import annotations

import pytest
import torch

from src.models import PrunableNetwork


@pytest.fixture
def small_model() -> PrunableNetwork:
    """Create a compact prunable model for unit tests."""
    return PrunableNetwork(hidden_dims=[32, 16])


@pytest.fixture
def dummy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a dummy CIFAR-10-shaped batch."""
    inputs = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 10, (4,))
    return inputs, targets
