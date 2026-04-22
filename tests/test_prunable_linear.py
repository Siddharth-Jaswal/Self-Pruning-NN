"""Unit tests for the PrunableLinear layer."""

from __future__ import annotations

import pytest
import torch

from src.layers import PrunableLinear


def test_output_shape() -> None:
    """PrunableLinear should preserve the expected output shape."""
    layer = PrunableLinear(8, 4)
    output = layer(torch.randn(2, 8))
    assert output.shape == (2, 4)


def test_gate_range() -> None:
    """Gate values should remain strictly between zero and one."""
    layer = PrunableLinear(8, 4)
    gates = layer.get_gates()
    assert torch.all(gates > 0.0)
    assert torch.all(gates < 1.0)


def test_weight_grad_flows() -> None:
    """Backpropagation should produce nonzero gradients on weights."""
    layer = PrunableLinear(8, 4)
    loss = layer(torch.randn(2, 8)).sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert torch.count_nonzero(layer.weight.grad).item() > 0


def test_gate_scores_grad_flows() -> None:
    """Backpropagation should produce nonzero gradients on gate scores."""
    layer = PrunableLinear(8, 4)
    loss = layer(torch.randn(2, 8)).sum()
    loss.backward()
    assert layer.gate_scores.grad is not None
    assert torch.count_nonzero(layer.gate_scores.grad).item() > 0


def test_zero_gate_zeroes_contribution() -> None:
    """Extreme negative gate scores should suppress weight contributions."""
    input_tensor = torch.randn(2, 8)
    layer_one = PrunableLinear(8, 4, bias=False)
    layer_two = PrunableLinear(8, 4, bias=False)

    with torch.no_grad():
        layer_one.gate_scores.fill_(-100.0)
        layer_two.gate_scores.fill_(-100.0)
        layer_one.weight.fill_(1.0)
        layer_two.weight.fill_(-7.0)

    output_one = layer_one(input_tensor)
    output_two = layer_two(input_tensor)
    assert torch.allclose(output_one, output_two, atol=1e-6)


def test_sparsity_with_all_pruned() -> None:
    """All strongly negative gate scores should count as pruned."""
    layer = PrunableLinear(8, 4)
    with torch.no_grad():
        layer.gate_scores.fill_(-100.0)
    assert layer.sparsity() == pytest.approx(1.0, rel=0.0, abs=1e-6)


def test_sparsity_with_none_pruned() -> None:
    """All strongly positive gate scores should count as unpruned."""
    layer = PrunableLinear(8, 4)
    with torch.no_grad():
        layer.gate_scores.fill_(100.0)
    assert layer.sparsity() == pytest.approx(0.0, rel=0.0, abs=1e-6)


def test_no_bias() -> None:
    """Disabling bias should leave the module without a bias parameter."""
    layer = PrunableLinear(4, 4, bias=False)
    assert layer.bias is None
