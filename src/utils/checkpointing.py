"""Checkpoint save and load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    config: dict,
    path: Path,
) -> None:
    """Save model, optimizer, and metadata to disk.

    Args:
        model: Model to serialize.
        optimizer: Optimizer state to serialize.
        epoch: Epoch at which the checkpoint was saved.
        val_acc: Validation accuracy associated with the checkpoint.
        config: Experiment configuration dictionary.
        path: Destination checkpoint path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
        "config": config,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a checkpoint into a model and optional optimizer.

    Args:
        path: Source checkpoint path.
        model: Model that should receive the saved weights.
        optimizer: Optional optimizer that should receive the saved state.

    Returns:
        Metadata dictionary containing saved checkpoint information.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return {
        "epoch": checkpoint["epoch"],
        "val_acc": checkpoint["val_acc"],
        "config": checkpoint["config"],
    }
