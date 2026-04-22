"""Visualization helpers for training dynamics and sparsity behavior."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch import nn

from src.layers.prunable_linear import PrunableLinear
from src.losses.sparsity_loss import compute_sparsity_metric

plt.style.use("seaborn-v0_8-whitegrid")

GATE_HIST_BINS = 100
BAR_WIDTH = 0.35
ROTATION_DEGREES = 15


def _collect_gate_values(model: nn.Module) -> np.ndarray:
    """Collect all gate values from prunable layers into one array."""
    gate_values = [
        layer.get_gates().cpu().numpy().ravel()
        for layer in model.modules()
        if isinstance(layer, PrunableLinear)
    ]
    if not gate_values:
        return np.array([], dtype=np.float32)
    return np.concatenate(gate_values)


def plot_gate_distribution(
    model: nn.Module,
    save_path: Path,
    title: str = "Gate Value Distribution",
    threshold: float = 1e-2,
) -> None:
    """Plot the distribution of gate values across all prunable layers.

    Args:
        model: Model containing prunable layers.
        save_path: Output figure path.
        title: Plot title.
        threshold: Threshold used to annotate the pruned fraction.
    """
    gate_values = _collect_gate_values(model)
    sparsity_pct = compute_sparsity_metric(model, threshold)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(gate_values, bins=GATE_HIST_BINS, color="steelblue", alpha=0.85)
    axis.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.1e}",
    )
    axis.set_yscale("log")
    axis.set_title(title)
    axis.set_xlabel("Gate Value")
    axis.set_ylabel("Count (log scale)")
    axis.annotate(
        f"{sparsity_pct:.1f}% gates below threshold",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(save_path)
    plt.close(figure)


def plot_loss_curves(history: dict, save_path: Path, title: str = "") -> None:
    """Plot training total loss and validation accuracy over epochs.

    Args:
        history: Training history dictionary.
        save_path: Output figure path.
        title: Optional plot title.
    """
    epochs = np.arange(1, len(history["train_total_loss"]) + 1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure, left_axis = plt.subplots(figsize=(10, 6))
    right_axis = left_axis.twinx()

    left_axis.plot(
        epochs,
        history["train_total_loss"],
        color="blue",
        linewidth=2,
        label="Train Total Loss",
    )
    right_axis.plot(
        epochs,
        history["val_acc"],
        color="red",
        linewidth=2,
        label="Validation Accuracy",
    )

    left_axis.set_xlabel("Epoch")
    left_axis.set_ylabel("Train Total Loss", color="blue")
    right_axis.set_ylabel("Validation Accuracy (%)", color="red")
    left_axis.set_title(title or "Training Loss and Validation Accuracy")

    figure.tight_layout()
    figure.savefig(save_path)
    plt.close(figure)


def plot_lambda_comparison(results: list[dict], save_path: Path) -> None:
    """Plot accuracy and sparsity across lambda experiments.

    Args:
        results: List of experiment result dictionaries.
        save_path: Output figure path.
    """
    labels = [f"{result['lambda']:.1e}" for result in results]
    test_accuracy = [result["test_acc"] for result in results]
    sparsity = [result["sparsity_pct"] for result in results]
    positions = np.arange(len(results))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure, left_axis = plt.subplots(figsize=(10, 6))
    right_axis = left_axis.twinx()

    left_axis.bar(
        positions - (BAR_WIDTH / 2),
        test_accuracy,
        width=BAR_WIDTH,
        color="cornflowerblue",
        label="Test Accuracy (%)",
    )
    right_axis.bar(
        positions + (BAR_WIDTH / 2),
        sparsity,
        width=BAR_WIDTH,
        color="salmon",
        label="Sparsity (%)",
    )

    left_axis.set_xlabel("Lambda")
    left_axis.set_ylabel("Test Accuracy (%)", color="cornflowerblue")
    right_axis.set_ylabel("Sparsity (%)", color="salmon")
    left_axis.set_xticks(positions)
    left_axis.set_xticklabels(labels, rotation=ROTATION_DEGREES)
    left_axis.set_title("Lambda Comparison")

    left_handles, left_labels = left_axis.get_legend_handles_labels()
    right_handles, right_labels = right_axis.get_legend_handles_labels()
    left_axis.legend(left_handles + right_handles, left_labels + right_labels)

    figure.tight_layout()
    figure.savefig(save_path)
    plt.close(figure)
