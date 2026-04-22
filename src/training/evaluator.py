"""Evaluation helpers for the Self-Pruning Neural Network project."""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.layers.prunable_linear import PrunableLinear
from src.losses.sparsity_loss import compute_sparsity_metric


class Evaluator:
    """Evaluate trained models on the test split.

    Args:
        device: Device used for model inference.
        logger: Project logger.
    """

    def __init__(self, device: torch.device, logger: logging.Logger) -> None:
        """Initialize the evaluator."""
        self.device = device
        self.logger = logger
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        threshold: float = 1e-2,
    ) -> dict[str, float | int]:
        """Run evaluation over the test set.

        Args:
            model: Trained model to evaluate.
            test_loader: Data loader for the test split.
            threshold: Gate threshold used for the sparsity metric.

        Returns:
            Dictionary containing test accuracy, test loss, sparsity, and
            parameter counts.
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total_examples = 0
        total_params = 0
        pruned_params = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits = model(inputs)
                loss = self.criterion(logits, targets)
                total_loss += loss.item()

                predictions = logits.argmax(dim=1)
                total_examples += targets.size(0)
                correct += (predictions == targets).sum().item()

            for module in model.modules():
                if isinstance(module, PrunableLinear):
                    gates = torch.sigmoid(module.gate_scores)
                    total_params += gates.numel()
                    pruned_params += (gates < threshold).sum().item()

        test_acc = (correct / total_examples) * 100.0 if total_examples else 0.0
        test_loss = total_loss / max(len(test_loader), 1)
        sparsity_pct = compute_sparsity_metric(model, threshold)

        results = {
            "test_acc": test_acc,
            "test_loss": test_loss,
            "sparsity_pct": sparsity_pct,
            "total_params": total_params,
            "pruned_params": pruned_params,
        }
        self.logger.info(
            "test_loss=%.4f test_acc=%.2f sparsity=%.2f pruned=%s/%s",
            test_loss,
            test_acc,
            sparsity_pct,
            pruned_params,
            total_params,
        )
        return results
