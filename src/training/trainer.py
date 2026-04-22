"""Training loop implementation for the Self-Pruning Neural Network project."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses.sparsity_loss import (
    compute_mean_gate_value,
    compute_sparsity_loss,
    compute_sparsity_metric,
    compute_total_loss,
)
from src.utils.checkpointing import save_checkpoint


class Trainer:
    """Encapsulate model training, validation, and checkpointing.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer for model parameters.
        config: Project configuration dictionary.
        device: Target device.
        logger: Project logger.
        checkpoint_dir: Directory where best checkpoints should be stored.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: dict,
        device: torch.device,
        logger: logging.Logger,
        checkpoint_dir: Path,
    ) -> None:
        """Initialize the trainer state."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir

        self.criterion = nn.CrossEntropyLoss()
        self.lambda_sparsity = config["experiment"]["lambda_sparsity"]
        self.target_mean_gate = config["experiment"].get("target_mean_gate", 0.2)
        self.binarization_beta = config["experiment"].get(
            "binarization_beta", 0.1
        )
        self.num_epochs = config["training"]["num_epochs"]
        self.history = {
            "train_ce_loss": [],
            "train_sparsity_loss": [],
            "train_mean_gate": [],
            "train_total_loss": [],
            "val_acc": [],
            "val_loss": [],
            "sparsity_pct": [],
        }
        self.best_val_acc = 0.0

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run a single training epoch.

        Args:
            epoch: One-based epoch index.

        Returns:
            Dictionary with average cross-entropy, sparsity, and total loss.
        """
        self.model.train()
        total_ce_loss = 0.0
        total_sparsity_loss = 0.0
        total_loss = 0.0

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.num_epochs}",
            leave=False,
        )
        for inputs, targets in progress:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(inputs)
            ce_loss = self.criterion(logits, targets)
            sparsity_loss = compute_sparsity_loss(
                self.model,
                target_mean_gate=self.target_mean_gate,
                binarization_beta=self.binarization_beta,
            )
            batch_total_loss = compute_total_loss(
                ce_loss,
                self.model,
                self.lambda_sparsity,
                target_mean_gate=self.target_mean_gate,
                binarization_beta=self.binarization_beta,
            )

            self.optimizer.zero_grad()
            batch_total_loss.backward()
            self.optimizer.step()

            total_ce_loss += ce_loss.item()
            total_sparsity_loss += sparsity_loss.item()
            total_loss += batch_total_loss.item()
            progress.set_postfix(
                ce_loss=f"{ce_loss.item():.4f}",
                total_loss=f"{batch_total_loss.item():.4f}",
            )

        num_batches = max(len(self.train_loader), 1)
        return {
            "ce_loss": total_ce_loss / num_batches,
            "sparsity_loss": total_sparsity_loss / num_batches,
            "total_loss": total_loss / num_batches,
            "mean_gate": compute_mean_gate_value(self.model),
        }

    def validate(self) -> dict[str, float]:
        """Run validation and return loss and accuracy metrics."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                total_loss += loss.item()

                predictions = logits.argmax(dim=1)
                total += targets.size(0)
                correct += (predictions == targets).sum().item()

        num_batches = max(len(self.val_loader), 1)
        val_acc = (correct / total) * 100.0 if total else 0.0
        return {"val_acc": val_acc, "val_loss": total_loss / num_batches}

    def train(self) -> dict[str, list[float]]:
        """Run the full training loop and save the best checkpoint.

        Returns:
            History dictionary containing losses, validation metrics, and
            sparsity percentages for each epoch.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        experiment_name = self.config["experiment"]["name"]

        for epoch in range(1, self.num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            sparsity_pct = compute_sparsity_metric(self.model)

            self.history["train_ce_loss"].append(train_metrics["ce_loss"])
            self.history["train_sparsity_loss"].append(
                train_metrics["sparsity_loss"]
            )
            self.history["train_mean_gate"].append(train_metrics["mean_gate"])
            self.history["train_total_loss"].append(train_metrics["total_loss"])
            self.history["val_acc"].append(val_metrics["val_acc"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["sparsity_pct"].append(sparsity_pct)

            if val_metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_acc"]
                checkpoint_path = (
                    self.checkpoint_dir / f"{experiment_name}_best.pt"
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    val_acc=self.best_val_acc,
                    config=self.config,
                    path=checkpoint_path,
                )

            self.logger.info(
                (
                    "epoch=%s train_ce=%.4f train_sparsity=%.4f "
                    "mean_gate=%.4f train_total=%.4f val_loss=%.4f "
                    "val_acc=%.2f sparsity=%.2f"
                ),
                epoch,
                train_metrics["ce_loss"],
                train_metrics["sparsity_loss"],
                train_metrics["mean_gate"],
                train_metrics["total_loss"],
                val_metrics["val_loss"],
                val_metrics["val_acc"],
                sparsity_pct,
            )

        return self.history
