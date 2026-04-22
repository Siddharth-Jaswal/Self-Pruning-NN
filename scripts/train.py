"""Entry point for running a single self-pruning network experiment."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_cifar10_loaders
from src.models import PrunableNetwork
from src.training import Evaluator, Trainer
from src.utils import get_logger, load_checkpoint, load_config
from src.utils.visualization import plot_gate_distribution, plot_loss_curves

RESULTS_DIRNAME = "results"


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train a self-pruning neural network on CIFAR-10."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def setup_device() -> torch.device:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Build the experiment optimizer from config values.

    Args:
        model: Model whose parameters should be optimized.
        config: Project configuration dictionary.

    Returns:
        Configured Adam optimizer.
    """
    training_config = config["training"]
    optimizer_name = training_config["optimizer"].lower()
    if optimizer_name != "adam":
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")

    return torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )


def _save_results_json(results: dict[str, Any], experiment_name: str) -> None:
    """Persist experiment results to a JSON artifact."""
    results_dir = PROJECT_ROOT / "outputs" / RESULTS_DIRNAME
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{experiment_name}_results.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)


def run_experiment(config_path: str) -> dict[str, Any]:
    """Run the full training, checkpoint, evaluation, and plotting pipeline.

    Args:
        config_path: Path to the YAML experiment config.

    Returns:
        Dictionary with final experiment results.
    """
    config = load_config(config_path)
    experiment_name = config["experiment"]["name"]
    output_config = config["output"]

    logs_dir = PROJECT_ROOT / Path(output_config["logs_dir"])
    log_file = logs_dir / f"{experiment_name}.log"
    logger = get_logger("spnn.train", log_file=log_file)

    set_seed(config["experiment"]["seed"])
    device = setup_device()
    logger.info("Using device: %s", device)

    train_loader, val_loader, test_loader = get_cifar10_loaders(config)

    model = PrunableNetwork(**config["model"]).to(device)
    optimizer = build_optimizer(model, config)

    checkpoint_dir = PROJECT_ROOT / Path(output_config["checkpoint_dir"])
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=config,
        device=device,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
    )
    history = trainer.train()

    checkpoint_path = checkpoint_dir / f"{experiment_name}_best.pt"
    load_checkpoint(checkpoint_path, model, optimizer=None)

    evaluator = Evaluator(device=device, logger=logger)
    threshold = config["evaluation"]["sparsity_threshold"]
    evaluation_results = evaluator.evaluate(model, test_loader, threshold=threshold)

    plots_dir = PROJECT_ROOT / Path(output_config["plots_dir"])
    plot_gate_distribution(
        model,
        plots_dir / f"gate_dist_{experiment_name}.png",
        title=f"Gate Distribution - {experiment_name}",
        threshold=threshold,
    )
    plot_loss_curves(
        history,
        plots_dir / f"loss_curve_{experiment_name}.png",
        title=f"Training Curves - {experiment_name}",
    )

    results = {
        "experiment_name": experiment_name,
        "lambda": config["experiment"]["lambda_sparsity"],
        **evaluation_results,
    }
    _save_results_json(results, experiment_name)
    logger.info("Final results: %s", results)
    return results


def _print_results_table(results: dict[str, Any]) -> None:
    """Print a compact results table to stdout."""
    print("=" * 60)
    print("EXPERIMENT RESULTS")
    print("-" * 60)
    print(f"{'Experiment':<20} {results['experiment_name']}")
    print(f"{'Lambda':<20} {results['lambda']:.1e}")
    print(f"{'Test Acc %':<20} {results['test_acc']:.2f}")
    print(f"{'Test Loss':<20} {results['test_loss']:.4f}")
    print(f"{'Sparsity %':<20} {results['sparsity_pct']:.2f}")
    print(f"{'Pruned Params':<20} {results['pruned_params']}")
    print(f"{'Total Params':<20} {results['total_params']}")


if __name__ == "__main__":
    arguments = parse_args()
    experiment_results = run_experiment(arguments.config)
    _print_results_table(experiment_results)
