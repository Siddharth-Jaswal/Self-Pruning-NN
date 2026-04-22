"""Run all lambda experiments and generate a comparison plot."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import run_experiment
from src.utils.visualization import plot_lambda_comparison

LAMBDA_CONFIGS = [
    "configs/low_lambda.yaml",
    "configs/medium_lambda.yaml",
    "configs/high_lambda.yaml",
]
SEPARATOR_WIDTH = 60


def main() -> None:
    """Run all configured lambda experiments and print a summary table."""
    all_results: list[dict] = []
    for config_path in LAMBDA_CONFIGS:
        print(f"\n{'=' * SEPARATOR_WIDTH}")
        print(f"Running experiment: {config_path}")
        result = run_experiment(config_path)
        all_results.append(result)

    print("\n" + ("=" * SEPARATOR_WIDTH))
    print("LAMBDA COMPARISON RESULTS")
    print(f"{'Lambda':<12} {'Test Acc %':<15} {'Sparsity %':<15}")
    print("-" * 42)
    for result in all_results:
        print(
            f"{result['lambda']:<12.1e} "
            f"{result['test_acc']:<15.2f} "
            f"{result['sparsity_pct']:<15.2f}"
        )

    plot_lambda_comparison(
        all_results,
        PROJECT_ROOT / Path("outputs/plots/lambda_comparison.png"),
    )


if __name__ == "__main__":
    main()
