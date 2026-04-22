# Self-Pruning Neural Network

Feed-forward CIFAR-10 classifier that learns connection sparsity during training through sigmoid-gated weights and L1 regularization.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run One Experiment

```bash
python scripts/train.py --config configs/medium_lambda.yaml
```

## Run All Experiments

```bash
python scripts/run_experiments.py
```

## Run Tests

```bash
pytest tests/ --cov=src
```

## Repository Structure

```text
Self-Pruning-NN/
├── README.md
├── requirements.txt
├── setup.py
├── Makefile
├── .gitignore
├── configs/
│   ├── base_config.yaml
│   ├── low_lambda.yaml
│   ├── medium_lambda.yaml
│   └── high_lambda.yaml
├── src/
│   ├── layers/
│   │   └── prunable_linear.py
│   ├── models/
│   │   └── prunable_network.py
│   ├── losses/
│   │   └── sparsity_loss.py
│   ├── data/
│   │   └── cifar10_loader.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── evaluator.py
│   └── utils/
│       ├── config_loader.py
│       ├── logger.py
│       ├── checkpointing.py
│       └── visualization.py
├── scripts/
│   ├── train.py
│   └── run_experiments.py
├── tests/
│   ├── conftest.py
│   ├── test_prunable_linear.py
│   ├── test_sparsity_loss.py
│   └── test_data_loader.py
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── plots/
└── workflow/
```

## Self-Pruning Mechanism

Each `PrunableLinear` layer learns both standard weights and a matching tensor of `gate_scores`. During the forward pass, the layer computes `sigmoid(gate_scores)` and multiplies those gates element-wise with the weights before applying the linear transform.

Training minimizes classification loss plus `lambda_sparsity * sum(sigmoid(gate_scores))`. Larger lambda values penalize open gates more aggressively, which pushes unnecessary connections toward zero contribution without hard-thresholding them during training.
