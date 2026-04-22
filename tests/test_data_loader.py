"""Unit tests for the CIFAR-10 data loading pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.cifar10_loader import get_cifar10_loaders

TRAINSET_SIZE = 100


class FakeCIFAR10(Dataset):
    """Minimal CIFAR-10 stand-in for loader tests."""

    def __init__(
        self,
        root: Path,
        train: bool,
        download: bool,
        transform=None,
    ) -> None:
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.size = TRAINSET_SIZE if train else TRAINSET_SIZE // 2

    def __len__(self) -> int:
        """Return dataset size."""
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return a deterministic CIFAR-like sample."""
        value = int((index / self.size) * 255)
        image_array = np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
        image = Image.fromarray(image_array)
        label = index % 10
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _build_config(tmp_path: Path) -> dict:
    """Create a minimal loader config for tests."""
    return {
        "experiment": {"seed": 123},
        "data": {
            "root": str(tmp_path / "data"),
            "batch_size": 8,
            "num_workers": 0,
            "val_split": 0.2,
            "pin_memory": False,
        },
    }


def test_loader_returns_three_splits(monkeypatch, tmp_path: Path) -> None:
    """Loader helper should return train, val, and test loaders."""
    monkeypatch.setattr("src.data.cifar10_loader.datasets.CIFAR10", FakeCIFAR10)
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        _build_config(tmp_path)
    )
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None


def test_batch_shape(monkeypatch, tmp_path: Path) -> None:
    """Loader batches should match the expected CIFAR-10 tensor shapes."""
    monkeypatch.setattr("src.data.cifar10_loader.datasets.CIFAR10", FakeCIFAR10)
    train_loader, _, _ = get_cifar10_loaders(_build_config(tmp_path))
    images, targets = next(iter(train_loader))
    assert images.shape == (8, 3, 32, 32)
    assert targets.shape == (8,)


def test_no_overlap_between_train_val(monkeypatch, tmp_path: Path) -> None:
    """Train and validation subsets should use disjoint indices."""
    monkeypatch.setattr("src.data.cifar10_loader.datasets.CIFAR10", FakeCIFAR10)
    train_loader, val_loader, _ = get_cifar10_loaders(_build_config(tmp_path))
    train_indices = set(train_loader.dataset.indices)
    val_indices = set(val_loader.dataset.indices)
    assert train_indices.isdisjoint(val_indices)
