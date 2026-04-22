"""CIFAR-10 dataset loading utilities with augmentation and validation split."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
IMAGE_SIZE = 32
CROP_PADDING = 4


def get_cifar10_loaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test data loaders for CIFAR-10.

    Args:
        config: Project configuration dictionary.

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader)``.
    """
    data_config = config["data"]
    seed = config["experiment"]["seed"]
    root = Path(data_config["root"])

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(IMAGE_SIZE, padding=CROP_PADDING),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_dataset_full = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=train_transform,
    )
    val_dataset_full = datasets.CIFAR10(
        root=root,
        train=True,
        download=False,
        transform=eval_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=eval_transform,
    )

    total_train_examples = len(train_dataset_full)
    val_size = int(total_train_examples * data_config["val_split"])
    train_size = total_train_examples - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset_indices = torch.utils.data.random_split(
        range(total_train_examples),
        [train_size, val_size],
        generator=generator,
    )

    train_dataset = Subset(train_dataset_full, train_subset.indices)
    val_dataset = Subset(val_dataset_full, val_subset_indices.indices)

    loader_kwargs = {
        "batch_size": data_config["batch_size"],
        "num_workers": data_config["num_workers"],
        "pin_memory": data_config["pin_memory"],
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def get_cifar10_classes() -> list[str]:
    """Return the CIFAR-10 class names."""
    return list(CIFAR10_CLASSES)
