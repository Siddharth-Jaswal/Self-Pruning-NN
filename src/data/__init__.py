"""Data loading helpers for the Self-Pruning Neural Network project."""

from .cifar10_loader import get_cifar10_classes, get_cifar10_loaders

__all__ = ["get_cifar10_classes", "get_cifar10_loaders"]
