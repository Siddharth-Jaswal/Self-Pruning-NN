"""Package setup for the Self-Pruning Neural Network project."""

from setuptools import find_packages, setup

setup(
    name="self_pruning_nn",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
