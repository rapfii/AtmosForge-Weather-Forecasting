"""Global seed management for reproducibility.

Ensures deterministic behavior across PyTorch, NumPy, Python random,
CUDA operations, and DataLoader workers. Must be called at the start
of every training run.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for full reproducibility across all components.

    Fixes randomness in:
    - Python's built-in random module
    - NumPy random number generator
    - PyTorch CPU random number generator
    - PyTorch CUDA random number generator (all GPUs)
    - cuDNN backend (deterministic mode)
    - DataLoader worker processes (via PYTHONHASHSEED)

    Args:
        seed: Integer seed value. Default: 42.

    Note:
        Setting CUBLAS_WORKSPACE_CONFIG is required for full determinism
        on CUDA 10.2+. This may have a minor performance impact.

    Example:
        >>> from src.utils.seed import set_seed
        >>> set_seed(42)  # Call at the start of every experiment
    """
    # Python built-in random module
    random.seed(seed)

    # Environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy random number generator
    np.random.seed(seed)  # noqa: NPY002

    # PyTorch CPU random number generator
    torch.manual_seed(seed)

    # PyTorch CUDA random number generator (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # cuDNN deterministic mode (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Required for full determinism on CUDA 10.2+ with certain operations
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def worker_init_fn(worker_id: int) -> None:
    """Initialize DataLoader worker with deterministic seed.

    Ensures each DataLoader worker produces reproducible data sequences.
    Pass this to DataLoader's worker_init_fn parameter.

    Args:
        worker_id: Worker index (provided automatically by DataLoader).

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, worker_init_fn=worker_init_fn)
    """
    # Derive per-worker seed from base seed + worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002
    random.seed(worker_seed)


def get_device() -> torch.device:
    """Get the best available compute device.

    Prioritizes CUDA GPU (RTX 4050) for training efficiency.

    Returns:
        torch.device for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
