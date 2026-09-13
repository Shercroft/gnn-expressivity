"""Random seeds and automatic accelerator selection."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed generators and select deterministic cuDNN behavior.

    PYTHONHASHSEED affects subsequently started interpreters, not the current
    interpreter. Seeds do not guarantee identical results across platforms or
    deterministic behavior for every accelerator operation.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
