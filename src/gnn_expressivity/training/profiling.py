"""Resource measurements. Memory values use MiB (1024 ** 2 bytes)."""

import platform
import time

import psutil
import torch
import torch_geometric

from .reproducibility import get_device


def count_parameters(model: torch.nn.Module) -> int:
    """Count only parameters that require gradients."""
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class Timer:
    """Measure a block with perf_counter and accelerator synchronization.

    Use separate instances for preprocessing, training, and inference:
    ``with Timer(device) as training: ...`` then read ``training.elapsed_sec``.
    Pass ``"cpu"`` for CPU-only preprocessing. The default is get_device().
    Elapsed time remains None until the block has exited.
    """

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.device = get_device() if device is None else torch.device(device)
        self.elapsed_sec: float | None = None

    def __enter__(self) -> "Timer":
        self.elapsed_sec = None
        _synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        _synchronize(self.device)
        self.elapsed_sec = time.perf_counter() - self._start


def process_memory_mb() -> float:
    """Return current process RSS in MiB."""
    return psutil.Process().memory_info().rss / (1024 ** 2)


def reset_cuda_peak_memory(device: str | torch.device | None = None) -> None:
    """Reset peak stats on the specified (or current) CUDA device, if available."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_cuda_memory_mb(device: str | torch.device | None = None) -> float | None:
    """Return peak allocated CUDA MiB since reset, or None without CUDA."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def system_metadata(device: str | torch.device | None = None) -> dict[str, str | None]:
    """Describe the selected device and software used for a run."""
    device = get_device() if device is None else torch.device(device)
    accelerator = None
    if device.type == "cuda":
        accelerator = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        accelerator = "Apple MPS"  # Backend name; exact chip name is not exposed.
    return {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_geometric_version": str(torch_geometric.__version__),
        "device": str(device),
        "accelerator": accelerator,
    }
