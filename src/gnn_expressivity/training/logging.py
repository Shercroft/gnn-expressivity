"""Per-run JSON logging and the standard experiment result schema."""

import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, TypedDict


class RunResult(TypedDict):
    """Standard flat result record.

    timestamp is an ISO 8601 UTC string. Use None for measurements that were
    not collected or are unavailable; never substitute invented metrics.
    primary_metric is the task's primary score. All times are seconds and
    memory fields use MiB (1024 ** 2 bytes). This schema is for static typing,
    not runtime validation; save_run also accepts partial records.
    """

    run_id: str
    timestamp: str
    git_commit: str
    seed: int
    task: str
    dataset: str
    model: str
    encoding: str
    hidden_dim: int
    num_layers: int
    primary_metric: float | None
    loss: float | None
    num_parameters: int | None
    preprocessing_time_sec: float | None
    training_time_sec: float | None
    inference_time_sec: float | None
    process_memory_mb: float | None
    peak_gpu_memory_mb: float | None
    os: str
    python_version: str
    torch_version: str
    torch_geometric_version: str
    device: str
    accelerator: str | None


def get_git_commit() -> str:
    """Return this source checkout's short commit, or unknown outside Git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_run_id(config: Mapping[str, Any], seed: int) -> str:
    """Build an ID from the existing merged, nested config sections."""
    return (
        f"{config['task']['name']}_{config['model']['name']}_"
        f"{config['encoding']['name']}_seed{seed}"
    )


def save_run(
    result: Mapping[str, Any], results_dir: str | Path = Path("results/runs")
) -> Path:
    """Write <run_id>.json and return its path; refuse to overwrite a run.

    Values must be JSON serializable and finite. Convert tensors/NumPy values
    to Python scalars before calling. Invalid records are not written.
    """
    run_id = result["run_id"]
    if not isinstance(run_id, str) or not run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in run_id
    ):
        raise ValueError("run_id must contain only ASCII letters, digits, '_' or '-'")
    payload = json.dumps(dict(result), indent=2, ensure_ascii=False, allow_nan=False)
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.json"
    with path.open("x", encoding="utf-8") as file:
        file.write(payload + "\n")
    return path
