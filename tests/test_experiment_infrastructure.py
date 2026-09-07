import json
import os
from pathlib import Path
import random
import subprocess
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from gnn_expressivity.training.config import load_yaml, merge_configs
from gnn_expressivity.training import logging, profiling, reproducibility


def test_set_seed_repeats_random_values():
    reproducibility.set_seed(42)
    first = (random.random(), np.random.rand(3), torch.rand(3))
    reproducibility.set_seed(42)
    second = (random.random(), np.random.rand(3), torch.rand(3))
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert os.environ["PYTHONHASHSEED"] == "42"


@pytest.mark.parametrize("cuda,mps,expected", [
    (True, True, "cuda"), (False, True, "mps"), (False, False, "cpu"),
])
def test_device_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert reproducibility.get_device() == torch.device(expected)


def test_build_run_id_uses_existing_configs():
    root = Path(__file__).resolve().parents[1] / "configs"
    config = merge_configs(*(load_yaml(root / name) for name in (
        "tasks/brec.yaml", "models/gin.yaml", "encodings/rwse.yaml",
    )))
    assert logging.build_run_id(config, 2) == "brec_gin_rwse_seed2"


def test_save_run(tmp_path):
    result = {"run_id": "brec_gin_rwse_seed2", "seed": 2, "loss": None}
    path = logging.save_run(result, tmp_path / "results" / "runs")
    assert path.name == "brec_gin_rwse_seed2.json"
    assert json.loads(path.read_text(encoding="utf-8")) == result
    with pytest.raises(FileExistsError):
        logging.save_run(result, path.parent)


@pytest.mark.parametrize("run_id", ["../escape", "a/b", "a\\b", "", "a:b"])
def test_save_run_rejects_unsafe_names(tmp_path, run_id):
    with pytest.raises(ValueError):
        logging.save_run({"run_id": run_id}, tmp_path)


def test_save_run_rejects_nonfinite_metrics(tmp_path):
    with pytest.raises(ValueError):
        logging.save_run({"run_id": "run", "loss": float("nan")}, tmp_path)
    assert not (tmp_path / "run.json").exists()


def test_count_trainable_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    model[0].weight.requires_grad_(False)
    assert profiling.count_parameters(model) == 5  # 2 biases + 2 weights + 1 bias


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "mps"])
def test_timer_synchronizes_around_measurement(monkeypatch, device):
    events = []
    ticks = iter([10.0, 12.5])

    def counter():
        events.append("counter")
        return next(ticks)

    monkeypatch.setattr(profiling.time, "perf_counter", counter)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d: events.append("sync"))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: events.append("sync"))
    with profiling.Timer(device) as timer:
        assert timer.elapsed_sec is None
        events.append("work")
    assert timer.elapsed_sec == 2.5
    assert events == (["counter", "work", "counter"] if device == "cpu"
                      else ["sync", "counter", "work", "sync", "counter"])


def test_timer_propagates_errors():
    with pytest.raises(RuntimeError, match="training failed"):
        with profiling.Timer("cpu"):
            raise RuntimeError("training failed")


def test_cuda_memory_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert profiling.reset_cuda_peak_memory() is None
    assert profiling.peak_cuda_memory_mb() is None


def test_cuda_memory_units_and_reset(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    reset = Mock()
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d: 3 * 1024 ** 2)
    profiling.reset_cuda_peak_memory("cuda:0")
    reset.assert_called_once_with("cuda:0")
    assert profiling.peak_cuda_memory_mb("cuda:0") == 3


def test_memory_and_metadata():
    assert profiling.process_memory_mb() > 0
    metadata = profiling.system_metadata("cpu")
    assert metadata.keys() == {
        "os", "python_version", "torch_version", "torch_geometric_version",
        "device", "accelerator",
    }
    assert metadata["device"] == "cpu"
    assert metadata["accelerator"] is None
    assert metadata["torch_version"] == str(torch.__version__)


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.CalledProcessError(1, "git")])
def test_git_commit_unavailable(monkeypatch, error):
    monkeypatch.setattr(logging.subprocess, "run", Mock(side_effect=error))
    assert logging.get_git_commit() == "unknown"


def test_git_commit_success(monkeypatch):
    monkeypatch.setattr(logging.subprocess, "run", Mock(return_value=Mock(stdout="abc123\n")))
    assert logging.get_git_commit() == "abc123"
