import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest

from gnn_expressivity.analysis.brec_analysis import (
    CATEGORIES,
    ENVIRONMENT,
    RESOURCES,
    aggregate_runs,
    load_full_runs,
    save_summary,
)


@pytest.fixture
def runs():
    config = {
        "model": {"name": "gin", "hidden_dim": 8, "num_layers": 1, "dropout": 0.1,
                  "activation": "relu", "pooling": "sum", "train_eps": True},
        "encoding": {"name": "none"},
        "task": {"name": "brec", "dataset": "BREC", "batch_size": 32, "num_workers": 0},
        "training": {"epochs": 1, "learning_rate": 0.001, "weight_decay": 0.0},
        "evaluation": {"save_embeddings": False},
    }
    result = []
    for seed in range(5):
        def counts(n, distinguished):
            return {"pairs": n, "distinguished": distinguished, "distinction_rate": distinguished / n,
                    "candidate_distinguished": distinguished, "reliability_failures": 0}
        result.append({
            "seed": seed, "run_id": f"brec_gin_none_seed{seed}", "task": "brec", "dataset": "BREC",
            "model": "gin", "encoding": "none", "hidden_dim": 8, "num_layers": 1,
            "evaluated_pairs": 400, "full_benchmark_pairs": 400, "development_subset": False,
            "full_benchmark_completed": True, "benchmark_valid": True, "primary_metric": seed / 400,
            "source_sha256": "a" * 64, "config": copy.deepcopy(config), "protocol": {"threshold": 72.34},
            "overall": counts(400, seed),
            "by_category": {name: counts(n, seed if name == "Basic" else 0) for name, n in CATEGORIES.items()},
            **dict.fromkeys(RESOURCES, 10.0 + seed), "num_parameters": 100, "peak_gpu_memory_mb": None,
            **dict.fromkeys(ENVIRONMENT, "fixture"), "device": "cpu", "accelerator": None, "torch_num_threads": 1,
        })
    return result


def test_mean_sample_std_and_hardware_groups(runs):
    runs[-1]["device"] = "mps"
    summary, by_seed = aggregate_runs(runs)
    row = next(r for r in summary if r["scope"] == "scientific" and r["category"] == "Overall" and r["metric"] == "distinguished")
    assert row["mean"] == 2
    assert row["sample_std"] == pytest.approx(2.5 ** 0.5)
    assert row["min"] == 0 and row["max"] == 4 and row["n"] == 5
    resource = [r for r in summary if r["scope"] == "resources" and r["metric"] == "training_time_sec"]
    assert sorted(r["n"] for r in resource) == [1, 4]
    assert next(r for r in resource if r["n"] == 1)["sample_std"] is None
    assert len(by_seed) == 35
    assert all(r["benchmark_valid"] for r in by_seed)


def test_duplicate_and_missing_seeds(runs):
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_runs([runs[0], runs[0], *runs[2:]])
    with pytest.raises(ValueError, match="exactly seeds"):
        aggregate_runs(runs[:-1])


def test_exclude_unintended_files_and_no_overwrite(tmp_path, runs):
    for run in runs:
        (tmp_path / f"{run['run_id']}.json").write_text(json.dumps(run), encoding="utf-8")
    for filename in ("brec_gin_none_seed42.json", "brec_gin_none_seed0_first20.json", "brec_graphgps_none_seed0.json"):
        (tmp_path / filename).write_text("not even valid JSON", encoding="utf-8")
    loaded = load_full_runs(tmp_path)
    assert [r["seed"] for r in loaded] == list(range(5))
    paths = save_summary(loaded, tmp_path / "summaries")
    before = [p.read_bytes() for p in paths]
    with paths[0].open(newline="", encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == 42
    with pytest.raises(FileExistsError):
        save_summary(loaded, tmp_path / "summaries")
    assert [p.read_bytes() for p in paths] == before


@pytest.mark.parametrize("key,value", [
    ("evaluated_pairs", 20), ("development_subset", True), ("full_benchmark_completed", False),
    ("model", "graphgps"), ("encoding", "rwse"), ("source_sha256", "b" * 64),
    ("benchmark_valid", None),
])
def test_incomplete_and_mismatched_metadata_rejected(runs, key, value):
    runs[2][key] = value
    with pytest.raises(ValueError):
        aggregate_runs(runs)


def test_mismatched_training_config_rejected(runs):
    runs[2]["config"]["training"]["epochs"] = 100
    with pytest.raises(ValueError, match="Mismatched"):
        aggregate_runs(runs)


def test_invalid_benchmark_remains_visible(runs):
    runs[0]["overall"]["reliability_failures"] = 1
    runs[0]["by_category"]["Basic"]["reliability_failures"] = 1
    runs[0]["benchmark_valid"] = False
    runs[0]["primary_metric"] = None
    summary, by_seed = aggregate_runs(runs)
    assert not any(r["all_benchmarks_valid"] for r in summary)
    assert not by_seed[0]["benchmark_valid"]


def test_runner_skips_completed_and_reports_bad_seed(tmp_path, runs, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/run_gin_brec_seeds.py"
    spec = importlib.util.spec_from_file_location("seed_runner", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    for run in runs:
        (tmp_path / f"{run['run_id']}.json").write_text(json.dumps(run), encoding="utf-8")
    def unexpected(*args, **kwargs):
        raise AssertionError("Completed runs must not launch experiments")
    monkeypatch.setattr(runner.subprocess, "run", unexpected)
    assert runner.run_seeds(tmp_path) == []
    corrupt = tmp_path / "brec_gin_none_seed2.json"
    corrupt.write_text("incomplete", encoding="utf-8")
    assert runner.run_seeds(tmp_path) == [2]
    assert corrupt.read_text() == "incomplete"
