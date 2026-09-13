"""Strict aggregation of five full GIN/no-encoding BREC runs.

Only canonical seed 0..4 filenames are read. Scientific statistics include all
five runs with validity flags exposed. Resources are grouped by recorded
hardware/software/thread metadata, never pooled across different environments.
"""

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any

SEEDS = [0, 1, 2, 3, 4]
CATEGORIES = {"Basic": 60, "Regular": 100, "Extension": 100, "CFI": 100,
              "4-Vertex_Condition": 20, "Distance_Regular": 20}
RESOURCES = ["preprocessing_time_sec", "training_time_sec", "inference_time_sec",
             "evaluation_time_sec", "process_memory_mb", "peak_gpu_memory_mb", "num_parameters"]
ENVIRONMENT = ["os", "python_version", "torch_version", "torch_geometric_version",
               "device", "accelerator", "torch_num_threads"]


def validate_full_run(run: Mapping[str, Any], seed: int) -> None:
    """Reject incomplete/wrong-identity records and inconsistent summary counts."""
    if (run.get("seed") != seed or type(run.get("seed")) is not int or seed not in SEEDS
            or run.get("run_id") != f"brec_gin_none_seed{seed}"):
        raise ValueError(f"Unexpected run identity for seed {seed}")
    for key, expected in {"task": "brec", "dataset": "BREC", "model": "gin", "encoding": "none",
                          "evaluated_pairs": 400, "full_benchmark_pairs": 400,
                          "development_subset": False, "full_benchmark_completed": True}.items():
        if run.get(key) != expected or type(run.get(key)) is not type(expected):
            raise ValueError(f"Seed {seed}: invalid {key}")
    if type(run.get("benchmark_valid")) is not bool:
        raise ValueError(f"Seed {seed}: missing benchmark_valid status")
    source = run.get("source_sha256", "")
    if len(source) != 64 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError("Missing or invalid BREC source hash")
    config = run["config"]
    for section, fields in {
        "model": ["name", "hidden_dim", "num_layers", "dropout", "activation", "pooling", "train_eps"],
        "training": ["epochs", "learning_rate", "weight_decay"],
        "task": ["name", "dataset", "batch_size", "num_workers"],
        "encoding": ["name"],
    }.items():
        if any(field not in config[section] for field in fields):
            raise ValueError(f"Incomplete {section} config")
    if (config["model"]["name"] != "gin" or config["encoding"]["name"] != "none"
            or config["model"]["hidden_dim"] != run["hidden_dim"]
            or config["model"]["num_layers"] != run["num_layers"]):
        raise ValueError("Top-level model/encoding metadata disagrees with config")
    if not run.get("protocol"):
        raise ValueError("Missing evaluation protocol")
    if set(run["by_category"]) != set(CATEGORIES):
        raise ValueError("Missing or unexpected categories")
    for name, expected in {"overall": 400, **CATEGORIES}.items():
        row = run["overall"] if name == "overall" else run["by_category"][name]
        if row["pairs"] != expected:
            raise ValueError(f"Incorrect pair count: {name}")
        for key in ("distinguished", "candidate_distinguished", "reliability_failures"):
            if type(row[key]) is not int or not 0 <= row[key] <= expected:
                raise ValueError(f"Invalid {name}/{key}")
        if (row["distinguished"] > row["candidate_distinguished"]
                or row["distinguished"] > expected - row["reliability_failures"]
                or not math.isclose(row["distinction_rate"], row["distinguished"] / expected)):
            raise ValueError(f"Inconsistent distinction results: {name}")
    for key in ("pairs", "distinguished", "candidate_distinguished", "reliability_failures"):
        if sum(row[key] for row in run["by_category"].values()) != run["overall"][key]:
            raise ValueError(f"Category totals disagree for {key}")
    reliable = run["overall"]["reliability_failures"] == 0
    if run["benchmark_valid"] != reliable:
        raise ValueError("Validity disagrees with reliability failures")
    if reliable:
        if run["primary_metric"] != run["overall"]["distinction_rate"]:
            raise ValueError("Primary metric disagrees with distinction rate")
    elif run["primary_metric"] is not None:
        raise ValueError("Invalid benchmark must not have a primary metric")
    for key in RESOURCES:
        value = run[key]
        if value is None and key == "peak_gpu_memory_mb":
            continue
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid resource measurement: {key}")
    for key in ENVIRONMENT:
        if key not in run:
            raise ValueError(f"Missing environment metadata: {key}")


def experiment_signature(run: Mapping[str, Any]) -> dict[str, Any]:
    """Compare model/training settings and protocol, excluding seed/path annotations."""
    config = run["config"]
    return {
        "source_sha256": run["source_sha256"],
        "model": config["model"], "encoding": config["encoding"], "training": config["training"],
        "task": {k: v for k, v in config["task"].items() if k != "data_root"},
        "protocol": {k: v for k, v in run["protocol"].items() if k != "deviations"},
        "num_parameters": run["num_parameters"],
    }


def load_full_runs(results_dir: str | Path) -> list[dict[str, Any]]:
    """Read precisely the five canonical filenames; ignore seed42/firstN/other files."""
    runs = []
    for seed in SEEDS:
        path = Path(results_dir) / f"brec_gin_none_seed{seed}.json"
        with path.open(encoding="utf-8") as file:
            run = json.load(file)
        validate_full_run(run, seed)
        runs.append(run)
    return runs


def aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Return long-form aggregate and per-seed CSV rows; sample std uses ddof=1."""
    seeds = [r["seed"] for r in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seeds")
    if len(runs) != 5 or set(seeds) != set(SEEDS):
        raise ValueError("Expected exactly seeds 0,1,2,3,4")
    runs = sorted(runs, key=lambda r: r["seed"])
    for run in runs:
        validate_full_run(run, run["seed"])
        if experiment_signature(run) != experiment_signature(runs[0]):
            raise ValueError("Mismatched model/config/source/protocol metadata")
    common = {"model": "gin", "encoding": "none", "dataset": "BREC",
              "all_benchmarks_valid": all(r["benchmark_valid"] for r in runs),
              "experiment_json": json.dumps(experiment_signature(runs[0]), sort_keys=True)}
    summary, by_seed = [], []

    def add_stat(scope, category, metric, group, values, hardware=""):
        present = [value for value in values if value is not None]
        summary.append({**common, "scope": scope, "category": category, "metric": metric,
                        "seeds": json.dumps([r["seed"] for r in group]), "n": len(present),
                        "mean": mean(present) if present else None,
                        "sample_std": stdev(present) if len(present) > 1 else None,
                        "min": min(present) if present else None, "max": max(present) if present else None,
                        "hardware_json": hardware})

    for category in ("Overall", *CATEGORIES):
        for metric in ("pairs", "distinguished", "distinction_rate", "candidate_distinguished", "reliability_failures"):
            add_stat("scientific", category, metric, runs,
                     [(r["overall"] if category == "Overall" else r["by_category"][category])[metric] for r in runs])
    environments = defaultdict(list)
    for run in runs:
        hardware = json.dumps({key: run[key] for key in ENVIRONMENT}, sort_keys=True)
        environments[hardware].append(run)
        for category in ("Overall", *CATEGORIES):
            row = run["overall"] if category == "Overall" else run["by_category"][category]
            by_seed.append({**common, "seed": run["seed"], "run_id": run["run_id"],
                            "benchmark_valid": run["benchmark_valid"], "category": category, **row,
                            **{key: run[key] for key in RESOURCES}, "hardware_json": hardware,
                            "git_commit": run.get("git_commit"), "timestamp": run.get("timestamp")})
    for hardware, group in environments.items():
        for metric in RESOURCES:
            add_stat("resources", "Overall", metric, group, [r[metric] for r in group], hardware)
    return summary, by_seed


def save_summary(runs: Sequence[Mapping[str, Any]], output_dir: str | Path) -> tuple[Path, Path]:
    """Validate before writing; refuse to overwrite either existing CSV."""
    summary, by_seed = aggregate_runs(runs)
    directory = Path(output_dir)
    paths = (directory / "gin_brec_summary.csv", directory / "gin_brec_by_seed.csv")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Summary already exists: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    for path, rows in zip(paths, (summary, by_seed)):
        with path.open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return paths
