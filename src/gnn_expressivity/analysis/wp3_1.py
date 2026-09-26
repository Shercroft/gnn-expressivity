"""Audit and compare full GIN none/degree runs without rerunning valid seeds."""

import csv
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from .brec_analysis import ENVIRONMENT, RESOURCES, SEEDS, experiment_signature, validate_full_run


def validate_run(run: dict, encoding: str, seed: int) -> None:
    """Only complete, reliable runs may enter the WP3.1 table."""
    if encoding not in ("none", "degree"):
        raise ValueError("WP3.1 requires none or degree")
    try:
        validate_full_run(run, seed, encoding)
        if run["reliability_valid"] is not True or run["benchmark_valid"] is not True:
            raise ValueError(f"Unreliable full run: {encoding} seed {seed}")
        if run["num_parameters"] <= 0:
            raise ValueError("num_parameters must be positive")
    except KeyError as error:
        raise ValueError(f"Missing required field: {error}") from error


def audit(results_dir: Path, config: dict) -> tuple[list[dict], list[tuple[str, int]]]:
    """Search recursively, validate saved artifacts/configs, and identify missing seeds."""
    found = {}
    for path in sorted(results_dir.rglob("*.json")):
        with path.open(encoding="utf-8") as file:
            run = json.load(file)
        if (run.get("model") != "gin" or run.get("encoding") not in ("none", "degree")
                or run.get("seed") not in SEEDS):
            continue
        if run.get("development_subset") is True and "_first" in path.stem:
            continue
        encoding, seed = run["encoding"], run["seed"]
        validate_run(run, encoding, seed)
        if path.stem != run["run_id"]:
            raise ValueError(f"Filename/seed mismatch: {path}")
        for section in ("model", "task", "training", "evaluation"):
            if run["config"][section] != config[section]:
                raise ValueError(f"Incompatible {section}: {path}")
        if run["config"]["encoding"] != {"name": encoding}:
            raise ValueError(f"Incompatible encoding config: {path}")
        if run["config"]["evaluation"]["save_embeddings"]:
            artifact = Path(run.get("embeddings_file") or "")
            if not artifact.is_file():
                raise ValueError(f"Missing embeddings: {path}")
            with np.load(artifact, allow_pickle=False) as arrays:
                if not np.array_equal(arrays["pair_ids"], np.arange(400)):
                    raise ValueError(f"Misaligned embedding IDs: {artifact}")
                for key in ("test", "reliability"):
                    if arrays[key].shape != (400, 64, 16) or not np.isfinite(arrays[key]).all():
                        raise ValueError(f"Invalid embedding array: {artifact}/{key}")
        key = (encoding, seed)
        if key in found:
            raise ValueError(f"Duplicate full result: {key}")
        found[key] = run | {"result_path": str(path)}
    missing = []
    print("Condition | Seed | Full run found | Valid | Result path", flush=True)
    for encoding in ("none", "degree"):
        for seed in SEEDS:
            run = found.get((encoding, seed))
            print(f"{encoding} | {seed} | {bool(run)} | {bool(run)} | "
                  f"{run['result_path'] if run else '-'}", flush=True)
            if run is None:
                orphan = results_dir / f"brec_gin_{encoding}_seed{seed}_embeddings.npz"
                if orphan.exists():
                    raise ValueError(f"Orphan embeddings; refusing overwrite: {orphan}")
                missing.append((encoding, seed))
    return list(found.values()), missing


def aggregate(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return overall per-seed and summary rows; sample SD uses ddof=1."""
    expected = {(encoding, seed) for encoding in ("none", "degree") for seed in SEEDS}
    if len(runs) != 10 or {(r["encoding"], r["seed"]) for r in runs} != expected:
        raise ValueError("Expected exactly none/degree seeds 0..4, without duplicates")
    reference = None
    by_seed = []
    for run in sorted(runs, key=lambda r: (r["encoding"], r["seed"])):
        validate_run(run, run["encoding"], run["seed"])
        signature = experiment_signature(run)
        signature.pop("encoding")
        signature.pop("num_parameters")
        if reference is not None and signature != reference:
            raise ValueError("Incompatible model/training/source/protocol")
        reference = signature
        by_seed.append({"encoding": run["encoding"], "seed": run["seed"],
            "evaluated_pairs": run["evaluated_pairs"],
            **{key: run["overall"][key] for key in
               ("distinguished", "distinction_rate", "reliability_failures")},
            "reliability_valid": run["reliability_valid"], "benchmark_valid": run["benchmark_valid"],
            **{key: run[key] for key in RESOURCES},
            "environment_json": json.dumps({key: run[key] for key in ENVIRONMENT}, sort_keys=True),
            "result_path": run.get("result_path", ""), "git_commit": run.get("git_commit", "")})
    summary = []
    for encoding in ("none", "degree"):
        group = [r for r in by_seed if r["encoding"] == encoding]
        if len({r["num_parameters"] for r in group}) != 1:
            raise ValueError("Parameter count changed within encoding")
        row = {"encoding": encoding, "seeds": 5, "pairs_per_seed": 400,
               "total_reliability_failures": sum(r["reliability_failures"] for r in group),
               "all_reliability_valid": all(r["reliability_valid"] for r in group),
               "environment_count": len({r["environment_json"] for r in group})}
        for key in ("distinguished", "distinction_rate", *RESOURCES):
            values = [r[key] for r in group if r[key] is not None]
            row[f"{key}_mean"] = mean(values) if values else None
            row[f"{key}_std"] = stdev(values) if len(values) > 1 else None
        row["distinguished_min"] = min(r["distinguished"] for r in group)
        row["distinguished_max"] = max(r["distinguished"] for r in group)
        summary.append(row)
    return by_seed, summary


def save_tables(runs: list[dict], output_dir: Path) -> list[Path]:
    by_seed, summary = aggregate(runs)
    stem = "wp3_1_gin_brec_baseline"
    paths = [output_dir / f"{stem}_{suffix}" for suffix in
             ("by_seed.csv", "summary.csv", "summary.md")]
    if any(p.exists() for p in paths):
        raise FileExistsError("WP3.1 tables already exist; use another output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, rows in zip(paths, (by_seed, summary)):
        with path.open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    text = "| Encoding | Seeds | Distinguished mean ± sd | Rate mean ± sd | Reliability failures | Parameters | Mean eval seconds |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for r in summary:
        text += (f"| {r['encoding']} | 5 | {r['distinguished_mean']:.3f} ± {r['distinguished_std']:.3f} | "
                 f"{r['distinction_rate_mean']:.6f} ± {r['distinction_rate_std']:.6f} | "
                 f"{r['total_reliability_failures']} | {r['num_parameters_mean']:.0f} | {r['evaluation_time_sec_mean']:.2f} |\n")
    text += "\nSD is sample SD (ddof=1); rates are fractions. Resource means describe recorded runs; consult per-seed environment metadata before comparing hardware costs.\n"
    with paths[2].open("x", encoding="utf-8") as file:
        file.write(text)
    return paths
