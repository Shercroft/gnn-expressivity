"""Run five complete GIN/BREC seeds sequentially, preserving existing results."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from gnn_expressivity.analysis.brec_analysis import validate_full_run

SEEDS = [0, 1, 2, 3, 4]


def run_seeds(results_dir: Path, threads: int = 1) -> list[int]:
    """Use the single-run CLI in sequential child processes for resource isolation.

    That CLI calls evaluate_gin_brec; no protocol implementation is duplicated.
    Invalid existing JSON/orphan embeddings are reported as failures, not overwritten.
    """
    if threads <= 0:
        raise ValueError("threads must be positive")
    completed = 0
    failures = []
    script = Path(__file__).resolve().with_name("reproduce_gin_brec.py")
    for seed in SEEDS:
        path = results_dir / f"brec_gin_none_seed{seed}.json"
        print(f"Seed {seed}; seeds completed {completed}/5", flush=True)
        try:
            if path.exists():
                print(f"Existing JSON found, validating before skip: {path}", flush=True)
            else:
                subprocess.run([sys.executable, "-u", str(script), "--seed", str(seed),
                                "--threads", str(threads), "--results-dir", str(results_dir)], check=True)
            with path.open(encoding="utf-8") as file:
                run = json.load(file)
            validate_full_run(run, seed)
            if run["config"]["evaluation"]["save_embeddings"]:
                embedding_file = run.get("embeddings_file")
                if not embedding_file or not Path(embedding_file).is_file():
                    raise ValueError("Completed run is missing its configured embeddings artifact")
            completed += 1
            print(f"Seed {seed} complete (saved JSON reused if present); seeds completed {completed}/5; "
                  f"distinguished={run['overall']['distinguished']}; "
                  f"reliability failures={run['overall']['reliability_failures']}; "
                  f"runtime={run['evaluation_time_sec']:.2f}s", flush=True)
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            failures.append(seed)
            print(f"FAILED seed {seed}: {error}. Other successful results remain intact.", flush=True)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    failures = run_seeds(args.results_dir, args.threads)
    if failures:
        raise SystemExit(f"Failed seeds: {failures}")
    print("All five seeds complete. Run scripts/aggregate_gin_brec.py to create summaries.")


if __name__ == "__main__":
    main()
