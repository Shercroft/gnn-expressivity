"""Aggregate only the five complete GIN/BREC seeds 0 through 4."""

import argparse
from pathlib import Path

from gnn_expressivity.analysis.brec_analysis import load_full_runs, save_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/summaries"))
    args = parser.parse_args()
    runs = load_full_runs(args.results_dir)
    paths = save_summary(runs, args.output_dir)
    for run in runs:
        print(f"Seed {run['seed']}: {run['overall']['distinguished']}/400; "
              f"reliability failures={run['overall']['reliability_failures']}; valid={run['benchmark_valid']}")
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
