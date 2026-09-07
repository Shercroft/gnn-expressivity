"""Run the seed-42 GIN/BREC RPC development experiment."""

import argparse
from pathlib import Path

import torch

from gnn_expressivity.training.brec_evaluation import evaluate_gin_brec
from gnn_expressivity.training.config import load_yaml, merge_configs
from gnn_expressivity.training.logging import build_run_id, save_run


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--threads", type=int, default=1, help="CPU threads; small graphs favor one thread")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    config = merge_configs(*(load_yaml(root / "configs" / path) for path in (
        "models/gin.yaml", "tasks/brec.yaml", "encodings/none.yaml",
    )))
    run_id = build_run_id(config, 42)
    if (args.results_dir / f"{run_id}.json").exists():
        raise FileExistsError("Run already exists; use --results-dir for a separate development run")
    embeddings_path = args.results_dir / f"{run_id}_embeddings.npz" if config["evaluation"]["save_embeddings"] else None
    if embeddings_path is not None and embeddings_path.exists():
        raise FileExistsError(f"Embeddings already exist: {embeddings_path}")
    result = evaluate_gin_brec(config, embeddings_path=embeddings_path)
    path = save_run(result, args.results_dir)
    print(f"{'Category':<22} {'Pairs':>6} {'Distinguished':>14} {'Rate':>9}")
    for category, row in [*result["by_category"].items(), ("Overall", result["overall"])]:
        print(f"{category:<22} {row['pairs']:>6} {row['distinguished']:>14} {row['distinction_rate']:>8.2%}")
    print(f"Reliability failures: {result['overall']['reliability_failures']}; benchmark valid: {result['benchmark_valid']}")
    print(f"Saved: {path}")
    print(f"Runtime: {result['evaluation_time_sec']:.2f}s; RSS: {result['process_memory_mb']:.2f} MiB")


if __name__ == "__main__":
    main()
