"""Run one seeded GIN/BREC RPC experiment."""

import argparse
from pathlib import Path

import torch

from gnn_expressivity.training.brec_evaluation import evaluate_gin_brec
from gnn_expressivity.training.config import load_yaml, merge_configs
from gnn_expressivity.training.logging import build_run_id, save_run


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/runs"),
        help="Directory where run JSON and embeddings are saved.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPU threads; small graphs favor one thread.",
    )

    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help=(
            "Evaluate only the first N BREC pairs for development. "
            "If omitted, evaluate the full benchmark."
        ),
    )

    args = parser.parse_args()

    if args.threads <= 0:
        parser.error("--threads must be positive")

    if args.max_pairs is not None and args.max_pairs <= 0:
        parser.error("--max-pairs must be positive")

    torch.set_num_threads(args.threads)

    config = merge_configs(
        *(
            load_yaml(root / "configs" / path)
            for path in (
                "models/gin.yaml",
                "tasks/brec.yaml",
                "encodings/none.yaml",
            )
        )
    )

    # Base run ID produced by the repository logging infrastructure.
    run_id = build_run_id(config, args.seed)

    # Development subsets must not collide with the eventual full 400-pair run.
    if args.max_pairs is not None:
        run_id = f"{run_id}_first{args.max_pairs}"

    result_path = args.results_dir / f"{run_id}.json"

    if result_path.exists():
        raise FileExistsError(
            f"Run already exists: {result_path}\n"
            "Use a different --results-dir or remove the old development result."
        )

    embeddings_path = None
    if config["evaluation"]["save_embeddings"]:
        embeddings_path = args.results_dir / f"{run_id}_embeddings.npz"

        if embeddings_path.exists():
            raise FileExistsError(
                f"Embeddings already exist: {embeddings_path}"
            )

    result = evaluate_gin_brec(
        config,
        seed=args.seed,
        embeddings_path=embeddings_path,
        max_pairs=args.max_pairs,
    )

    # evaluate_gin_brec uses the standard experiment ID internally.
    # Override it here so subset runs receive distinct filenames.
    result["run_id"] = run_id

    path = save_run(result, args.results_dir)

    print()
    print(f"{'Category':<22} {'Pairs':>6} {'Distinguished':>14} {'Rate':>9}")

    for category, row in [
        *result["by_category"].items(),
        ("Overall", result["overall"]),
    ]:
        print(
            f"{category:<22} "
            f"{row['pairs']:>6} "
            f"{row['distinguished']:>14} "
            f"{row['distinction_rate']:>8.2%}"
        )

    print()

    print(
        f"Evaluated pairs: "
        f"{result['evaluated_pairs']}/{result['full_benchmark_pairs']}"
    )

    print(
        f"Development subset: {result['development_subset']}"
    )

    print(
        f"Reliability failures: "
        f"{result['overall']['reliability_failures']}; "
        f"benchmark valid: {result['benchmark_valid']}"
    )

    print(f"Saved: {path}")

    if result["embeddings_file"] is not None:
        print(f"Embeddings: {result['embeddings_file']}")

    print(
        f"Runtime: {result['evaluation_time_sec']:.2f}s; "
        f"RSS: {result['process_memory_mb']:.2f} MiB"
    )


if __name__ == "__main__":
    main()
