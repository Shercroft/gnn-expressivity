"""Audit existing WP3.1 artifacts and produce the full none/degree comparison."""
import argparse
from pathlib import Path

from gnn_expressivity.analysis.wp3_1 import audit, save_tables
from gnn_expressivity.training.config import load_yaml, merge_configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = merge_configs(*(load_yaml(root / "configs" / p) for p in
                             ("models/gin.yaml", "tasks/brec.yaml", "encodings/none.yaml")))
    runs, missing = audit(args.results_dir, config)
    print(f"Valid none: {sum(r['encoding']=='none' for r in runs)}; "
          f"valid degree: {sum(r['encoding']=='degree' for r in runs)}; missing: {missing}")
    if args.audit_only:
        return
    if missing:
        raise ValueError(f"Missing full runs: {missing}")
    for path in save_tables(runs, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
