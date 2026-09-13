"""Validate and summarize BREC representative pairs without a GPU or training."""

import argparse
import json
from pathlib import Path

from gnn_expressivity.data.brec import BRECDataset, summarize_brec
from gnn_expressivity.training.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tasks/brec.yaml"))
    parser.add_argument("--root", type=Path, help="Override task.data_root")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.root is not None:
        config["task"]["data_root"] = args.root
    dataset = BRECDataset.from_config(config)
    print(json.dumps(summarize_brec(dataset), indent=2))


if __name__ == "__main__":
    main()
