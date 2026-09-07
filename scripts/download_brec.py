"""Download the pinned official BREC data, using the existing task config."""

import argparse
from pathlib import Path

from gnn_expressivity.data.brec import DEFAULT_ROOT, download_brec
from gnn_expressivity.training.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tasks/brec.yaml"))
    parser.add_argument("--root", type=Path, help="Override task.data_root")
    args = parser.parse_args()
    config = load_yaml(args.config)
    root = args.root if args.root is not None else Path(config["task"].get("data_root", DEFAULT_ROOT))
    print(f"Verified BREC: {download_brec(root)}")


if __name__ == "__main__":
    main()
