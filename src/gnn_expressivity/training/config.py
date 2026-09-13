from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def merge_configs(*configs: dict) -> dict:
    merged = {}

    for config in configs:
        merged.update(config)

    return merged
