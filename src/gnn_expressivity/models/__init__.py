"""Graph neural network baselines."""

from .gin import GIN
from .graphgps import GraphGPS
from collections.abc import Mapping
from typing import Any
from torch import nn


def build_model(
    config: Mapping[str, Any], *, in_dim: int = 1, out_dim: int | None = None,
) -> nn.Module:
    """Construct a graph model using the existing nested model config."""
    name = config["model"]["name"]
    models = {"gin": GIN, "graphgps": GraphGPS}
    if name not in models:
        raise ValueError(f"Unsupported model: {name}")
    return models[name].from_config(config, in_dim=in_dim, out_dim=out_dim)


__all__ = ["GIN", "GraphGPS", "build_model"]
