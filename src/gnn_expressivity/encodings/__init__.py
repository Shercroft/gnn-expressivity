from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import GraphEncoding
from .degree import DegreeEncoding
from .none import NoStructuralEncoding
from .random_features import RandomNodeEncoding
from .rwse import RWSEEncoding
from .uid import UIDEncoding


def build_encoding(config: Mapping[str, Any]) -> GraphEncoding:
    """Build an encoding from either a full config or an encoding section."""

    encoding_config = config.get("encoding", config)

    name = str(encoding_config.get("name", "none")).lower()

    if name == "none":
        return NoStructuralEncoding()
    if name == "degree":
        return DegreeEncoding()
    if name == "uid":
        return UIDEncoding()
    if name == "random":
        return RandomNodeEncoding(
            dim=encoding_config.get("dim", 8),
            seed=encoding_config.get("seed", 0),
        )

    if name == "rwse":
        return RWSEEncoding(walk_length=encoding_config.get("walk_length", 16))

    raise ValueError(f"Unsupported encoding: {name}")


__all__ = [
    "GraphEncoding",
    "DegreeEncoding",
    "NoStructuralEncoding",
    "RandomNodeEncoding",
    "RWSEEncoding",
    "UIDEncoding",
    "build_encoding",
]
