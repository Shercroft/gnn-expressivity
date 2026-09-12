from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import GraphEncoding
from .none import NoStructuralEncoding


def build_encoding(config: Mapping[str, Any]) -> GraphEncoding:
    """Build an encoding from either a full config or an encoding section."""

    encoding_config = config.get("encoding", config)

    name = str(encoding_config.get("name", "none")).lower()

    if name == "none":
        return NoStructuralEncoding()

    raise ValueError(f"Unsupported encoding: {name}")


__all__ = [
    "GraphEncoding",
    "NoStructuralEncoding",
    "build_encoding",
]
