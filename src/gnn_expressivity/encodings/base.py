from __future__ import annotations

from abc import ABC, abstractmethod

from torch_geometric.data import Data


class GraphEncoding(ABC):
    """Common interface for graph node-feature / structural encodings."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable encoding name used in configs and result metadata."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, data: Data) -> Data:
        """Return an encoded graph without mutating the input graph."""
        raise NotImplementedError

    @abstractmethod
    def output_dim(self, data: Data) -> int:
        """Return the node-feature width produced for this graph."""
        raise NotImplementedError

    def __call__(self, data: Data) -> Data:
        return self.encode(data)
