from __future__ import annotations

import torch
from torch_geometric.data import Data

from .base import GraphEncoding


class NoStructuralEncoding(GraphEncoding):
    """Preserve existing features, or use constant-one features if absent."""

    @property
    def name(self) -> str:
        return "none"

    def encode(self, data: Data) -> Data:
        encoded = data.clone()

        if encoded.x is None:
            num_nodes = encoded.num_nodes
            if num_nodes is None:
                raise ValueError(
                    "Cannot create constant node features when num_nodes is unknown."
                )

            if encoded.edge_index is not None:
                device = encoded.edge_index.device
            else:
                device = torch.device("cpu")

            encoded.x = torch.ones(
                (int(num_nodes), 1),
                dtype=torch.float32,
                device=device,
            )

        return encoded

    def output_dim(self, data: Data) -> int:
        if data.x is None:
            return 1

        if data.x.ndim == 1:
            return 1

        return int(data.x.shape[-1])
