import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

from ._features import append_features, baseline
from .base import GraphEncoding


class DegreeEncoding(GraphEncoding):
    """Append raw source-node degree to baseline features."""

    @property
    def name(self) -> str:
        return "degree"

    def encode(self, data: Data) -> Data:
        encoded = baseline(data)
        if encoded.edge_index is None:
            raise ValueError("Degree encoding requires edge_index (possibly empty)")
        counts = degree(
            encoded.edge_index[0], num_nodes=encoded.num_nodes, dtype=torch.long
        )
        return append_features(encoded, counts.reshape(-1, 1))

    def output_dim(self, data: Data) -> int:
        return int(baseline(data).x.shape[1]) + 1
