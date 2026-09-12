import torch
from torch_geometric.data import Data

from ._features import append_features, baseline
from .base import GraphEncoding


class UIDEncoding(GraphEncoding):
    """Append normalized current node indices; intentionally label-sensitive."""

    @property
    def name(self) -> str:
        return "uid"

    def encode(self, data: Data) -> Data:
        encoded = baseline(data)
        n = encoded.num_nodes
        values = torch.arange(n, dtype=torch.float64) / max(n - 1, 1)
        return append_features(encoded, values.reshape(-1, 1))

    def output_dim(self, data: Data) -> int:
        return int(baseline(data).x.shape[1]) + 1
