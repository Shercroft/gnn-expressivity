import torch
from torch_geometric.data import Data

from ._features import append_features, baseline
from .base import GraphEncoding


class RWSEEncoding(GraphEncoding):
    """Append return probabilities for steps 1 through walk_length."""

    def __init__(self, walk_length: int = 16) -> None:
        if (
            isinstance(walk_length, bool)
            or not isinstance(walk_length, int)
            or walk_length <= 0
        ):
            raise ValueError("walk_length must be a positive integer")
        self.walk_length = walk_length

    @property
    def name(self) -> str:
        return "rwse"

    def encode(self, data: Data) -> Data:
        edges = data.edge_index
        if not isinstance(edges, torch.Tensor):
            raise ValueError("RWSE requires edge_index (possibly empty)")
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edges.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_index must contain integer node indices")

        encoded = baseline(data)
        n = encoded.num_nodes
        # CPU float64 keeps preprocessing portable, including for MPS inputs.
        sources, destinations = edges.to(device="cpu", dtype=torch.long)
        if edges.numel() and (
            min(sources.min(), destinations.min()) < 0
            or max(sources.max(), destinations.max()) >= n
        ):
            raise ValueError("edge_index node indices must be in [0, num_nodes)")

        transition = torch.zeros((n, n), dtype=torch.float64, device="cpu")
        transition.index_put_(
            (sources, destinations),
            torch.ones(sources.numel(), dtype=torch.float64, device="cpu"),
            accumulate=True,
        )
        # Zero-outdegree rows remain zero; parallel edge counts accumulate.
        transition /= transition.sum(dim=1, keepdim=True).clamp_min(1)
        values = torch.empty((n, self.walk_length), dtype=torch.float64, device="cpu")
        power = transition
        for step in range(self.walk_length):
            values[:, step] = power.diagonal()
            if step + 1 < self.walk_length:
                power = power @ transition
        return append_features(encoded, values)

    def output_dim(self, data: Data) -> int:
        return int(baseline(data).x.shape[1]) + self.walk_length
