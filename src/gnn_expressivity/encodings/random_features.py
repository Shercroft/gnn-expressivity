import torch
from torch_geometric.data import Data

from ._features import append_features, baseline
from .base import GraphEncoding


class RandomNodeEncoding(GraphEncoding):
    """Append seeded normal vectors tied to current node indices."""

    def __init__(self, dim: int = 8, seed: int = 0) -> None:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.dim = dim
        self.seed = seed

    @property
    def name(self) -> str:
        return "random"

    def encode(self, data: Data) -> Data:
        encoded = baseline(data)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        values = torch.randn(
            (encoded.num_nodes, self.dim), generator=generator, device="cpu",
            dtype=torch.float32,
        )
        return append_features(encoded, values)

    def output_dim(self, data: Data) -> int:
        return int(baseline(data).x.shape[1]) + self.dim
