import math
from numbers import Real

import torch
from torch_geometric.data import Data

from ._features import append_features, baseline
from .base import GraphEncoding


class LapPEEncoding(GraphEncoding):
    """Append nontrivial symmetric-normalized Laplacian eigenvectors."""

    def __init__(self, k: int = 8, eigen_tol: float = 1e-8) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if (
            isinstance(eigen_tol, bool)
            or not isinstance(eigen_tol, Real)
            or not math.isfinite(eigen_tol)
            or eigen_tol <= 0
        ):
            raise ValueError("eigen_tol must be a finite positive real scalar")
        self.k = k
        self.eigen_tol = float(eigen_tol)

    @property
    def name(self) -> str:
        return "lappe"

    def encode(self, data: Data) -> Data:
        edges = data.edge_index
        if not isinstance(edges, torch.Tensor):
            raise ValueError("LapPE requires edge_index (possibly empty)")
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edges.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_index must contain integer node indices")

        encoded = baseline(data)
        n = encoded.num_nodes
        indices = edges.to(device="cpu", dtype=torch.long)
        if indices.numel() and (indices.min() < 0 or indices.max() >= n):
            raise ValueError("edge_index node indices must be in [0, num_nodes)")
        adjacency = torch.zeros((n, n), dtype=torch.float64, device="cpu")
        adjacency.index_put_(
            (indices[0], indices[1]),
            torch.ones(indices.shape[1], dtype=torch.float64, device="cpu"),
            accumulate=True,
        )
        if not torch.equal(adjacency, adjacency.T):
            raise ValueError("LapPE requires an undirected graph with symmetric adjacency")

        degree = adjacency.sum(dim=1)
        active = degree > 0
        inverse_sqrt = torch.zeros_like(degree)
        inverse_sqrt[active] = degree[active].rsqrt()
        laplacian = (
            torch.diag(active.to(torch.float64))
            - inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
        )
        # Isolated rows/columns are zero, including their diagonal entries.
        values = torch.zeros((n, self.k), dtype=torch.float64, device="cpu")
        if n:
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
            selected = eigenvectors[:, eigenvalues > self.eigen_tol][:, :self.k]
            count = selected.shape[1]
            if count:
                anchors = selected.abs().argmax(dim=0)
                signs = selected[anchors, torch.arange(count)].sign()
                # This fixes signs for this basis, not degenerate eigenspaces.
                values[:, :count] = selected * signs
        return append_features(encoded, values)

    def output_dim(self, data: Data) -> int:
        return int(baseline(data).x.shape[1]) + self.k
