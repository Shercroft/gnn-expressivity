"""GIN graph encoder using PyG sum-aggregation GINConv layers."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import (
    GINConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


class GIN(nn.Module):
    """Encode a PyG Data or Batch into one vector per graph.

    Each message-passing layer uses a two-linear-layer MLP with ReLU,
    followed by ReLU and dropout. The final node states are pooled (sum by
    default). No normalization, self-loop insertion, or edge features are used.
    GINConv already includes the (1 + epsilon) self contribution.

    Missing x is replaced locally with one constant feature per node, requiring
    in_dim=1. Otherwise x must be floating point with shape [num_nodes, in_dim].
    Embeddings have shape [num_graphs, hidden_dim]. An optional linear head
    changes the output width to out_dim, with no softmax or other output activation.
    Inputs and model must be on the same device; no device is selected implicitly.
    """

    def __init__(
        self,
        in_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
        pooling: str = "sum",
        train_eps: bool = True,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        for name, value in (("in_dim", in_dim), ("hidden_dim", hidden_dim),
                            ("num_layers", num_layers), ("out_dim", out_dim)):
            if name == "out_dim" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if activation != "relu":
            raise ValueError("GIN currently supports activation='relu' only")
        pools = {"sum": global_add_pool, "mean": global_mean_pool, "max": global_max_pool}
        if pooling not in pools:
            raise ValueError("pooling must be 'sum', 'mean', or 'max'")
        if not isinstance(train_eps, bool):
            raise TypeError("train_eps must be a bool")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.pool = pools[pooling]
        self.convs = nn.ModuleList([
            GINConv(
                nn.Sequential(
                    nn.Linear(in_dim if layer == 0 else hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                train_eps=train_eps,
            )
            for layer in range(num_layers)
        ])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Identity() if out_dim is None else nn.Linear(hidden_dim, out_dim)

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, in_dim: int = 1, out_dim: int | None = None
    ) -> "GIN":
        """Use the existing nested model config; feature/output widths are task-specific."""
        settings = dict(config["model"])
        if settings.pop("name") != "gin":
            raise ValueError("Expected model.name=gin")
        return cls(in_dim=in_dim, out_dim=out_dim, **settings)

    def encode(self, data: Data) -> torch.Tensor:
        """Return graph embeddings before the optional task head."""
        if data.num_nodes is None or data.num_nodes <= 0:
            raise ValueError("GIN requires a graph with at least one node")
        edge_index = data.edge_index
        if edge_index is None or edge_index.dtype != torch.long:
            raise ValueError("edge_index must be a torch.long tensor")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        x = data.x
        if x is None:
            if self.in_dim != 1:
                raise ValueError("Featureless graphs require in_dim=1")
            x = torch.ones(
                (data.num_nodes, 1), device=edge_index.device,
                dtype=next(self.parameters()).dtype,
            )
        elif x.ndim != 2 or x.shape != (data.num_nodes, self.in_dim) or not x.is_floating_point():
            raise ValueError("x must be floating point with shape [num_nodes, in_dim]")

        for conv in self.convs:
            x = self.dropout(self.activation(conv(x, edge_index)))
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=x.device)
            num_graphs = 1
        else:
            num_graphs = data.num_graphs
        return self.pool(x, batch, size=num_graphs)

    def forward(self, data: Data) -> torch.Tensor:
        return self.head(self.encode(data))
