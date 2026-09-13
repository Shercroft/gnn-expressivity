"""GPSConv baseline consuming externally encoded node features."""
from collections.abc import Mapping
from numbers import Real
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, GPSConv, global_add_pool, global_mean_pool, global_max_pool


class GraphGPS(nn.Module):
    """Input projection, GPS layers, graph pooling, and optional task head."""

    name = "graphgps"

    def __init__(
        self, in_dim: int = 1, hidden_dim: int = 128, num_layers: int = 4,
        dropout: float = 0.1, activation: str = "relu", pooling: str = "sum",
        train_eps: bool = True, out_dim: int | None = None, num_heads: int = 4,
    ) -> None:
        super().__init__()
        for name, value in (("in_dim", in_dim), ("hidden_dim", hidden_dim),
                            ("num_layers", num_layers), ("num_heads", num_heads),
                            ("out_dim", out_dim)):
            if name == "out_dim" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if isinstance(dropout, bool) or not isinstance(dropout, Real) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if activation != "relu":
            raise ValueError("GraphGPS currently supports activation='relu' only")
        pools = {"sum": global_add_pool, "mean": global_mean_pool, "max": global_max_pool}
        if pooling not in pools:
            raise ValueError("pooling must be 'sum', 'mean', or 'max'")
        if not isinstance(train_eps, bool):
            raise ValueError("train_eps must be a bool")
        self.in_dim, self.hidden_dim = in_dim, hidden_dim
        self.num_layers, self.num_heads = num_layers, num_heads
        self.dropout, self.pooling = dropout, pooling
        self.pool = pools[pooling]
        self.input_projection = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GPSConv(
                channels=hidden_dim,
                conv=GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ), train_eps=train_eps),
                heads=num_heads, dropout=dropout, act=activation,
                norm="layer_norm", norm_kwargs={"mode": "node"},
                attn_type="multihead", attn_kwargs={"dropout": dropout},
            ) for _ in range(num_layers)
        ])
        self.head = nn.Identity() if out_dim is None else nn.Linear(hidden_dim, out_dim)

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, in_dim: int = 1, out_dim: int | None = None,
    ) -> "GraphGPS":
        settings = dict(config["model"])
        if settings.pop("name") != "graphgps":
            raise ValueError("Expected model.name=graphgps")
        return cls(in_dim=in_dim, out_dim=out_dim, **settings)

    def reset_parameters(self) -> None:
        self.input_projection.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
            # PyTorch MHA._reset_parameters omits out_proj.weight.
            layer.attn.out_proj.reset_parameters()
        if isinstance(self.head, nn.Linear):
            self.head.reset_parameters()

    def encode(self, data: Data) -> torch.Tensor:
        """Return graph embeddings before the optional task head."""
        n = data.num_nodes
        if n is None or n <= 0:
            raise ValueError("GraphGPS requires at least one node")
        x, edges = data.x, data.edge_index
        if (x is None or x.ndim != 2 or x.shape != (n, self.in_dim)
                or not x.is_floating_point() or not torch.isfinite(x).all()):
            raise ValueError("x must be finite floating point with shape [num_nodes, in_dim]")
        if edges is None or edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must be a torch.long tensor with shape [2, E]")
        if edges.numel() and (edges.min() < 0 or edges.max() >= n):
            raise ValueError("edge_index contains invalid node indices")
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(n, dtype=torch.long, device=x.device)
            num_graphs = 1
        else:
            if (batch.dtype != torch.long or batch.shape != (n,) or batch.min() < 0
                    or torch.any(batch[1:] < batch[:-1])):
                raise ValueError("batch must be a sorted nonnegative torch.long node-membership vector")
            num_graphs = int(batch.max()) + 1
            if edges.numel() and torch.any(batch[edges[0]] != batch[edges[1]]):
                raise ValueError("edge_index must not connect different graphs")
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x, edges, batch=batch)
        return self.pool(x, batch, size=num_graphs)

    def forward(self, data: Data) -> torch.Tensor:
        return self.head(self.encode(data))
