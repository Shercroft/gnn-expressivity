"""Shared baseline validation and feature concatenation."""

import torch
from torch_geometric.data import Data

from .none import NoStructuralEncoding


def baseline(data: Data) -> Data:
    encoded = NoStructuralEncoding()(data)
    x = encoded.x
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise ValueError("Baseline x must be a floating-point tensor")
    if x.ndim != 2 or x.shape[0] != encoded.num_nodes or x.shape[1] == 0:
        raise ValueError("Baseline x must have shape (num_nodes, positive width)")
    if not torch.isfinite(x).all():
        raise ValueError("Baseline x must contain finite values only")
    return encoded


def append_features(encoded: Data, features: torch.Tensor) -> Data:
    features = features.to(device=encoded.x.device, dtype=encoded.x.dtype)
    if not torch.isfinite(features).all():
        raise ValueError("Structural features must contain finite values only")
    encoded.x = torch.cat((encoded.x, features), dim=1)
    return encoded
