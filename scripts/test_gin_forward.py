"""Run a small, deterministic GIN embedding smoke test on CPU."""

from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from gnn_expressivity.models import GIN
from gnn_expressivity.training.config import load_yaml
from gnn_expressivity.training.reproducibility import set_seed


def main() -> None:
    set_seed(42)
    device = torch.device("cpu")  # This smoke test intentionally requires no accelerator.
    config_path = Path(__file__).resolve().parents[1] / "configs/models/gin.yaml"
    config = load_yaml(config_path)
    graphs = []
    for num_nodes in (1, 3, 4, 6):
        edges = [
            (u, v)
            for node in range(num_nodes - 1)
            for u, v in ((node, node + 1), (node + 1, node))
        ]
        graphs.append(Data(
            x=torch.ones(num_nodes, 1),
            num_nodes=num_nodes,
            edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous(),
        ))
    batch = Batch.from_data_list(graphs).to(device)
    model = GIN.from_config(config, in_dim=1).to(device).eval()
    with torch.inference_mode():
        embeddings = model.encode(batch)
    if embeddings.shape != (len(graphs), config["model"]["hidden_dim"]):
        raise RuntimeError(f"Unexpected embedding shape: {list(embeddings.shape)}")
    if embeddings.device.type != "cpu" or not torch.isfinite(embeddings).all():
        raise RuntimeError("Expected finite CPU embeddings")
    print(f"Batch size: {batch.num_graphs}")
    print(f"Embedding shape: {list(embeddings.shape)}")
    print("GIN forward test passed.")


if __name__ == "__main__":
    main()
