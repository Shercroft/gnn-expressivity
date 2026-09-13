from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from gnn_expressivity.models import GIN
from gnn_expressivity.training.config import load_yaml
from gnn_expressivity.training.profiling import count_parameters
from gnn_expressivity.training.reproducibility import get_device, set_seed


def path_graph(nodes=4, x=None):
    edges = [(u, v) for i in range(nodes - 1) for u, v in ((i, i + 1), (i + 1, i))]
    return Data(
        num_nodes=nodes, x=x,
        edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous(),
    )


def test_existing_config_and_featureless_graph():
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs/models/gin.yaml")
    model = GIN.from_config(config).eval()
    graph = path_graph()
    result = model(graph)
    assert result.shape == (1, 128)
    assert torch.isfinite(result).all()
    assert len(model.convs) == 4
    assert all(conv.eps.requires_grad for conv in model.convs)
    assert graph.x is None  # Constant features are not written back to the dataset.


def test_encode_before_head_on_selected_device():
    device = get_device()
    model = GIN(in_dim=2, hidden_dim=8, num_layers=2, out_dim=3).to(device).eval()
    graphs = [path_graph(n, torch.ones(n, 2)) for n in (1, 3, 4, 6)]
    batch = Batch.from_data_list(graphs).to(device)
    with torch.inference_mode():
        embeddings = model.encode(batch)
        output = model(batch)
        head_output = model.head(embeddings)
        separate = torch.cat([model.encode(graph.to(device)) for graph in graphs])
    assert embeddings.shape == (4, 8)
    assert output.shape == (4, 3)
    assert embeddings.device.type == device.type
    torch.testing.assert_close(embeddings, separate)
    torch.testing.assert_close(output, head_output)


@pytest.mark.parametrize("train_eps", [True, False])
def test_gin_parameter_count_with_profiler(train_eps):
    model = GIN(in_dim=2, hidden_dim=4, num_layers=2, out_dim=3, train_eps=train_eps)
    # First MLP: 12 + 20; second: 20 + 20; head: 15; one epsilon per conv.
    expected = 87 + (2 if train_eps else 0)
    assert count_parameters(model) == expected
    model.head.requires_grad_(False)
    assert count_parameters(model) == expected - 15


@pytest.mark.parametrize("pooling", ["sum", "mean", "max"])
def test_batch_matches_individual_graphs(pooling):
    model = GIN(hidden_dim=8, num_layers=2, pooling=pooling).eval()
    graphs = [path_graph(), path_graph(1), path_graph(6)]
    with torch.no_grad():
        separate = torch.cat([model(graph) for graph in graphs])
        batched = model(Batch.from_data_list(graphs))
    torch.testing.assert_close(batched, separate)


def test_node_permutation_invariance():
    set_seed(42)
    graph = path_graph(x=torch.randn(4, 3))
    permutation = torch.tensor([2, 0, 3, 1])  # Maps old node IDs to new IDs.
    features = torch.empty_like(graph.x)
    features[permutation] = graph.x
    permuted = Data(num_nodes=4, x=features, edge_index=permutation[graph.edge_index])
    model = GIN(in_dim=3, hidden_dim=8, num_layers=3).eval()
    torch.testing.assert_close(model(graph), model(permuted))


@pytest.mark.parametrize("train_eps", [True, False])
def test_backward_and_task_head(train_eps):
    set_seed(42)
    model = GIN(in_dim=2, hidden_dim=8, num_layers=2, out_dim=3, train_eps=train_eps)
    batch = Batch.from_data_list([path_graph(x=torch.randn(4, 2)), path_graph(1, torch.randn(1, 2))])
    prediction = model(batch)
    assert prediction.shape == (2, 3)
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.convs[0].nn[0].weight.grad.abs().sum() > 0
    assert all(conv.eps.requires_grad == train_eps for conv in model.convs)
    assert model.encode(batch).shape == (2, 8)


def test_seeded_initialization_and_eval_dropout():
    graph = path_graph()
    set_seed(7)
    first = GIN(hidden_dim=8, dropout=0.5).eval()
    set_seed(7)
    second = GIN(hidden_dim=8, dropout=0.5).eval()
    torch.testing.assert_close(first(graph), second(graph))
    torch.testing.assert_close(first(graph), first(graph))


def test_sum_aggregation_and_epsilon():
    model = GIN(hidden_dim=1, num_layers=1, dropout=0).eval()
    # Make the MLP identity on positive values to check the GIN equation exactly.
    with torch.no_grad():
        for layer in (model.convs[0].nn[0], model.convs[0].nn[2]):
            layer.weight.fill_(1)
            layer.bias.zero_()
        model.convs[0].eps.fill_(0.5)
    # Degrees [1, 2, 1] plus self contribution 1.5 each: sum = 8.5.
    torch.testing.assert_close(model(path_graph(3)), torch.tensor([[8.5]]))


def test_regular_graphs_remain_indistinguishable():
    # Six-cycle and two triangles are a known 1-WL limitation with constant x.
    def graph_from_edges(edges):
        return Data(num_nodes=6, edge_index=torch.tensor(
            [(u, v) for a, b in edges for u, v in ((a, b), (b, a))], dtype=torch.long,
        ).t().contiguous())

    cycle = graph_from_edges([(i, (i + 1) % 6) for i in range(6)])
    triangles = graph_from_edges([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    model = GIN(hidden_dim=8).eval()
    torch.testing.assert_close(model(cycle), model(triangles))


@pytest.mark.parametrize("settings", [
    {"num_layers": 0}, {"hidden_dim": -1}, {"in_dim": 0}, {"out_dim": 0},
    {"dropout": 1.0}, {"activation": "unknown"}, {"pooling": "unknown"},
])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        GIN(**settings)


def test_invalid_features():
    with pytest.raises(TypeError, match="train_eps"):
        GIN(train_eps="false")
    with pytest.raises(ValueError, match="in_dim=1"):
        GIN(in_dim=3)(path_graph())
    with pytest.raises(ValueError, match="floating point"):
        GIN()(path_graph(x=torch.ones(4, 1, dtype=torch.long)))


def test_official_brec_forward_if_installed():
    from gnn_expressivity.data.brec import DEFAULT_ROOT, BRECDataset

    root = Path(__file__).resolve().parents[1] / DEFAULT_ROOT
    if not (root / "brec_v3.npy").is_file():
        pytest.skip("Official BREC data not installed")
    pair = BRECDataset(root)[0]
    model = GIN(hidden_dim=8, num_layers=2).eval()
    with torch.no_grad():
        embeddings = model(Batch.from_data_list(list(pair.graphs)))
    assert embeddings.shape == (2, 8)
    assert torch.isfinite(embeddings).all()
