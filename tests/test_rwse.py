from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from gnn_expressivity.encodings import RWSEEncoding, build_encoding
from gnn_expressivity.training.config import load_yaml
from test_encodings import asymmetric_graph, make_featureless_graph, permute_graph


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("length", [None, 1, 4])
def test_factory(full, length):
    section = {"name": "RWSE"}
    if length is not None:
        section["walk_length"] = length
    encoder = build_encoding({"encoding": section} if full else section)
    assert isinstance(encoder, RWSEEncoding)
    assert encoder.name == "rwse"
    assert encoder.walk_length == (16 if length is None else length)


def test_config():
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs/encodings/rwse.yaml")
    assert config == {"encoding": {"name": "rwse", "walk_length": 16}}
    assert build_encoding(config).output_dim(make_featureless_graph()) == 17


@pytest.mark.parametrize("length", [0, -1, 2.5, "3", None, True, False])
def test_invalid_walk_length(length):
    with pytest.raises(ValueError, match="walk_length"):
        RWSEEncoding(length)
    with pytest.raises(ValueError, match="walk_length"):
        build_encoding({"name": "rwse", "walk_length": length})


@pytest.mark.parametrize("edges,n,expected", [
    ([[0, 1, 1, 2], [1, 0, 2, 1]], 3,
     [[0, .5, 0], [0, 1, 0], [0, .5, 0]]),
    ([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], 3,
     [[0, .5, .25]] * 3),
    ([[0, 1, 2], [1, 2, 0]], 3, [[0, 0, 1]] * 3),
    ([[0], [0]], 1, [[1, 1, 1, 1]]),
    ([[], []], 3, [[0, 0, 0, 0]] * 3),
    # Unequal outgoing counts and a sink distinguish outgoing normalization
    # from incoming normalization; duplicate 0->1 entries carry mass 2/3.
    ([[0, 0, 0, 1], [1, 1, 2, 0]], 4,
     [[0, 2/3, 0], [0, 2/3, 0], [0, 0, 0], [0, 0, 0]]),
])
def test_analytical_values(edges, n, expected):
    data = Data(num_nodes=n, edge_index=torch.tensor(edges, dtype=torch.long))
    encoder = RWSEEncoding(len(expected[0]))
    encoded = encoder(data)
    torch.testing.assert_close(encoded.x[:, 1:], torch.tensor(expected, dtype=torch.float32))
    assert torch.equal(encoded.x[:, 0], torch.ones(n))
    assert data.x is None
    assert torch.equal(encoded.edge_index, data.edge_index)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_existing_features_metadata_and_determinism(dtype):
    data = asymmetric_graph()
    data.x = torch.arange(21, dtype=dtype).reshape(7, 3)
    before = data.clone()
    encoder = RWSEEncoding(4)
    encoded = encoder(data)
    assert encoded is not data
    assert encoded.x.shape == (7, 7)
    assert encoder.output_dim(data) == 7
    assert encoded.x.dtype == dtype
    assert encoded.x.device == data.x.device
    assert torch.isfinite(encoded.x).all()
    assert torch.equal(encoded.x[:, :3], before.x)
    assert torch.equal(data.x, before.x)
    assert torch.equal(data.edge_index, before.edge_index)
    assert torch.equal(encoded.edge_index, before.edge_index)
    assert encoded.label == before.label
    assert torch.equal(encoded.y, before.y)
    assert torch.equal(encoded.x, encoder(data).x)


@pytest.mark.parametrize("existing", [False, True])
def test_permutation_equivariance(existing):
    data = asymmetric_graph(existing)
    before = data.clone()
    order = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    permuted = permute_graph(data, order)
    edges_before = permuted.edge_index.clone()
    encoder = RWSEEncoding(4)
    original = encoder(data)
    actual = encoder(permuted)
    width = 2 if existing else 1
    assert original.x[:, width:].unique(dim=0).shape[0] > 1
    torch.testing.assert_close(actual.x, original.x[order])
    assert torch.equal(data.edge_index, before.edge_index)
    assert torch.equal(permuted.edge_index, edges_before)
    assert torch.equal(actual.edge_index, edges_before)


def test_batching_and_zero_nodes():
    encoder = RWSEEncoding(4)
    empty = Data(num_nodes=0, edge_index=torch.empty(2, 0, dtype=torch.long))
    assert encoder(empty).x.shape == (0, 5)
    batch = Batch.from_data_list([encoder(make_featureless_graph(3)),
                                 encoder(make_featureless_graph(4))])
    assert batch.num_graphs == 2
    assert batch.x.shape == (7, 5)
    assert encoder.output_dim(empty) == 5


@pytest.mark.parametrize("edges", [
    None, torch.zeros(3, 2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
    torch.zeros(2, 1), torch.tensor([[0], [-1]]), torch.tensor([[3], [0]]),
])
def test_invalid_edges(edges):
    with pytest.raises(ValueError, match="edge_index"):
        RWSEEncoding()(Data(num_nodes=3, edge_index=edges))


@pytest.mark.parametrize("x", [
    torch.ones(3), torch.ones(3, 2, dtype=torch.long), torch.ones(2, 2),
    torch.ones(3, 0), torch.full((3, 2), float("nan")),
    torch.full((3, 2), float("inf")),
])
def test_invalid_baseline(x):
    data = make_featureless_graph()
    data.x = x
    encoder = RWSEEncoding()
    for operation in (encoder.encode, encoder.output_dim):
        with pytest.raises(ValueError, match="Baseline x"):
            operation(data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("edge_device,x_device", [("cpu", "cuda"), ("cuda", "cpu")])
def test_mixed_devices(edge_device, x_device):
    data = make_featureless_graph()
    data.edge_index = data.edge_index.to(edge_device)
    data.x = torch.ones(3, 2, device=x_device)
    encoded = RWSEEncoding(3)(data)
    assert encoded.x.device == data.x.device
    assert encoded.edge_index.device == data.edge_index.device
    torch.testing.assert_close(encoded.x[:, 2:].cpu(),
                               torch.tensor([[0., .5, 0.], [0., 1., 0.], [0., .5, 0.]]))
