from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from gnn_expressivity.encodings import LapPEEncoding, build_encoding
from gnn_expressivity.training.config import load_yaml
from test_encodings import asymmetric_graph, make_featureless_graph, permute_graph


def graph(edges, n):
    return Data(num_nodes=n, edge_index=torch.tensor(edges, dtype=torch.long),
                x=torch.ones(n, 1, dtype=torch.float64))


def assert_up_to_sign(actual, expected):
    error = torch.minimum((actual - expected).norm(dim=0),
                          (actual + expected).norm(dim=0))
    assert torch.all(error < 1e-7)


@pytest.mark.parametrize("full", [False, True])
def test_factory(full):
    section = {"name": "LAPPE", "k": 3, "eigen_tol": 1e-6}
    encoder = build_encoding({"encoding": section} if full else section)
    assert isinstance(encoder, LapPEEncoding)
    assert (encoder.name, encoder.k, encoder.eigen_tol) == ("lappe", 3, 1e-6)
    default = build_encoding({"name": "lappe"})
    assert (default.k, default.eigen_tol) == (8, 1e-8)


def test_config():
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs/encodings/lappe.yaml")
    assert config == {"encoding": {"name": "lappe", "k": 8, "eigen_tol": 1e-8}}
    assert build_encoding(config).output_dim(make_featureless_graph()) == 9


@pytest.mark.parametrize("key,value", [
    *(('k', v) for v in [0, -1, 2.5, '3', None, True]),
    *(('eigen_tol', v) for v in [0, -1, True, None, '1e-8', float('nan'),
                                float('inf'), complex(1, 1)]),
])
def test_invalid_parameters(key, value):
    with pytest.raises(ValueError, match=key):
        LapPEEncoding(**{key: value})
    with pytest.raises(ValueError, match=key):
        build_encoding({"name": "lappe", key: value})


def test_two_node_edge_and_sign_rule():
    data = graph([[0, 1], [1, 0]], 2)
    encoder = LapPEEncoding(1)
    column = encoder(data).x[:, 1:]
    expected = torch.tensor([[1.], [-1.]], dtype=torch.float64) / 2**.5
    assert_up_to_sign(column, expected)
    assert column[column.abs().argmax(), 0] >= 0
    assert torch.equal(column, encoder(data).x[:, 1:])


def test_path_eigenpairs_and_padding():
    data = make_featureless_graph()
    data.x = torch.ones(3, 1, dtype=torch.float64)
    features = LapPEEncoding(5)(data).x[:, 1:]
    q = features[:, :2]
    a = 1 / 2**.5
    laplacian = torch.tensor([[1, -a, 0], [-a, 1, -a], [0, -a, 1]], dtype=torch.float64)
    torch.testing.assert_close(q.T @ q, torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(laplacian @ q, q * torch.tensor([1., 2.]))
    assert torch.count_nonzero(features[:, 2:]) == 0
    # The threshold is applied to eigenvalues, not merely column indices.
    high_tol = LapPEEncoding(2, eigen_tol=1.5)(data).x[:, 1:]
    assert_up_to_sign(high_tol[:, :1], q[:, 1:2])
    assert torch.count_nonzero(high_tol[:, 1:]) == 0


def test_disconnected_nullspace_and_isolate():
    data = graph([[0, 1], [1, 0]], 3)
    features = LapPEEncoding(3)(data).x[:, 1:]
    assert_up_to_sign(features[:, :1],
                      torch.tensor([[1.], [-1.], [0.]], dtype=torch.float64) / 2**.5)
    assert torch.count_nonzero(features[:, 1:]) == 0
    assert torch.count_nonzero(features[2]) == 0


@pytest.mark.parametrize("n", [0, 1, 4])
def test_edgeless(n):
    data = Data(num_nodes=n, edge_index=torch.empty(2, 0, dtype=torch.long))
    encoded = LapPEEncoding(8)(data)
    assert encoded.x.shape == (n, 9)
    assert torch.count_nonzero(encoded.x[:, 1:]) == 0


def test_existing_self_loop():
    assert torch.count_nonzero(LapPEEncoding()(graph([[0], [0]], 1)).x[:, 1:]) == 0


def test_duplicate_weighted_edges():
    data = graph([[0, 0, 1, 1, 1, 2], [1, 1, 0, 0, 2, 1]], 3)
    q = LapPEEncoding(2)(data).x[:, 1:]
    a, b = (2/3)**.5, (1/3)**.5
    laplacian = torch.tensor([[1, -a, 0], [-a, 1, -b], [0, -b, 1]], dtype=torch.float64)
    torch.testing.assert_close(laplacian @ q, q * torch.tensor([1., 2.]))
    torch.testing.assert_close(q.T @ q, torch.eye(2, dtype=torch.float64))


@pytest.mark.parametrize("edges", [[[0], [1]], [[0, 0, 1], [1, 1, 0]]])
def test_asymmetric_adjacency_rejected(edges):
    with pytest.raises(ValueError, match="undirected.*symmetric"):
        LapPEEncoding()(graph(edges, 2))


@pytest.mark.parametrize("existing", [False, True])
def test_simple_spectrum_permutation(existing):
    data = asymmetric_graph(existing)
    if existing:
        data.x = data.x.double()
    original_edges = data.edge_index.clone()
    order = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    permuted = permute_graph(data, order)
    permuted_edges = permuted.edge_index.clone()
    # Check the fixture's selected spectrum really is simple.
    adjacency = torch.zeros(7, 7, dtype=torch.float64)
    adjacency[data.edge_index[0], data.edge_index[1]] = 1
    inverse = adjacency.sum(1).rsqrt()
    laplacian = torch.eye(7, dtype=torch.float64) - inverse[:, None] * adjacency * inverse[None, :]
    eigenvalues = torch.linalg.eigvalsh(laplacian)[1:4]
    assert torch.all(eigenvalues.diff() > 1e-6)
    encoder = LapPEEncoding(3)
    original, actual = encoder(data), encoder(permuted)
    width = 2 if existing else 1
    torch.testing.assert_close(actual.x[:, :width], original.x[order, :width])
    assert_up_to_sign(actual.x[:, width:], original.x[order, width:])
    assert torch.equal(data.edge_index, original_edges)
    assert torch.equal(permuted.edge_index, permuted_edges)
    assert torch.equal(actual.edge_index, permuted_edges)


def test_complete_degenerate_eigenspace_projector():
    # C4 has spectrum 0, 1, 1, 2. Select the entire repeated lambda=1 block.
    data = graph([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], 4)
    order = torch.tensor([2, 0, 3, 1])
    encoder = LapPEEncoding(2)
    q = encoder(data).x[:, 1:]
    actual = encoder(permute_graph(data, order)).x[:, 1:]
    expected = q[order]
    torch.testing.assert_close(actual @ actual.T, expected @ expected.T)
    torch.testing.assert_close(q.T @ q, torch.eye(2, dtype=torch.float64))
    # Individual basis columns and truncation through this block are not canonical.


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_preservation_and_repeatability(dtype):
    data = asymmetric_graph()
    data.x = torch.arange(35, dtype=dtype).reshape(7, 5)
    before = data.clone()
    encoder = LapPEEncoding()
    encoded = encoder(data)
    assert encoded is not data
    assert encoded.x.shape == (7, 13)
    assert encoder.output_dim(data) == 13
    assert encoded.x.dtype == dtype
    assert encoded.x.device == data.x.device
    assert torch.isfinite(encoded.x).all()
    assert torch.equal(encoded.x[:, :5], before.x)
    assert torch.equal(data.x, before.x)
    assert torch.equal(data.edge_index, before.edge_index)
    assert torch.equal(encoded.edge_index, before.edge_index)
    assert encoded.label == before.label
    assert torch.equal(encoded.y, before.y)
    assert torch.equal(encoded.x, encoder(data).x)
    # Check canonicalization before reduced-precision casts can create new ties.
    full_precision = before.clone()
    full_precision.x = full_precision.x.double()
    columns = encoder(full_precision).x[:, 5:11]
    anchors = columns.abs().argmax(0)
    assert torch.all(columns[anchors, torch.arange(6)] >= 0)


def test_batching():
    encoder = LapPEEncoding(4)
    batch = Batch.from_data_list([encoder(make_featureless_graph()),
                                 encoder(make_featureless_graph(4))])
    assert batch.num_graphs == 2
    assert batch.x.shape == (7, 5)


@pytest.mark.parametrize("edges", [None, torch.zeros(3, 2, dtype=torch.long),
    torch.zeros(2, 1), torch.tensor([[-1], [0]]), torch.tensor([[3], [0]])])
def test_invalid_structure(edges):
    with pytest.raises(ValueError, match="edge_index"):
        LapPEEncoding()(Data(num_nodes=3, edge_index=edges))


@pytest.mark.parametrize("x", [torch.ones(3), torch.ones(3, 2, dtype=torch.long),
    torch.ones(2, 2), torch.ones(3, 0), torch.full((3, 2), float('nan')),
    torch.full((3, 2), float('inf'))])
def test_invalid_baseline(x):
    data = make_featureless_graph()
    data.x = x
    for operation in (LapPEEncoding().encode, LapPEEncoding().output_dim):
        with pytest.raises(ValueError, match="Baseline x"):
            operation(data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("edge_device,x_device", [('cpu', 'cuda'), ('cuda', 'cpu')])
def test_mixed_devices(edge_device, x_device):
    data = make_featureless_graph()
    data.x = torch.ones(3, 1, device=x_device)
    data.edge_index = data.edge_index.to(edge_device)
    encoded = LapPEEncoding(2)(data)
    assert encoded.x.device == data.x.device
    assert encoded.edge_index.device == data.edge_index.device
    assert torch.isfinite(encoded.x).all()
