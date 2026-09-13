import pytest
import torch
from torch_geometric.data import Batch, Data

from gnn_expressivity.encodings import (
    DegreeEncoding,
    NoStructuralEncoding,
    RandomNodeEncoding,
    UIDEncoding,
    build_encoding,
)


def make_featureless_graph(num_nodes: int = 3) -> Data:
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=torch.long,
    )

    return Data(
        edge_index=edge_index,
        num_nodes=num_nodes,
    )


def test_none_encoding_adds_constant_one_features() -> None:
    data = make_featureless_graph()

    encoder = NoStructuralEncoding()
    encoded = encoder(data)

    assert encoded.x is not None
    assert encoded.x.shape == (3, 1)
    assert torch.equal(encoded.x, torch.ones((3, 1)))


def test_none_encoding_does_not_mutate_input() -> None:
    data = make_featureless_graph()

    original_edge_index = data.edge_index.clone()

    encoder = NoStructuralEncoding()
    encoded = encoder(data)

    assert data.x is None
    assert torch.equal(data.edge_index, original_edge_index)
    assert encoded is not data


def test_none_encoding_preserves_existing_features() -> None:
    x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    data = make_featureless_graph()
    data.x = x

    encoder = NoStructuralEncoding()
    encoded = encoder(data)

    assert torch.equal(encoded.x, x)
    assert encoder.output_dim(data) == 2


def test_none_encoding_output_dim_for_featureless_graph() -> None:
    data = make_featureless_graph()

    encoder = NoStructuralEncoding()

    assert encoder.output_dim(data) == 1


def test_build_encoding_from_full_config() -> None:
    config = {
        "encoding": {
            "name": "none",
        }
    }

    encoder = build_encoding(config)

    assert isinstance(encoder, NoStructuralEncoding)
    assert encoder.name == "none"


def test_build_encoding_rejects_unknown_encoding() -> None:
    config = {
        "encoding": {
            "name": "does-not-exist",
        }
    }

    with pytest.raises(ValueError, match="Unsupported encoding"):
        build_encoding(config)


def test_encoded_graphs_can_be_batched() -> None:
    encoder = NoStructuralEncoding()

    graph_a = encoder(make_featureless_graph(3))
    graph_b = encoder(make_featureless_graph(3))

    batch = Batch.from_data_list([graph_a, graph_b])

    assert batch.x is not None
    assert batch.x.shape == (6, 1)
    assert batch.num_graphs == 2


NEW_ENCODERS = [DegreeEncoding(), UIDEncoding(), RandomNodeEncoding(dim=4, seed=7)]


def asymmetric_graph(existing: bool = False) -> Data:
    # Asymmetric tree: three branches of distinct lengths (1, 2, 3).
    edges = torch.tensor([[0, 0, 2, 0, 4, 5], [1, 2, 3, 4, 5, 6]])
    data = Data(edge_index=torch.cat((edges, edges.flip(0)), dim=1),
                num_nodes=7, label="example", y=torch.tensor([2]))
    if existing:
        data.x = torch.arange(14, dtype=torch.float64).reshape(7, 2)
    return data


def permute_graph(data: Data, order: torch.Tensor) -> Data:
    """order[new_index] = old_index; inverse maps edge endpoints to new IDs."""
    result = data.clone()
    inverse = torch.argsort(order)
    result.edge_index = inverse[data.edge_index]
    if data.x is not None:
        result.x = data.x[order]
    result.num_nodes = data.num_nodes
    return result


@pytest.mark.parametrize("encoder", NEW_ENCODERS)
@pytest.mark.parametrize("existing", [False, True])
def test_augmentation_metadata_nonmutation_and_batching(encoder, existing):
    data = asymmetric_graph(existing)
    before = data.clone()
    encoded = encoder(data)
    base = before.x if existing else torch.ones(7, 1)
    extra = encoder.dim if isinstance(encoder, RandomNodeEncoding) else 1
    assert encoded is not data
    assert torch.equal(encoded.x[:, :base.shape[1]], base)
    assert encoded.x.dtype == base.dtype
    assert encoded.x.device == base.device
    assert encoded.x.shape == (7, base.shape[1] + extra)
    assert encoder.output_dim(data) == encoded.x.shape[1]
    assert torch.isfinite(encoded.x).all()
    assert torch.equal(data.edge_index, before.edge_index)
    assert torch.equal(encoded.edge_index, before.edge_index)
    assert encoded.label == before.label
    assert torch.equal(encoded.y, before.y)
    if existing:
        assert torch.equal(data.x, before.x)
    else:
        assert data.x is None
    other = asymmetric_graph(existing) if existing else make_featureless_graph()
    other.label = "other"
    other.y = torch.tensor([3])
    batch = Batch.from_data_list([encoded, encoder(other)])
    assert batch.num_graphs == 2
    assert batch.x.shape[1] == encoder.output_dim(data)


def test_degree_values_isolates_and_empty_edges():
    data = make_featureless_graph(4)
    assert torch.equal(DegreeEncoding()(data).x,
                       torch.tensor([[1., 1.], [1., 2.], [1., 1.], [1., 0.]]))
    empty = Data(num_nodes=3, edge_index=torch.empty(2, 0, dtype=torch.long))
    assert torch.equal(DegreeEncoding()(empty).x[:, 1], torch.zeros(3))


@pytest.mark.parametrize("n", [0, 1, 7])
def test_uid_values(n):
    data = Data(num_nodes=n, edge_index=torch.empty(2, 0, dtype=torch.long))
    encoder = UIDEncoding()
    actual = encoder(data).x[:, 1]
    torch.testing.assert_close(actual, torch.arange(n) / max(n - 1, 1))
    assert actual.unique().numel() == n
    assert torch.equal(actual, encoder(data).x[:, 1])


@pytest.mark.parametrize("encoder", NEW_ENCODERS)
@pytest.mark.parametrize("existing", [False, True])
def test_permutation_behavior(encoder, existing):
    data = asymmetric_graph(existing)
    before = data.clone()
    order = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    permuted = permute_graph(data, order)
    edges_before = permuted.edge_index.clone()
    original = encoder(data)
    recomputed = encoder(permuted)
    width = 2 if existing else 1
    torch.testing.assert_close(recomputed.x[:, :width], original.x[order, :width])
    if isinstance(encoder, DegreeEncoding):
        torch.testing.assert_close(recomputed.x, original.x[order])
    else:
        assert torch.equal(recomputed.x[:, width:], original.x[:, width:])
        assert not torch.equal(recomputed.x[:, width:], original.x[order, width:])
        if isinstance(encoder, UIDEncoding):
            torch.testing.assert_close(recomputed.x[:, width],
                                       torch.arange(7, dtype=recomputed.x.dtype) / 6)
    assert torch.equal(data.edge_index, before.edge_index)
    assert torch.equal(permuted.edge_index, edges_before)
    assert torch.equal(recomputed.edge_index, edges_before)


def test_random_reproducibility_and_global_rng():
    data = make_featureless_graph()
    encoder = RandomNodeEncoding(dim=8, seed=5)
    first = encoder(data).x
    encoder(make_featureless_graph(4))
    assert torch.equal(first, encoder(data).x)
    assert torch.equal(first, RandomNodeEncoding(dim=8, seed=5)(data).x)
    assert not torch.equal(first[:, 1:], RandomNodeEncoding(dim=8, seed=6)(data).x[:, 1:])
    torch.manual_seed(123)
    expected = torch.rand(10)
    torch.manual_seed(123)
    assert torch.equal(first, encoder(data).x)
    assert torch.equal(expected, torch.rand(10))
    torch.manual_seed(456)
    assert torch.equal(first, encoder(data).x)


@pytest.mark.parametrize("key,values", [
    ("dim", [0, -1, 1.5, "8", True, None]),
    ("seed", [-1, 1.5, "0", True, None]),
])
def test_random_invalid_parameters(key, values):
    for value in values:
        with pytest.raises(ValueError, match=key):
            RandomNodeEncoding(**{key: value})
        with pytest.raises(ValueError, match=key):
            build_encoding({"name": "random", key: value})


@pytest.mark.parametrize("name,cls", [
    ("degree", DegreeEncoding), ("uid", UIDEncoding),
    ("random", RandomNodeEncoding), ("none", NoStructuralEncoding),
])
@pytest.mark.parametrize("full", [False, True])
def test_factory_conditions(name, cls, full):
    section = {"name": name.upper()}
    encoder = build_encoding({"encoding": section} if full else section)
    assert isinstance(encoder, cls)
    assert encoder.name == name
    if name == "random":
        assert (encoder.dim, encoder.seed) == (8, 0)


def test_random_config_parameters():
    encoder = build_encoding({"encoding": {"name": "random", "dim": 3, "seed": 42}})
    assert (encoder.dim, encoder.seed) == (3, 42)


@pytest.mark.parametrize("encoder", NEW_ENCODERS)
@pytest.mark.parametrize("x", [
    torch.ones(7, dtype=torch.float32), torch.ones(7, 2, dtype=torch.long),
    torch.ones(6, 2), torch.ones(7, 0), torch.ones(7, 1, 1),
    torch.full((7, 2), float("nan")), torch.full((7, 2), float("inf")),
])
def test_invalid_baseline_rejected(encoder, x):
    data = asymmetric_graph()
    data.x = x
    for operation in (encoder.encode, encoder.output_dim):
        with pytest.raises(ValueError, match="Baseline x"):
            operation(data)
