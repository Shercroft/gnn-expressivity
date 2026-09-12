import pytest
import torch
from torch_geometric.data import Batch, Data

from gnn_expressivity.encodings import (
    NoStructuralEncoding,
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
