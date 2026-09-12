from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from gnn_expressivity.models import GIN
from gnn_expressivity.encodings import RWSEEncoding, build_encoding
from gnn_expressivity.training.brec_evaluation import (
    embed_batches,
    pair_batches,
    rpc_decision,
    rpc_statistic,
    summarize_results,
    train_pair,
)


def test_statistic_matches_reference_without_sample_multiplier():
    values = torch.zeros(64, 16)
    values[0::2, 0] = torch.arange(1, 33)
    torch.testing.assert_close(rpc_statistic(values), torch.tensor(16.5 ** 2 / 88))
    differences = (values[0::2] - values[1::2]).T
    mean = differences.mean(1).reshape(-1, 1)
    reference = (mean.T @ torch.linalg.pinv(torch.cov(differences)) @ mean).squeeze()
    torch.testing.assert_close(rpc_statistic(values), reference)


def test_zero_covariance_follows_reference_pseudoinverse():
    assert rpc_statistic(torch.zeros(64, 16)) == 0
    values = torch.zeros(64, 16)
    values[0::2] = 1
    assert rpc_statistic(values) == 0


@pytest.mark.parametrize("test,control,candidate,reliable", [
    (100, 0, True, True), (0, 0, False, True), (100, 100, False, False),
    (200, 100, True, False), (72.34, 0, False, True), (100, 72.34, True, False),
])
def test_reference_decision_and_reliability_gate(test, control, candidate, reliable):
    result = rpc_decision(torch.tensor(test, dtype=torch.float32), torch.tensor(control, dtype=torch.float32))
    assert result["candidate_distinguished"] == candidate
    assert result["reliable"] == reliable
    assert result["distinguished"] == (candidate and reliable)


@pytest.mark.parametrize("values", [torch.zeros(2, 16), torch.zeros(64, 8), torch.full((64, 16), float("nan")),
                                   torch.full((64, 16), float("inf"))])
def test_invalid_embeddings(values):
    with pytest.raises(ValueError):
        rpc_statistic(values)


class ToyDataset:
    def __getitem__(self, pair_id):
        return self.get_pair(pair_id, 0)

    def get_pair(self, pair_id, variant):
        def graph(nodes):
            return Data(num_nodes=nodes, edge_index=torch.empty(2, 0, dtype=torch.long))
        return SimpleNamespace(pair_id=pair_id, category="Basic", family="Basic",
                               graphs=(graph(2), graph(3)),
                               metadata={"raw_indices": (pair_id * 64 + 2 * variant, pair_id * 64 + 2 * variant + 1)})

    def get_reliability_graph(self, pair_id, variant):
        return self.get_pair(pair_id, 0).graphs[0]


def test_pair_batching_training_and_embedding():
    torch.manual_seed(42)
    pair, batches, controls = pair_batches(ToyDataset(), 0, 10, torch.device("cpu"))
    assert pair.pair_id == 0
    assert sum(batch.num_graphs for batch in batches) == 64
    assert sum(batch.num_graphs for batch in controls) == 64
    assert torch.cat([batch.ptr.diff() for batch in batches]).tolist() == [2, 3] * 32
    model = GIN(hidden_dim=4, num_layers=1, out_dim=16, dropout=0)
    before = model.head.weight.detach().clone()
    loss, epochs = train_pair(model, batches, {"training": {"epochs": 2, "learning_rate": 0.001, "weight_decay": 0.0}})
    assert 1 <= epochs <= 2 and 0 <= loss <= 1
    assert not torch.equal(before, model.head.weight)
    output = embed_batches(model, batches)
    assert output.shape == (64, 16)
    assert not model.training
    assert torch.isfinite(output).all()
    assert rpc_statistic(embed_batches(model, controls)) == 0


def test_odd_batch_size_rejected():
    with pytest.raises(ValueError, match="even"):
        pair_batches(ToyDataset(), 0, 3, torch.device("cpu"))


def test_pair_batches_with_rwse():
    _, batches, controls = pair_batches(
        ToyDataset(), 0, 10, torch.device("cpu"), encoder=RWSEEncoding(4)
    )
    for batch in batches + controls:
        assert batch.x is not None
        assert batch.x.shape == (batch.num_nodes, 5)
        assert batch.x.is_floating_point()
        assert torch.isfinite(batch.x).all()
        assert torch.equal(batch.x[:, 0], torch.ones(batch.num_nodes))
        assert torch.count_nonzero(batch.x[:, 1:]) == 0


@pytest.mark.parametrize("name,width", [("none", 1), ("degree", 2), ("uid", 2), ("random", 9)])
def test_pair_batches_with_encodings(name, width):
    encoder = build_encoding({"name": name})
    _, batches, controls = pair_batches(ToyDataset(), 0, 10, torch.device("cpu"), encoder)
    for batch in batches + controls:
        assert batch.x is not None
        assert batch.x.shape == (batch.num_nodes, width)
        assert batch.x.is_floating_point()
        assert torch.isfinite(batch.x).all()
    if name == "random":
        # Same-size A/B graphs receive identical index-tied draws.
        graphs = controls[0].to_data_list()
        assert torch.equal(graphs[0].x, graphs[1].x)


def test_metadata_misalignment_rejected():
    class Misaligned(ToyDataset):
        def get_pair(self, pair_id, variant):
            pair = super().get_pair(pair_id, variant)
            if variant == 1:
                pair.category = "CFI"
            return pair

    with pytest.raises(ValueError, match="misaligned"):
        pair_batches(Misaligned(), 0, 32, torch.device("cpu"))


def test_aggregation_and_sanity_checks():
    rows = [{"pair_id": i, "category": "Basic", **rpc_decision(torch.tensor(float(i * 100)), torch.tensor(0.0))}
            for i in range(2)]
    summary = summarize_results(rows, 2)
    assert summary["overall"]["distinction_rate"] == 0.5
    assert summary["by_category"]["Basic"]["pairs"] == 2
    with pytest.raises(ValueError, match="pair IDs"):
        summarize_results(rows[:1], 2)
    with pytest.raises(ValueError, match="pair IDs"):
        summarize_results([rows[0], rows[0]], 2)
    with pytest.raises(ValueError, match="category"):
        summarize_results([rows[0], {**rows[1], "category": "CFI"}], 2)
