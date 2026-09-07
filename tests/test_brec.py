"""Offline format fixtures plus an optional official-data integration test."""

import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import networkx as nx
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from gnn_expressivity.data import brec
from gnn_expressivity.training.config import load_yaml


@pytest.fixture
def raw_root(tmp_path):
    # Synthetic records exercise layout only; they are not benchmark data.
    a = nx.to_graph6_bytes(nx.path_graph(4), header=False).strip()
    b = nx.to_graph6_bytes(nx.empty_graph(5), header=False).strip()
    records = np.tile(np.array([a, b], dtype="S8"), brec.NUM_RECORDS // 2)
    records[2:4] = [b, a]  # Distinct relabeling to catch ignored variant offsets.
    records[25600:25602] = [b, b]  # Reliability records are separate.
    np.save(tmp_path / "brec_v3.npy", records)
    return tmp_path


def test_pair_layout_metadata_and_cpu(raw_root, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", Mock(side_effect=AssertionError("GPU queried")))
    dataset = brec.BRECDataset(raw_root, verify_checksum=False)
    assert len(dataset) == 400
    pair = dataset[0]
    assert pair.pair_id == 0
    assert pair.category == pair.family == "Basic"
    assert len(pair.graphs) == 2
    assert pair.graph_a.num_nodes == 4
    assert pair.graph_a.num_edges == 6
    assert pair.graph_b.num_nodes == 5  # Isolates survive graph6 -> PyG.
    assert pair.graph_b.edge_index.shape == (2, 0)
    assert pair.graph_a.edge_index.device.type == "cpu"
    assert pair.graph_a.x is None and pair.graph_a.y is None
    assert pair.metadata["raw_indices"] == (0, 1)
    assert pair.metadata["reliability_range"] == (25600, 25664)
    assert dataset.get_pair(0, 1).graph_a.num_nodes == 5
    assert dataset.get_pair(399, 31).metadata["raw_indices"] == (25598, 25599)
    assert dataset.get_reliability_graph(0).num_nodes == 5
    assert dataset[-1].pair_id == 399
    assert [p.pair_id for p in dataset[1:3]] == [1, 2]


def test_validation_categories_and_determinism(raw_root):
    first = brec.BRECDataset(raw_root, verify_checksum=False)
    second = brec.BRECDataset(raw_root, verify_checksum=False)
    summary = brec.summarize_brec(first)
    assert summary["total_pairs"] == 400
    assert summary["total_graphs"] == 800
    assert summary["pairs_per_category"] == {
        "Basic": 60, "Regular": 100, "Extension": 100, "CFI": 100,
        "4-Vertex_Condition": 20, "Distance_Regular": 20,
    }
    assert summary["pairs_per_family"]["Simple Regular"] == 50
    assert summary["pairs_per_family"]["Strongly Regular"] == 50
    assert len({pair.pair_id for pair in first}) == 400
    for a, b in zip(first, second):
        assert (a.pair_id, a.category, a.family, a.metadata) == (
            b.pair_id, b.category, b.family, b.metadata,
        )
        for ga, gb in zip(a.graphs, b.graphs):
            assert ga.num_nodes == gb.num_nodes
            assert torch.equal(ga.edge_index, gb.edge_index)


def test_config_integration(raw_root, monkeypatch):
    monkeypatch.setattr(brec, "DATA_SHA256", brec._sha256(raw_root / "brec_v3.npy"))
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs/tasks/brec.yaml")
    config["task"]["data_root"] = str(raw_root)
    assert len(brec.BRECDataset.from_config(config)) == 400


@pytest.mark.parametrize("pair_id,variant", [(-1, 0), (400, 0), (0, -1), (0, 32)])
def test_invalid_pair_access(raw_root, pair_id, variant):
    dataset = brec.BRECDataset(raw_root, verify_checksum=False)
    with pytest.raises(IndexError):
        dataset.get_pair(pair_id, variant)


def test_missing_and_wrong_checksum(tmp_path, raw_root):
    with pytest.raises(FileNotFoundError, match="download_brec"):
        brec.BRECDataset(tmp_path / "missing")
    with pytest.raises(ValueError, match="SHA-256"):
        brec.BRECDataset(raw_root)


@pytest.mark.parametrize("array", [
    np.array([b"C?"], dtype="S2"),
    np.full((25600, 2), b"C?", dtype="S2"),
    np.full(51200, "C?", dtype="U2"),
    np.array([{"untrusted": "object"}], dtype=object),
])
def test_wrong_array_layout(tmp_path, array):
    np.save(tmp_path / "brec_v3.npy", array)
    with pytest.raises(ValueError):
        brec.BRECDataset(tmp_path, verify_checksum=False)


def test_malformed_graph6(raw_root):
    records = np.load(raw_root / "brec_v3.npy", allow_pickle=False)
    records[0] = b"bad!"
    np.save(raw_root / "brec_v3.npy", records)
    with pytest.raises(ValueError, match="raw record 0"):
        brec.BRECDataset(raw_root, verify_checksum=False)[0]


@pytest.mark.parametrize("nodes,edges", [
    (0, torch.empty((2, 0), dtype=torch.long)),
    (2, torch.tensor([[0, -1], [-1, 0]])),
    (2, torch.tensor([[0, 2], [2, 0]])),
    (2, torch.tensor([[0.0, 1.0], [1.0, 0.0]])),
    (2, torch.tensor([0, 1])),
    (2, torch.tensor([[0], [1]])),
    (2, torch.tensor([[0], [0]])),
    (2, torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])),
])
def test_invalid_graph(nodes, edges):
    with pytest.raises(ValueError):
        brec.validate_graph(Data(num_nodes=nodes, edge_index=edges))


def test_pair_validation_failures(raw_root):
    pairs = list(brec.BRECDataset(raw_root, verify_checksum=False))
    with pytest.raises(ValueError, match="Expected 400"):
        brec.validate_brec(pairs[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        brec.validate_brec([pairs[0], pairs[0], *pairs[2:]])
    with pytest.raises(ValueError, match="category/family"):
        brec.validate_brec([replace(pairs[0], category="wrong"), *pairs[1:]])
    with pytest.raises(ValueError, match="exactly two"):
        brec.validate_brec([SimpleNamespace(pair_id=0, graphs=(pairs[0].graph_a,)), *pairs[1:]])


def test_download_verification_and_reuse(tmp_path, monkeypatch):
    raw = b"fixture payload for download only"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("brec_v3.npy", raw)
        archive.writestr("../unwanted.txt", b"must never be extracted")
    payload = buffer.getvalue()
    monkeypatch.setattr(brec, "DATA_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(brec, "ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
    request = Mock(return_value=io.BytesIO(payload))
    monkeypatch.setattr(brec, "urlopen", request)
    root = tmp_path / "raw"
    path = brec.download_brec(root)
    assert path.read_bytes() == raw
    assert not (tmp_path / "unwanted.txt").exists()
    manifest = json.loads((root / "source.json").read_text())
    assert manifest["source_commit"] == brec.SOURCE_COMMIT
    assert manifest["data_sha256"] == brec.DATA_SHA256
    assert brec.download_brec(root) == path
    request.assert_called_once_with(brec.SOURCE_URL, timeout=60)


def test_corrupt_download_is_not_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(brec, "urlopen", Mock(return_value=io.BytesIO(b"corrupt")))
    with pytest.raises(ValueError, match="archive SHA-256"):
        brec.download_brec(tmp_path)
    assert not (tmp_path / "brec_v3.npy").exists()
    assert not (tmp_path / "BREC_data_all.zip").exists()


def test_official_brec_if_installed():
    root = Path(__file__).resolve().parents[1] / brec.DEFAULT_ROOT
    if not (root / "brec_v3.npy").is_file():
        pytest.skip("Official BREC not installed; run scripts/download_brec.py")
    dataset = brec.BRECDataset(root)
    summary = brec.summarize_brec(dataset)
    assert summary["total_pairs"] == 400
    assert summary["total_graphs"] == 800
    assert summary["pairs_per_category"] == {
        name: stop - start for name, start, stop in brec.CATEGORY_RANGES
    }
    for pair_id in (0, 60, 110, 160, 260, 360, 380, 399):
        a, b = dataset[pair_id], dataset[pair_id]
        for first, second in zip(a.graphs, b.graphs):
            assert torch.equal(first.edge_index, second.edge_index)
        brec.validate_graph(dataset.get_pair(pair_id, 31).graph_a)
        brec.validate_graph(dataset.get_reliability_graph(pair_id, 63))
