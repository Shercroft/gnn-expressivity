import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from gnn_expressivity.training import brec_evaluation as evaluation


@pytest.mark.parametrize("seed", [0, 1, 4, 42])
def test_seed_propagation_and_development_subset(monkeypatch, seed):
    # Mock expensive data/training only: exercise the real evaluator's seed,
    # iteration, statistics, metadata and full/subset validity logic.
    dataset = SimpleNamespace(sha256="a" * 64)

    class Dataset:
        sha256 = dataset.sha256

        def __len__(self):
            return 400

    monkeypatch.setattr(
        evaluation.BRECDataset,
        "from_config",
        lambda config: Dataset(),
    )
    monkeypatch.setattr(
        evaluation,
        "validate_brec",
        lambda data: None,
    )

    seed_call = Mock()
    monkeypatch.setattr(
        evaluation,
        "set_seed",
        seed_call,
    )
    monkeypatch.setattr(
        evaluation,
        "get_device",
        lambda: torch.device("cpu"),
    )

    seen_encoders = []

    def fake_pair_batches(
        data,
        pair_id,
        size,
        device,
        encoder=None,
    ):
        seen_encoders.append(encoder)

        dummy_batch = SimpleNamespace(
            x=torch.ones(
                (1, 1),
                dtype=torch.float32,
            )
        )

        return (
            SimpleNamespace(
                category="Basic",
                family="Basic",
            ),
            [dummy_batch],
            [dummy_batch],
        )

    batch_call = Mock(
        side_effect=fake_pair_batches
    )

    monkeypatch.setattr(
        evaluation,
        "pair_batches",
        batch_call,
    )

    monkeypatch.setattr(
        evaluation,
        "train_pair",
        lambda *args: (0.5, 1),
    )

    monkeypatch.setattr(
        evaluation,
        "embed_batches",
        lambda *args: torch.zeros(
            64,
            16,
        ),
    )

    config = {
        "model": {
            "name": "gin",
            "hidden_dim": 4,
            "num_layers": 1,
        },
        "encoding": {
            "name": "none",
        },
        "task": {
            "name": "brec",
            "dataset": "BREC",
            "batch_size": 32,
        },
    }

    result = evaluation.evaluate_gin_brec(
        config,
        seed=seed,
        max_pairs=2,
        progress=lambda msg: None,
    )

    seed_call.assert_called_once_with(seed)

    assert result["seed"] == seed
    assert result["run_id"] == f"brec_gin_none_seed{seed}"
    assert result["evaluated_pairs"] == 2

    assert batch_call.call_count == 2

    assert len(seen_encoders) == 2
    assert all(
        encoder is not None
        for encoder in seen_encoders
    )
    assert all(
        encoder.name == "none"
        for encoder in seen_encoders
    )

    assert result["development_subset"] is True
    assert result["full_benchmark_completed"] is False
    assert result["benchmark_valid"] is False
    assert result["primary_metric"] is None


def test_single_cli_seed_subset_suffix_and_overwrite_protection(
    tmp_path,
    monkeypatch,
):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/reproduce_gin_brec.py"
    )

    spec = importlib.util.spec_from_file_location(
        "single_brec_cli",
        script,
    )

    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    monkeypatch.setattr(
        "sys.argv",
        [
            str(script),
            "--seed",
            "0",
            "--max-pairs",
            "20",
            "--results-dir",
            str(tmp_path),
        ],
    )

    counts = {
        "pairs": 20,
        "distinguished": 3,
        "distinction_rate": 0.15,
        "reliability_failures": 0,
    }

    evaluate = Mock(
        return_value={
            "seed": 0,
            "run_id": "brec_gin_none_seed0",
            "overall": counts,
            "by_category": {
                "Basic": counts,
            },
            "evaluated_pairs": 20,
            "full_benchmark_pairs": 400,
            "development_subset": True,
            "benchmark_valid": False,
            "embeddings_file": None,
            "evaluation_time_sec": 1.0,
            "process_memory_mb": 10.0,
        }
    )

    monkeypatch.setattr(
        cli,
        "evaluate_gin_brec",
        evaluate,
    )

    cli.main()

    assert evaluate.call_args.kwargs["seed"] == 0
    assert evaluate.call_args.kwargs["max_pairs"] == 20

    assert (
        evaluate.call_args.kwargs[
            "embeddings_path"
        ].name
        == "brec_gin_none_seed0_first20_embeddings.npz"
    )

    path = (
        tmp_path
        / "brec_gin_none_seed0_first20.json"
    )

    original = path.read_bytes()

    assert (
        json.loads(original)["run_id"]
        == "brec_gin_none_seed0_first20"
    )

    with pytest.raises(FileExistsError):
        cli.main()

    assert evaluate.call_count == 1
    assert path.read_bytes() == original
