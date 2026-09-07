"""Single-seed GIN development experiment following the official BREC RPC code.

Reference: GraphPKU/BREC, commit d09e8c349a8bbc0882d2932f7b37b2726f576ce9,
base/test_BREC.py. The statistic deliberately has NO sample-count multiplier
and NO covariance ridge: mean(D).T @ pinv(cov(D)) @ mean(D).
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from gnn_expressivity.data.brec import (
    CATEGORY_RANGES,
    NUM_RELABELINGS,
    SOURCE_COMMIT,
    BRECDataset,
    validate_brec,
)
from gnn_expressivity.models import GIN
from gnn_expressivity.training.logging import build_run_id, get_git_commit
from gnn_expressivity.training.profiling import (
    Timer,
    count_parameters,
    peak_cuda_memory_mb,
    process_memory_mb,
    reset_cuda_peak_memory,
    system_metadata,
)
from gnn_expressivity.training.reproducibility import get_device, set_seed

THRESHOLD = 72.34
ATOL = 1e-6
RTOL = 1e-5  # torch.isclose default used by the reference.
OUTPUT_DIM = 16
LOSS_THRESHOLD = 0.2
PROTOCOL_URL = f"https://github.com/GraphPKU/BREC/blob/{SOURCE_COMMIT}/base/test_BREC.py"


def rpc_statistic(embeddings: torch.Tensor) -> torch.Tensor:
    """Official T-square for 64 interleaved A/B outputs (32 observations).

    Use float32 CPU linear algebra, including for MPS runs (pinv portability).
    Singular covariance uses the reference pseudoinverse; identical embeddings
    produce zero. A constant nonzero difference can also produce zero under
    this protocol, so this is not a generic distance test.
    """
    if embeddings.ndim != 2 or embeddings.shape != (2 * NUM_RELABELINGS, OUTPUT_DIM):
        raise ValueError("RPC requires exactly 64 interleaved 16-dimensional embeddings")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Nonfinite RPC embeddings")
    values = embeddings.detach().to(device="cpu", dtype=torch.float32)
    differences = (values[0::2] - values[1::2]).T
    mean = differences.mean(dim=1, keepdim=True)
    statistic = (mean.T @ torch.linalg.pinv(torch.cov(differences)) @ mean).squeeze()
    if not torch.isfinite(statistic):
        raise ValueError("Nonfinite RPC statistic")
    return statistic


def rpc_decision(test: torch.Tensor, reliability: torch.Tensor) -> dict[str, Any]:
    """Report reference candidate flag and stricter reliability-qualified success.

    Official code counts candidate flags and reliability failures separately.
    Here a pair is distinguished only if both reference checks pass. A run with
    any reliability failure is explicitly invalid for a benchmark score.
    """
    if not torch.isfinite(test).all() or not torch.isfinite(reliability).all():
        raise ValueError("Nonfinite RPC statistic")
    candidate = bool(test > THRESHOLD) and not bool(torch.isclose(test, reliability, atol=ATOL, rtol=RTOL))
    reliable = bool(reliability < THRESHOLD)
    return {
        "t_squared": float(test), "reliability_t_squared": float(reliability),
        "candidate_distinguished": candidate, "reliable": reliable,
        "distinguished": candidate and reliable,
    }


def pair_batches(dataset: BRECDataset, pair_id: int, batch_size: int, device: torch.device):
    """Preserve every official relabeling, A/B order, and reliability offset."""
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("BREC batch_size must be positive and even to preserve A/B pairs")
    representative = dataset[pair_id]
    graphs = []
    for variant in range(NUM_RELABELINGS):
        pair = dataset.get_pair(pair_id, variant)
        expected = (pair_id * 64 + variant * 2, pair_id * 64 + variant * 2 + 1)
        if (pair.pair_id != pair_id or pair.category != representative.category
                or pair.family != representative.family or pair.metadata["raw_indices"] != expected):
            raise ValueError(f"Pair/relabeling metadata misaligned at {pair_id}/{variant}")
        if len(pair.graphs) != 2:
            raise ValueError("Expected both A and B graphs")
        graphs.extend(pair.graphs)
    controls = [dataset.get_reliability_graph(pair_id, i) for i in range(64)]

    def batches(items: list[Data]) -> list[Batch]:
        return [Batch.from_data_list(items[i:i + batch_size]).to(device)
                for i in range(0, len(items), batch_size)]

    return representative, batches(graphs), batches(controls)


def train_pair(model: GIN, batches: Sequence[Batch], config: Mapping[str, Any]) -> tuple[float, int]:
    """Fresh Adam + plateau scheduler; official cosine-margin loss and early stop."""
    settings = config["training"]
    if settings["epochs"] <= 0:
        raise ValueError("training.epochs must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    loss_function = nn.CosineEmbeddingLoss(margin=0.0)
    model.train()
    for epoch in range(settings["epochs"]):
        total = 0.0
        observations = 0
        for batch in batches:
            optimizer.zero_grad()
            prediction = model(batch)
            if prediction.shape != (batch.num_graphs, OUTPUT_DIM) or not torch.isfinite(prediction).all():
                raise ValueError("Missing or nonfinite training embeddings")
            targets = prediction.new_full((batch.num_graphs // 2,), -1)
            loss = loss_function(prediction[0::2], prediction[1::2], targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            observations += len(targets)
            total += len(targets) * loss.item()
        if observations != NUM_RELABELINGS:
            raise ValueError("Training did not process all 32 A/B observations")
        average = total / observations
        if average < LOSS_THRESHOLD:
            break
        scheduler.step(average)
    return average, epoch + 1


def embed_batches(model: GIN, batches: Sequence[Batch]) -> torch.Tensor:
    """Return every graph's 16-D RPC representation, in source order."""
    model.eval()
    with torch.no_grad():
        predictions = []
        for batch in batches:
            prediction = model(batch)
            if prediction.shape != (batch.num_graphs, OUTPUT_DIM):
                raise ValueError("Missing graph embeddings")
            predictions.append(prediction.detach().cpu())
        embeddings = torch.cat(predictions)
    if embeddings.shape != (64, OUTPUT_DIM) or not torch.isfinite(embeddings).all():
        raise ValueError("Expected 64 finite graph embeddings")
    return embeddings


def summarize_results(records: Sequence[Mapping[str, Any]], expected_pairs: int) -> dict[str, Any]:
    if len(records) != expected_pairs or {r["pair_id"] for r in records} != set(range(expected_pairs)):
        raise ValueError("Missing, duplicate, or misaligned evaluated pair IDs")
    for row in records:
        expected = next(name for name, start, stop in CATEGORY_RANGES if start <= row["pair_id"] < stop)
        if row["category"] != expected:
            raise ValueError("Evaluated category metadata misaligned")

    def summary(rows):
        distinguished = sum(r["distinguished"] for r in rows)
        return {"pairs": len(rows), "distinguished": distinguished,
                "distinction_rate": distinguished / len(rows),
                "candidate_distinguished": sum(r["candidate_distinguished"] for r in rows),
                "reliability_failures": sum(not r["reliable"] for r in rows)}

    return {
        "overall": summary(records),
        "by_category": {name: summary([r for r in records if r["category"] == name])
                        for name in Counter(r["category"] for r in records)},
    }


def evaluate_gin_brec(
    config: Mapping[str, Any], *, embeddings_path: Path | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run all official pairs with development seed 42; no multi-seed sweep."""
    seed = 42
    set_seed(seed)
    device = get_device()
    if config["encoding"]["name"] != "none":
        raise ValueError("This baseline requires encoding=none")
    if device.type == "cuda":
        reset_cuda_peak_memory(device)
    preprocessing = training = inference = 0.0
    records, saved_test, saved_controls = [], [], []
    with Timer(device) as total_timer:
        with Timer("cpu") as timer:
            dataset = BRECDataset.from_config(config)
            validate_brec(dataset)
        preprocessing += timer.elapsed_sec
        for pair_id in range(len(dataset)):
            with Timer(device) as timer:
                pair, batches, controls = pair_batches(dataset, pair_id, config["task"]["batch_size"], device)
                model = GIN.from_config(config, in_dim=1, out_dim=OUTPUT_DIM).to(device)
                parameters = count_parameters(model)
            preprocessing += timer.elapsed_sec
            with Timer(device) as timer:
                loss, epochs = train_pair(model, batches, config)
            training += timer.elapsed_sec
            with Timer(device) as timer:
                test_embeddings = embed_batches(model, batches)
                control_embeddings = embed_batches(model, controls)
            inference += timer.elapsed_sec
            decision = rpc_decision(rpc_statistic(test_embeddings), rpc_statistic(control_embeddings))
            records.append({"pair_id": pair_id, "category": pair.category, "family": pair.family,
                            "loss": loss, "epochs": epochs, "graphs_evaluated": 64,
                            "control_graphs_evaluated": 64, **decision})
            if embeddings_path is not None:
                saved_test.append(test_embeddings.numpy())
                saved_controls.append(control_embeddings.numpy())
            if (pair_id + 1) % 10 == 0 or pair_id + 1 == len(dataset):
                progress(f"Evaluated {pair_id + 1}/{len(dataset)} pairs; "
                         f"distinguished={sum(r['distinguished'] for r in records)}, "
                         f"reliability failures={sum(not r['reliable'] for r in records)}")
            del model, batches, controls
    summary = summarize_results(records, len(dataset))
    valid = summary["overall"]["reliability_failures"] == 0
    if embeddings_path is not None:
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        with embeddings_path.open("xb") as file:
            np.savez_compressed(file, pair_ids=np.arange(len(dataset)),
                                categories=np.array([r["category"] for r in records]),
                                test=np.stack(saved_test), reliability=np.stack(saved_controls))
    return {
        "run_id": build_run_id(config, seed), "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(), "seed": seed, "task": config["task"]["name"],
        "dataset": config["task"]["dataset"], "model": "gin", "encoding": "none",
        "hidden_dim": config["model"]["hidden_dim"], "num_layers": config["model"]["num_layers"],
        "primary_metric": summary["overall"]["distinction_rate"] if valid else None,
        "primary_metric_name": "reliability_qualified_distinction_rate",
        "loss": sum(r["loss"] for r in records) / len(records),
        "loss_description": "mean final epoch training loss across independently trained pairs",
        "num_parameters": parameters, "preprocessing_time_sec": preprocessing,
        "training_time_sec": training, "inference_time_sec": inference,
        "evaluation_time_sec": total_timer.elapsed_sec,
        "process_memory_mb": process_memory_mb(),
        "peak_gpu_memory_mb": peak_cuda_memory_mb(device) if device.type == "cuda" else None,
        **system_metadata(device), "torch_num_threads": torch.get_num_threads(),
        "benchmark_valid": valid, "config": dict(config), "source_sha256": dataset.sha256,
        "protocol": {"reference": PROTOCOL_URL, "threshold": THRESHOLD,
                     "atol": ATOL, "rtol": RTOL, "output_dim": OUTPUT_DIM,
                     "relabelings": NUM_RELABELINGS, "loss_margin": 0.0,
                     "loss_threshold": LOSS_THRESHOLD, "statistic_device": "cpu",
                     "statistic_dtype": "float32", "covariance_regularization": None,
                     "sample_count_multiplier": False,
                     "deviations": [
                         "Seed 42 development run only; not a final multi-seed result.",
                         "Training epochs/lr/weight_decay/batch_size use repository config, not reference defaults.",
                         "Existing final-layer GIN encoder plus linear 16-D RPC head; constant-one input features.",
                         "Ordered PyG batches cached per pair; no DataLoader RNG consumption or workers.",
                         "Statistics computed on CPU for CUDA/MPS portability.",
                         "Distinguished requires candidate AND reliability; any failure invalidates primary_metric.",
                     ]},
        **summary, "pairs": records,
        "embeddings_file": str(embeddings_path) if embeddings_path is not None else None,
    }
