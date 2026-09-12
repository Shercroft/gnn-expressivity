"""GIN/BREC experiment following the official BREC RPC evaluation code.

Reference:
GraphPKU/BREC, commit d09e8c349a8bbc0882d2932f7b37b2726f576ce9,
base/test_BREC.py.

The RPC statistic deliberately has NO sample-count multiplier
and NO covariance ridge:

    mean(D).T @ pinv(cov(D)) @ mean(D)

The evaluator supports an optional ``max_pairs`` argument for
small development runs. Partial runs are useful for validating
the pipeline but are not considered full benchmark results.
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
from gnn_expressivity.encodings import (
    GraphEncoding,
    build_encoding,
)
from gnn_expressivity.models import GIN
from gnn_expressivity.training.logging import (
    build_run_id,
    get_git_commit,
)
from gnn_expressivity.training.profiling import (
    Timer,
    count_parameters,
    peak_cuda_memory_mb,
    process_memory_mb,
    reset_cuda_peak_memory,
    system_metadata,
)
from gnn_expressivity.training.reproducibility import (
    get_device,
    set_seed,
)


THRESHOLD = 72.34

ATOL = 1e-6

# torch.isclose default used by the reference implementation.
RTOL = 1e-5

OUTPUT_DIM = 16

LOSS_THRESHOLD = 0.2

PROTOCOL_URL = (
    f"https://github.com/GraphPKU/BREC/blob/"
    f"{SOURCE_COMMIT}/base/test_BREC.py"
)


def rpc_statistic(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute the official RPC T-square statistic.

    BREC supplies 32 relabelings of graph A and graph B.
    The embedding tensor therefore contains 64 interleaved rows:

        A_1, B_1, A_2, B_2, ..., A_32, B_32

    The reference statistic is:

        mean(D)^T pinv(cov(D)) mean(D)

    where D contains the A-B embedding differences.

    Float32 CPU linear algebra is used even when model inference
    runs on another accelerator. This improves portability,
    especially for MPS.

    Singular covariance matrices are handled using the
    pseudoinverse, matching the reference protocol.
    """

    expected_shape = (
        2 * NUM_RELABELINGS,
        OUTPUT_DIM,
    )

    if embeddings.ndim != 2 or embeddings.shape != expected_shape:
        raise ValueError(
            "RPC requires exactly "
            f"{2 * NUM_RELABELINGS} interleaved "
            f"{OUTPUT_DIM}-dimensional embeddings"
        )

    if not torch.isfinite(embeddings).all():
        raise ValueError("Nonfinite RPC embeddings")

    values = embeddings.detach().to(
        device="cpu",
        dtype=torch.float32,
    )

    differences = (
        values[0::2] - values[1::2]
    ).T

    mean = differences.mean(
        dim=1,
        keepdim=True,
    )

    covariance = torch.cov(differences)

    statistic = (
        mean.T
        @ torch.linalg.pinv(covariance)
        @ mean
    ).squeeze()

    if not torch.isfinite(statistic):
        raise ValueError("Nonfinite RPC statistic")

    return statistic


def rpc_decision(
    test: torch.Tensor,
    reliability: torch.Tensor,
) -> dict[str, Any]:
    """Apply the BREC RPC decision rules.

    The official implementation reports the candidate distinction
    test and reliability test separately.

    Here, a graph pair is counted as successfully distinguished only
    when:

    1. the test statistic exceeds the official threshold;
    2. the test statistic is not numerically equal to the reliability
       statistic; and
    3. the reliability statistic remains below the threshold.

    A full benchmark containing any reliability failure is considered
    invalid for the primary benchmark score.
    """

    if (
        not torch.isfinite(test).all()
        or not torch.isfinite(reliability).all()
    ):
        raise ValueError("Nonfinite RPC statistic")

    candidate = (
        bool(test > THRESHOLD)
        and not bool(
            torch.isclose(
                test,
                reliability,
                atol=ATOL,
                rtol=RTOL,
            )
        )
    )

    reliable = bool(
        reliability < THRESHOLD
    )

    return {
        "t_squared": float(test),
        "reliability_t_squared": float(reliability),
        "candidate_distinguished": candidate,
        "reliable": reliable,
        "distinguished": candidate and reliable,
    }


def pair_batches(
    dataset: BRECDataset,
    pair_id: int,
    batch_size: int,
    device: torch.device,
    encoder: GraphEncoding | None = None,
) -> tuple[
    Any,
    list[Batch],
    list[Batch],
]:
    """Construct BREC test and reliability batches for one pair.

    Every official relabeling is preserved, as is the A/B ordering
    required by the RPC protocol.

    ``encoder`` is optional for backward compatibility. When omitted,
    the existing ``none`` encoding is used, which supplies constant-one
    node features to featureless graphs.
    """

    if encoder is None:
        encoder = build_encoding(
            {
                "encoding": {
                    "name": "none",
                }
            }
        )

    if batch_size <= 0 or batch_size % 2:
        raise ValueError(
            "BREC batch_size must be positive and even "
            "to preserve A/B pairs"
        )

    representative = dataset[pair_id]

    graphs: list[Data] = []

    for variant in range(NUM_RELABELINGS):
        pair = dataset.get_pair(
            pair_id,
            variant,
        )

        expected = (
            pair_id * 64 + variant * 2,
            pair_id * 64 + variant * 2 + 1,
        )

        if (
            pair.pair_id != pair_id
            or pair.category != representative.category
            or pair.family != representative.family
            or pair.metadata["raw_indices"] != expected
        ):
            raise ValueError(
                "Pair/relabeling metadata misaligned "
                f"at {pair_id}/{variant}"
            )

        if len(pair.graphs) != 2:
            raise ValueError(
                "Expected both A and B graphs"
            )

        graphs.extend(pair.graphs)

    controls = [
        dataset.get_reliability_graph(pair_id, i)
        for i in range(64)
    ]

    encoded_graphs = [encoder(graph) for graph in graphs]
    encoded_controls = [encoder(graph) for graph in controls]

    all_encoded = encoded_graphs + encoded_controls

    feature_dims: set[int] = set()

    for graph in all_encoded:
        if graph.x is None:
            raise ValueError(
                f"Encoding '{encoder.name}' produced missing node features"
            )

        if graph.x.ndim != 2:
            raise ValueError(
                f"Encoding '{encoder.name}' must produce 2-D node features"
            )

        if not graph.x.is_floating_point():
            raise ValueError(
                f"Encoding '{encoder.name}' must produce floating-point features"
            )

        if graph.x.shape[0] != graph.num_nodes:
            raise ValueError(
                f"Encoding '{encoder.name}' produced the wrong number of node features"
            )

        if not torch.isfinite(graph.x).all():
            raise ValueError(
                f"Encoding '{encoder.name}' produced non-finite node features"
            )

        feature_dims.add(int(graph.x.shape[1]))

    if len(feature_dims) != 1:
        raise ValueError(
            f"Encoding '{encoder.name}' produced inconsistent feature dimensions"
        )

    feature_dims.pop()

    def batches(
        items: list[Data],
    ) -> list[Batch]:
        return [
            Batch.from_data_list(
                items[i : i + batch_size]
            ).to(device)
            for i in range(
                0,
                len(items),
                batch_size,
            )
        ]

    return (
        representative,
        batches(encoded_graphs),
        batches(encoded_controls),
    )


def train_pair(
    model: GIN,
    batches: Sequence[Batch],
    config: Mapping[str, Any],
) -> tuple[float, int]:
    """Train GIN independently on one BREC graph pair.

    This follows the BREC-style cosine-margin objective and early
    stopping structure.

    Each graph pair receives a fresh model and optimizer.
    """

    settings = config["training"]

    if settings["epochs"] <= 0:
        raise ValueError(
            "training.epochs must be positive"
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer
        )
    )

    loss_function = nn.CosineEmbeddingLoss(
        margin=0.0
    )

    model.train()

    average = float("nan")

    for epoch in range(settings["epochs"]):
        total = 0.0
        observations = 0

        for batch in batches:
            optimizer.zero_grad()

            prediction = model(batch)

            expected_shape = (
                batch.num_graphs,
                OUTPUT_DIM,
            )

            if (
                prediction.shape != expected_shape
                or not torch.isfinite(prediction).all()
            ):
                raise ValueError(
                    "Missing or nonfinite training embeddings"
                )

            targets = prediction.new_full(
                (batch.num_graphs // 2,),
                -1,
            )

            loss = loss_function(
                prediction[0::2],
                prediction[1::2],
                targets,
            )

            if not torch.isfinite(loss):
                raise ValueError(
                    "Nonfinite training loss"
                )

            loss.backward()
            optimizer.step()

            observations += len(targets)
            total += len(targets) * loss.item()

        if observations != NUM_RELABELINGS:
            raise ValueError(
                "Training did not process all "
                f"{NUM_RELABELINGS} A/B observations"
            )

        average = total / observations

        if average < LOSS_THRESHOLD:
            break

        scheduler.step(average)

    return average, epoch + 1


def embed_batches(
    model: GIN,
    batches: Sequence[Batch],
) -> torch.Tensor:
    """Return all graph-level 16-D RPC representations."""

    model.eval()

    predictions: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in batches:
            prediction = model(batch)

            expected_shape = (
                batch.num_graphs,
                OUTPUT_DIM,
            )

            if prediction.shape != expected_shape:
                raise ValueError(
                    "Missing graph embeddings"
                )

            if not torch.isfinite(prediction).all():
                raise ValueError(
                    "Nonfinite graph embeddings"
                )

            predictions.append(
                prediction.detach().cpu()
            )

    embeddings = torch.cat(
        predictions,
        dim=0,
    )

    expected_shape = (
        2 * NUM_RELABELINGS,
        OUTPUT_DIM,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            "Expected "
            f"{2 * NUM_RELABELINGS} graph embeddings"
        )

    if not torch.isfinite(embeddings).all():
        raise ValueError(
            "Expected finite graph embeddings"
        )

    return embeddings


def summarize_results(
    records: Sequence[Mapping[str, Any]],
    expected_pairs: int,
) -> dict[str, Any]:
    """Summarize completed BREC pair evaluations.

    ``expected_pairs`` is the number of graph pairs expected in the
    current run.

    For the full benchmark this is the complete BREC dataset size.
    For a development subset, such as ``--max-pairs 20``, it is the
    number of pairs intentionally evaluated.

    The evaluated pair IDs must form the contiguous prefix

        0, 1, ..., expected_pairs - 1

    with no missing or duplicate entries.
    """

    if (
        len(records) != expected_pairs
        or {
            record["pair_id"]
            for record in records
        }
        != set(range(expected_pairs))
    ):
        raise ValueError(
            "Missing, duplicate, or misaligned evaluated pair IDs"
        )

    for row in records:
        expected_category = next(
            name
            for name, start, stop in CATEGORY_RANGES
            if start <= row["pair_id"] < stop
        )

        if row["category"] != expected_category:
            raise ValueError(
                "Evaluated category metadata misaligned"
            )

    def summary(
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not rows:
            raise ValueError(
                "Cannot summarize an empty category"
            )

        distinguished = sum(
            bool(row["distinguished"])
            for row in rows
        )

        candidate_distinguished = sum(
            bool(row["candidate_distinguished"])
            for row in rows
        )

        reliability_failures = sum(
            not bool(row["reliable"])
            for row in rows
        )

        return {
            "pairs": len(rows),
            "distinguished": distinguished,
            "distinction_rate": (
                distinguished / len(rows)
            ),
            "candidate_distinguished": (
                candidate_distinguished
            ),
            "reliability_failures": (
                reliability_failures
            ),
        }

    categories = Counter(
        row["category"]
        for row in records
    )

    by_category = {
        name: summary(
            [
                row
                for row in records
                if row["category"] == name
            ]
        )
        for name in categories
    }

    return {
        "overall": summary(records),
        "by_category": by_category,
    }

    def summary(
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not rows:
            raise ValueError(
                "Cannot summarize an empty category"
            )

        distinguished = sum(
            bool(row["distinguished"])
            for row in rows
        )

        candidate_distinguished = sum(
            bool(row["candidate_distinguished"])
            for row in rows
        )

        reliability_failures = sum(
            not bool(row["reliable"])
            for row in rows
        )

        return {
            "pairs": len(rows),
            "distinguished": distinguished,
            "distinction_rate": (
                distinguished / len(rows)
            ),
            "candidate_distinguished": (
                candidate_distinguished
            ),
            "reliability_failures": (
                reliability_failures
            ),
        }

    categories = Counter(
        row["category"]
        for row in records
    )

    by_category = {
        name: summary(
            [
                row
                for row in records
                if row["category"] == name
            ]
        )
        for name in categories
    }

    return {
        "overall": summary(records),
        "by_category": by_category,
    }


def evaluate_gin_brec(
    config: Mapping[str, Any],
    *,
    seed: int = 42,
    embeddings_path: Path | None = None,
    progress: Callable[[str], None] = print,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    """Run one seeded GIN/BREC experiment.

    Parameters
    ----------
    config:
        Merged repository configuration.

    seed:
        Random seed for initialization and training (default 42).

    embeddings_path:
        Optional destination for compressed test and reliability
        embeddings.

    progress:
        Callback used for progress messages.

    max_pairs:
        Optional number of BREC pairs to evaluate.

        ``None`` means evaluate the complete benchmark.

        A smaller value is intended only for development/testing.
        Partial runs are explicitly marked and do not receive a
        valid full-benchmark primary metric.
    """

    set_seed(seed)

    device = get_device()

    if device.type == "cuda":
        reset_cuda_peak_memory(device)

    preprocessing = 0.0
    training = 0.0
    inference = 0.0

    records: list[dict[str, Any]] = []
    saved_test: list[np.ndarray] = []
    saved_controls: list[np.ndarray] = []

    parameters: int | None = None

    with Timer(device) as total_timer:

        # ---------------------------------------------------------
        # Dataset loading and validation
        # ---------------------------------------------------------

        with Timer("cpu") as timer:
            dataset = BRECDataset.from_config(
                config
            )

            validate_brec(dataset)

            encoder = build_encoding(config)

        preprocessing += timer.elapsed_sec

        full_pair_count = len(dataset)

        if max_pairs is None:
            pair_count = full_pair_count
        else:
            if max_pairs <= 0:
                raise ValueError(
                    "max_pairs must be positive"
                )

            if max_pairs > full_pair_count:
                raise ValueError(
                    "max_pairs cannot exceed BREC "
                    f"dataset size ({full_pair_count})"
                )

            pair_count = max_pairs

        development_subset = (
            pair_count < full_pair_count
        )

        # ---------------------------------------------------------
        # Pair-wise BREC evaluation
        # ---------------------------------------------------------

        for pair_id in range(pair_count):

            # -----------------------------------------------------
            # Pair preparation + fresh model initialization
            # -----------------------------------------------------

            with Timer(device) as timer:
                pair, batches, controls = (
                    pair_batches(
                        dataset,
                        pair_id,
                        config["task"]["batch_size"],
                        device,
                        encoder,
                    )
                )

                if not batches or batches[0].x is None:
                    raise ValueError(
                        f"Encoding '{encoder.name}' produced no batched node features"
                    )

                in_dim = int(batches[0].x.shape[1])

                model = GIN.from_config(
                    config,
                    in_dim=in_dim,
                    out_dim=OUTPUT_DIM,
                ).to(device)

                current_parameters = (
                    count_parameters(model)
                )

                if parameters is None:
                    parameters = current_parameters
                elif parameters != current_parameters:
                    raise ValueError(
                        "GIN parameter count changed "
                        "between BREC pairs"
                    )

            preprocessing += timer.elapsed_sec

            # -----------------------------------------------------
            # Training
            # -----------------------------------------------------

            with Timer(device) as timer:
                loss, epochs = train_pair(
                    model,
                    batches,
                    config,
                )

            training += timer.elapsed_sec

            # -----------------------------------------------------
            # Inference
            # -----------------------------------------------------

            with Timer(device) as timer:
                test_embeddings = embed_batches(
                    model,
                    batches,
                )

                control_embeddings = embed_batches(
                    model,
                    controls,
                )

            inference += timer.elapsed_sec

            # -----------------------------------------------------
            # Official-style RPC statistics
            # -----------------------------------------------------

            test_statistic = rpc_statistic(
                test_embeddings
            )

            reliability_statistic = rpc_statistic(
                control_embeddings
            )

            decision = rpc_decision(
                test_statistic,
                reliability_statistic,
            )

            records.append(
                {
                    "pair_id": pair_id,
                    "category": pair.category,
                    "family": pair.family,
                    "loss": loss,
                    "epochs": epochs,
                    "graphs_evaluated": (
                        2 * NUM_RELABELINGS
                    ),
                    "control_graphs_evaluated": (
                        2 * NUM_RELABELINGS
                    ),
                    **decision,
                }
            )

            # -----------------------------------------------------
            # Optional embedding collection
            # -----------------------------------------------------

            if embeddings_path is not None:
                saved_test.append(
                    test_embeddings.numpy()
                )

                saved_controls.append(
                    control_embeddings.numpy()
                )

            # -----------------------------------------------------
            # Progress output
            # -----------------------------------------------------

            completed = pair_id + 1

            if (
                completed % 10 == 0
                or completed == pair_count
            ):
                distinguished_so_far = sum(
                    bool(record["distinguished"])
                    for record in records
                )

                reliability_failures_so_far = sum(
                    not bool(record["reliable"])
                    for record in records
                )

                progress(
                    f"Evaluated "
                    f"{completed}/{pair_count} pairs; "
                    f"distinguished="
                    f"{distinguished_so_far}, "
                    f"reliability failures="
                    f"{reliability_failures_so_far}"
                )

            del model
            del batches
            del controls

    # -------------------------------------------------------------
    # Aggregate completed run
    # -------------------------------------------------------------

    summary = summarize_results(
        records,
        pair_count,
    )

    reliability_valid = (
        summary["overall"]["reliability_failures"]
        == 0
    )

    full_benchmark_completed = (
        pair_count == full_pair_count
    )

    # A subset may be perfectly reliable, but it is not a completed
    # BREC benchmark and therefore must not be reported as a valid
    # benchmark score.
    benchmark_valid = (
        full_benchmark_completed
        and reliability_valid
    )

    # -------------------------------------------------------------
    # Save embeddings
    # -------------------------------------------------------------

    if embeddings_path is not None:
        embeddings_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with embeddings_path.open("xb") as file:
            np.savez_compressed(
                file,
                pair_ids=np.arange(
                    pair_count
                ),
                categories=np.array(
                    [
                        record["category"]
                        for record in records
                    ]
                ),
                test=np.stack(
                    saved_test
                ),
                reliability=np.stack(
                    saved_controls
                ),
            )

    if parameters is None:
        raise RuntimeError(
            "No BREC pairs were evaluated"
        )

    deviations = [
        (
            "Single project-level seeded run; "
            "not the official repository's full seed/search procedure."
        ),
        (
            "Training epochs/lr/weight_decay/batch_size "
            "use repository config, not reference defaults."
        ),
        (
            "Existing final-layer GIN encoder plus "
            "linear 16-D RPC head; node features are "
            f"supplied by the configured '{encoder.name}' encoding."
        ),
        (
            "Ordered PyG batches cached per pair; "
            "no DataLoader RNG consumption or workers."
        ),
        (
            "Statistics computed on CPU for "
            "CUDA/MPS portability."
        ),
        (
            "Distinguished requires candidate AND "
            "reliability; any reliability failure "
            "invalidates the full benchmark "
            "primary metric."
        ),
    ]

    if development_subset:
        deviations.append(
            (
                f"Development subset only: "
                f"{pair_count}/{full_pair_count} "
                "BREC pairs evaluated. "
                "primary_metric is intentionally None."
            )
        )

    # -------------------------------------------------------------
    # Final result record
    # -------------------------------------------------------------

    return {
        "run_id": build_run_id(
            config,
            seed,
        ),
        "timestamp": datetime.now(
            UTC
        ).isoformat(),
        "git_commit": get_git_commit(),
        "seed": seed,
        "task": config["task"]["name"],
        "dataset": config["task"]["dataset"],
        "model": "gin",
        "encoding": encoder.name,
        "hidden_dim": (
            config["model"]["hidden_dim"]
        ),
        "num_layers": (
            config["model"]["num_layers"]
        ),

        # ---------------------------------------------------------
        # Development/full benchmark status
        # ---------------------------------------------------------

        "evaluated_pairs": pair_count,
        "full_benchmark_pairs": (
            full_pair_count
        ),
        "development_subset": (
            development_subset
        ),
        "full_benchmark_completed": (
            full_benchmark_completed
        ),
        "reliability_valid": (
            reliability_valid
        ),
        "benchmark_valid": (
            benchmark_valid
        ),

        # Only a complete reliable benchmark gets the main metric.
        "primary_metric": (
            summary["overall"][
                "distinction_rate"
            ]
            if benchmark_valid
            else None
        ),

        "primary_metric_name": (
            "reliability_qualified_"
            "distinction_rate"
        ),

        "loss": (
            sum(
                record["loss"]
                for record in records
            )
            / len(records)
        ),

        "loss_description": (
            "mean final epoch training loss "
            "across independently trained pairs"
        ),

        # ---------------------------------------------------------
        # Resource usage
        # ---------------------------------------------------------

        "num_parameters": parameters,
        "preprocessing_time_sec": (
            preprocessing
        ),
        "training_time_sec": training,
        "inference_time_sec": inference,
        "evaluation_time_sec": (
            total_timer.elapsed_sec
        ),
        "process_memory_mb": (
            process_memory_mb()
        ),

        "peak_gpu_memory_mb": (
            peak_cuda_memory_mb(device)
            if device.type == "cuda"
            else None
        ),

        **system_metadata(device),

        "torch_num_threads": (
            torch.get_num_threads()
        ),

        # ---------------------------------------------------------
        # Experiment metadata
        # ---------------------------------------------------------

        "config": dict(config),
        "source_sha256": dataset.sha256,

        "protocol": {
            "reference": PROTOCOL_URL,
            "threshold": THRESHOLD,
            "atol": ATOL,
            "rtol": RTOL,
            "output_dim": OUTPUT_DIM,
            "relabelings": NUM_RELABELINGS,
            "loss_margin": 0.0,
            "loss_threshold": (
                LOSS_THRESHOLD
            ),
            "statistic_device": "cpu",
            "statistic_dtype": (
                "float32"
            ),
            "covariance_regularization": (
                None
            ),
            "sample_count_multiplier": (
                False
            ),
            "deviations": deviations,
        },

        # ---------------------------------------------------------
        # Scientific results
        # ---------------------------------------------------------

        **summary,

        "pairs": records,

        "embeddings_file": (
            str(embeddings_path)
            if embeddings_path
            is not None
            else None
        ),
    }
