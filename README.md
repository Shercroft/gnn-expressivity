# gnn-expressivity
Studying when increased GNN expressivity improves generalization, structural reasoning, and efficiency.

## BREC ingestion (B1–B2)

[BREC](https://github.com/GraphPKU/BREC) benchmarks graph expressivity using
400 graph pairs. We use the official `BREC_data_all.zip` at commit
`d09e8c349a8bbc0882d2932f7b37b2726f576ce9`, specifically `brec_v3.npy`.
The downloader pins this revision and checks SHA-256 for both archive and array.

From the repository root, with the existing uv environment (Windows or macOS):

```sh
uv run python scripts/download_brec.py
uv run python scripts/inspect_brec.py
uv run pytest
```

Files go in ignored `data/raw/brec/`, alongside a `source.json` provenance
manifest. No dataset files belong in Git. Both scripts read the existing
`configs/tasks/brec.yaml`; `--root PATH` overrides the default location.
An optional `task.data_root` is also supported without changing the config schema.
For manual setup, download the same
[pinned official archive](https://raw.githubusercontent.com/GraphPKU/BREC/d09e8c349a8bbc0882d2932f7b37b2726f576ce9/BREC_data_all.zip),
extract only `brec_v3.npy` into `data/raw/brec/`, and run the commands above.
The downloader reuses valid files; loading itself never downloads anything.

The array contains 51,200 graph6 byte strings, not 25,600 benchmark pairs:
the first 25,600 records are 400 blocks of 32 interleaved A/B relabelings;
the remaining 25,600 are the official isomorphic reliability controls.
`BRECDataset.from_config(config)` returns 400 pair identities in official order,
each with `graph_a` and `graph_b` as CPU PyG `Data` objects, category, family,
and provenance/record offsets. Default iteration selects relabeling 0;
`get_pair(pair_id, relabeling=...)` accesses all 32 official variants and
`get_reliability_graph(pair_id, variant=...)` accesses the 64 controls per pair.
The raw array is memory-mapped and graphs are decoded on demand.

Category ranges follow the
[official generation code](https://github.com/GraphPKU/BREC/blob/d09e8c349a8bbc0882d2932f7b37b2726f576ce9/customize/dataset_v3.py):
Basic 60, Regular 100, Extension 100, CFI 100, 4-Vertex_Condition 20, and
Distance_Regular 20. Family metadata additionally splits Regular into 50 simple
and 50 strongly regular pairs. The archive supplies no class labels or node
features, so none are invented. Node IDs are preserved from each graph6 record;
undirected edges are stored in both directions and isolated nodes are retained.

Inspection verifies the full file checksum, pair count, unique IDs, categories,
and node/edge validity for all 800 representative graphs. It reports node and
undirected-edge min/median/max (each edge counted once). This is structural
ingestion validation, not a non-isomorphism proof or RPC evaluation. Tests use
small synthetic format fixtures offline; a separate integration test validates
the official dataset when it is installed and otherwise skips. No training is
performed.

## GIN baseline

`gnn_expressivity.models.GIN` accepts a PyG `Data` or `Batch`. It uses sum-based
GINConv layers with two-layer ReLU MLPs, ReLU/dropout after each convolution,
and pooling of the final node states. Existing `configs/models/gin.yaml`
settings are used unchanged. Default output is `[num_graphs, hidden_dim]`;
an optional `out_dim` adds a linear task head. This is a final-layer readout,
without concatenation of intermediate layers or batch normalization.

```python
from gnn_expressivity.models import GIN
from gnn_expressivity.training.config import load_yaml
from gnn_expressivity.training.reproducibility import get_device

device = get_device()
model = GIN.from_config(load_yaml("configs/models/gin.yaml"), in_dim=1).to(device)
model.eval()
embeddings = model.encode(graph_batch.to(device))  # Before the optional head
```

Featureless BREC graphs receive constant-one scalar inputs inside the model;
the stored graphs are unchanged. For existing floating-point node features,
pass `in_dim=feature_width`. Evaluation disables dropout. No positional
encodings, training loop, or BREC RPC evaluation are included.

Run the CPU-only forward smoke test with `uv run python scripts/test_gin_forward.py`.
It batches four small graphs and checks that `encode()` returns shape `[4, 128]`
with the current GIN config.

## GIN on BREC development experiment

Run `uv run python scripts/reproduce_gin_brec.py` after downloading BREC.
This defaults to **seed 42**, training a fresh GIN for each of the 400 pairs.
Use `--seed N` to select another seed.
It uses the existing model/task configs and `encodings/none.yaml`; CPU threads
default to 1 (`--threads` overrides). Device selection uses the existing helper.

The criterion follows the pinned
[official RPC implementation](https://github.com/GraphPKU/BREC/blob/d09e8c349a8bbc0882d2932f7b37b2726f576ce9/base/test_BREC.py):
train on 32 interleaved A/B relabelings with Adam, cosine embedding loss
(negative targets, margin 0), plateau scheduling, and early stopping below
loss 0.2. A linear 16-dimensional head supplies the RPC representations.
In evaluation mode, let D contain the 32 A-minus-B differences; compute
`mean(D).T @ pinv(cov(D)) @ mean(D)` with no sample-count multiplier or ridge.
Apply the same statistic to the 64 official isomorphic controls. A distinction
candidate requires test statistic >72.34 and not `isclose` to the control
statistic (`atol=1e-6`, `rtol=1e-5`). Reliability requires control statistic <72.34.
The table counts only candidates passing reliability. Any reliability failure
marks the run invalid and leaves `primary_metric` null; raw flags remain saved.

This is a development experiment, not a claim to reproduce published scores.
The repository config uses up to 100 epochs, learning rate 0.001, weight decay
0.00001 and batch size 32; reference defaults are 20, 0.0001, 0.0001 and 16.
Ordered batches are cached per pair (no DataLoader RNG/worker consumption),
and statistics run in float32 on CPU for portable pseudoinversion. The existing
GIN architecture and constant-one features are unchanged. Rank-deficient
covariance follows the reference pseudoinverse, including its zero statistic
for constant differences. No arbitrary distance threshold is substituted.

The per-run JSON in ignored `results/runs/` includes category and overall
counts, every pair's statistics/loss/epochs, source hash, resolved configs,
protocol differences and resource measurements. `save_embeddings` controls
an adjacent NPZ with test/control arrays `[400, 64, 16]` and aligned IDs and
categories. Pair decisions are always kept as the evaluation audit trail.
Preprocessing includes decoding/batching and model initialization; training
includes optimization; inference includes all test/control forward passes.
Total evaluation time includes these stages and statistics, but excludes final
artifact writing. RSS is sampled at completion, not peak process memory.
Use `--results-dir PATH` for a new output location; existing runs are not overwritten.

## Five-seed reproducibility (B5)

The evaluator API is `evaluate_gin_brec(config, seed=42, max_pairs=None, ...)`.
The existing RPC statistic, reliability rule and development-subset behavior
are unchanged. For example:

```sh
uv run python scripts/reproduce_gin_brec.py --seed 0 --max-pairs 20
uv run python -u scripts/run_gin_brec_seeds.py
uv run python scripts/aggregate_gin_brec.py
```

The sweep runs exactly seeds `[0, 1, 2, 3, 4]`, sequentially, using the single-run
CLI and existing evaluator. Each seed starts in a separate process with the
same uv Python environment for resource isolation. `--threads` and
`--results-dir` are supported. Completed full JSON runs are validated and skipped;
invalid JSON or orphan embedding files are reported without overwriting. Failures
do not erase successful seeds, and the runner returns a failure status if any
seed failed. Interrupted seeds restart from the beginning when rerun; there is
no per-pair checkpoint/resume mechanism.

Full runs are `results/runs/brec_gin_none_seed0.json` through `seed4.json`, with
adjacent `_embeddings.npz` files when configured. Development runs retain
`_firstN` suffixes. Aggregation reads exactly the five canonical full filenames,
ignoring seed 42, subset files and other models. It requires one record per seed,
400 evaluated pairs, complete/non-development flags, matching source hash,
model/encoding/training/task settings, parameter count and numerical protocol.
Category totals, rates, validity flags and finite resource values are checked.

Outputs are `results/summaries/gin_brec_summary.csv` (long-form statistics)
and `gin_brec_by_seed.csv` (each seed/category and recorded resource metadata).
Statistics are mean, **sample** standard deviation (`ddof=1`), minimum and maximum.
Missing CUDA measurements stay missing; a group of one has no sample deviation.
Scientific aggregates expose `all_benchmarks_valid`, and per-seed rows retain
`benchmark_valid`; invalid runs are not silently dropped. Resource statistics
are grouped by OS/device/accelerator/software/thread metadata, with those fields
preserved as JSON columns. They should not be compared as identical hardware
measurements across different groups. Physical CPU chip details are limited by
the existing profiler's metadata. Source/config/protocol metadata are retained
for future model comparisons. Existing CSVs are protected; use `--output-dir`
for a new summary location. This is a project reproducibility sweep, not an
exact reproduction of every official BREC seed/search procedure.
