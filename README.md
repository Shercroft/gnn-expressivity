# gnn-expressivity
Studying when increased GNN expressivity improves generalization, structural reasoning, and efficiency.

## Background, Replication, and Project Contribution

### Established prior work

This project builds on established results in graph neural network
expressivity.

Standard message-passing GNNs are limited by the distinguishing power
of the 1-dimensional Weisfeiler-Leman (1-WL) procedure under the
standard neighborhood-aggregation framework. GIN was designed to match
this level of discriminative power under suitable injectivity
assumptions.

BREC is an existing standardized benchmark for evaluating realized GNN
expressivity on difficult non-isomorphic graph pairs. These results and
the benchmark itself are prior work and are not contributions of this
project.

### Reproduction layer

Before studying stronger structural information, we validate the
experimental pipeline using a standard GIN baseline on BREC.

The current reproduction layer includes:

- official BREC data ingestion and provenance tracking;
- Reliable Paired Comparisons-style evaluation;
- a constant-feature GIN baseline;
- project-level runs with seeds 0, 1, 2, 3, and 4;
- category-level and aggregate result logging;
- resource and configuration metadata; and
- automated BREC, GIN, reproducibility, and analysis tests.

Across the current five project seeds, the GIN baseline distinguished
0 of 400 BREC pairs, with zero recorded reliability failures in each
run.

This is treated as a reproduction and pipeline-validation result, not
as a new theoretical result. The project five-seed sweep also does not
claim to reproduce every detail of the official BREC seed/search
procedure.

See `reports/gin_brec_reproduction.md` for the current reproduction
report.

### Project research question

The main project contribution is not the discovery of the 1-WL
limitation, the construction of GIN, or the introduction of BREC.

Instead, we study:

> Under realistic compute and feature budgets, when does increasing a
> graph model's realized structural expressivity improve
> generalization, and when does the additional expressivity fail to
> translate into useful predictive performance?

The experimental study therefore connects four axes:

1. realized structural expressivity;
2. IID and out-of-distribution generalization;
3. node features and positional / structural encodings; and
4. preprocessing, training, inference, and memory cost.

Later experiments extend the constant-feature GIN baseline with
structural information such as random-walk structural encodings and
truncated Laplacian positional encodings, followed by stronger model
families when feasible.

### Explicit non-claims

This project does not claim to:

- discover the 1-WL limitation of standard message-passing GNNs;
- prove the expressive power of GIN;
- introduce BREC;
- discover previously known substructure-counting limitations; or
- establish that greater BREC expressivity necessarily implies better
  generalization.

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

## WP2 node encodings

Encodings augment existing floating-point node features, using constant ones
for featureless graphs through the `none` baseline:

- `none`: baseline features only.
- `degree`: baseline + raw source-node degree; permutation-equivariant.
- `uid`: baseline + current node index divided by `n - 1` (zero for one node);
  intentionally label-sensitive.
- `random`: baseline + seeded standard-normal vectors (default dimension 8,
  seed 0), tied to current node indices; intentionally label-sensitive.
- `rwse`: baseline + node-wise return probabilities `diag(P), ..., diag(P^k)`,
  where `P = D^-1 A` uses outgoing edge counts and default `k=16` (no step zero).
  Isolated-node values are zero. This deterministic structural encoding is
  permutation-equivariant and does not depend on arbitrary node IDs.

RWSE respects directed edges, parallel edges, and existing self-loops without
adding edges. Dense float64 CPU preprocessing costs approximately `O(k n^3)`
time and `O(n^2 + nk)` memory; results match the baseline dtype/device.

Random encoding resets a local CPU generator on every call, so equal node
counts receive the same matrix without consuming the global PyTorch RNG.
UID and random encodings are intentionally not graph-relabeling invariant
when recomputed after relabeling. Effects on expressivity and generalization
remain experimental questions.

Select configs with the existing `load_yaml` / `merge_configs` utilities and
pass the merged model, task, and encoding config to `evaluate_gin_brec`.
The single-run CLI continues to select `none` by default.

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
