# Repository Guide

This document explains how the `gnn-expressivity` repository is organized, what each important file is for, how to run the main workflows, and how to decide whether an experiment is only a development check or a valid research result.

> **Recommended location:** repository root as `REPOSITORY_GUIDE.md`

---

## 1. Project in one paragraph

This project studies **realized graph neural network expressivity** under realistic training and compute constraints.

The project does **not** claim to discover the known 1-WL limitation of standard message-passing GNNs or the known theoretical relationship between GIN and 1-WL. Instead, those results are used as a reproduction layer.

The main experimental question is:

> When we give a graph model additional structural information—such as degree features, random/unique identifiers, RWSE, Laplacian positional encodings, or a GraphGPS-style architecture—how much does its **realized ability to distinguish graphs** change on BREC, how stable is that change across seeds, and what computational cost is paid for it?

Later work will connect realized expressivity to IID/OOD generalization.

---

## 2. Quick start

### Install / sync the environment

This repository uses `uv`.

```powershell
uv sync
```

On Windows PowerShell, if needed:

```powershell
.venv\Scripts\Activate.ps1
```

### Run all tests

```powershell
uv run pytest
```

A clean test run is required before merging a feature branch into `main`.

### Run a full GIN/BREC experiment

```powershell
uv run python scripts/reproduce_gin_brec.py --seed 0
```

### Run a small development-only smoke test

```powershell
uv run python scripts/reproduce_gin_brec.py --seed 0 --max-pairs 2
```

**Important:** any run using `--max-pairs` is a development run only. It must not be reported as a valid full BREC benchmark result.

### Run GraphGPS/BREC

```powershell
uv run python scripts/reproduce_graphgps_brec.py --seed 0
```

---

## 3. Repository map

```text
gnn-expressivity/
│
├── configs/
│   ├── encodings/
│   ├── models/
│   └── tasks/
│
├── reports/
├── results/
│   └── summaries/
├── scripts/
│
├── src/
│   └── gnn_expressivity/
│       ├── analysis/
│       ├── data/
│       ├── encodings/
│       ├── models/
│       └── training/
│
├── tests/
│
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── pyproject.toml
├── uv.lock
└── REPOSITORY_GUIDE.md
```

---

# 4. Top-level files

## `README.md`

Use this for:

- research question
- project positioning
- quick-start commands
- headline results
- short explanation of prior work vs project contribution

Do not turn the README into a full internal implementation manual; this guide serves that purpose.

## `pyproject.toml`

Contains:

- Python package configuration
- dependencies
- test/tool configuration
- project metadata

Change it when adding/removing dependencies or changing tool configuration.

## `uv.lock`

Exact dependency lock for reproducibility.

Normally:

- commit it
- do not edit it manually

## `.python-version`

Records the intended Python version.

Current development has been tested with Python 3.11.

## `.gitignore`

Prevents environments, caches, local config, temporary datasets, per-run outputs, checkpoints, and other generated artifacts from entering Git.

Important policy:

- raw/per-run outputs generally stay local
- compact final summaries may be committed
- temporary BREC data must not be tracked
- avoid blindly running `git add .`

---

# 5. Configurations

## `configs/encodings/`

Important encoding conditions:

```text
none.yaml
degree.yaml
random.yaml
uid.yaml
rwse.yaml
lappe.yaml
```

### `none.yaml`

Constant/no extra structural encoding.

Scientific role:

> pure structural baseline

### `degree.yaml`

Adds node degree as an explicit feature.

Scientific role:

> cheap hand-designed structural information

Used in WP3.1:

```text
GIN + constant features
vs
GIN + degree features
```

### `random.yaml`

Adds random node features.

Scientific role:

> tests how externally supplied node information changes realized expressivity

Interpret carefully because random identifiers may change invariance/generalization behavior.

### `uid.yaml`

Adds unique node identifiers.

Scientific role:

> targeted expressivity vs permutation/equivariance ablation

A higher BREC score with UID does not automatically mean better structural understanding.

### `rwse.yaml`

Random Walk Structural Encoding.

Scientific role:

> structural information based on random-walk return behavior

### `lappe.yaml`

Truncated Laplacian positional encoding.

Important parameter:

```text
k
```

controls how many nontrivial Laplacian eigenvectors are used.

---

## `configs/models/`

### `gin.yaml`

GIN baseline configuration.

Scientific role:

> representative 1-WL-tight message-passing baseline

Keep model/training settings fixed for matched encoding comparisons unless hyperparameters themselves are the research variable.

### `graphgps.yaml`

GraphGPS-style baseline configuration.

Scientific role:

> modern local + global graph architecture

GraphGPS integration is an engineering milestone. A scientific claim requires a valid full experiment.

---

## `configs/tasks/brec.yaml`

Task-level BREC configuration.

Keep the same task settings for matched comparisons unless the research question explicitly changes them.

---

# 6. Data layer

## `src/gnn_expressivity/data/brec.py`

Main BREC interface.

Responsibilities include:

- loading BREC
- validating benchmark structure
- exposing category information
- providing a consistent dataset interface

The full BREC benchmark contains **400 graph pairs**.

---

# 7. Encodings

## `src/gnn_expressivity/encodings/base.py`

Defines the common encoding interface.

Why it matters:

All encodings should follow the same API so comparisons do not silently use different preprocessing conventions.

## `src/gnn_expressivity/encodings/_features.py`

Shared feature helpers used to construct baseline node features and append new structural features.

## `none.py`

No additional structural encoding.

Used for the constant-feature baseline.

## `degree.py`

Degree feature encoding.

## `random_features.py`

Random node feature encoding.

## `uid.py`

Unique identifier encoding.

## `rwse.py`

RWSE implementation.

The tests cover behavior such as:

- batching
- determinism where appropriate
- finite outputs
- edge cases
- preservation of graph data
- permutation-equivariant behavior

## `lappe.py`

Truncated Laplacian positional encoding.

The implementation uses graph Laplacian eigendecomposition.

Important caveats:

- eigenvector sign ambiguity
- degenerate eigenspaces
- permutation interpretation
- preprocessing cost

The original project plan called for eigendecomposition caching. If caching is not implemented, document that explicitly rather than silently assuming it exists.

---

# 8. Models

## `src/gnn_expressivity/models/gin.py`

GIN implementation.

Responsibilities:

- message passing
- graph-level pooling
- graph embeddings
- optional output head

Scientific role:

> standard MPNN / 1-WL-style baseline

## `src/gnn_expressivity/models/graphgps.py`

GraphGPS-style model.

Combines local graph convolution and global attention and can consume external encodings.

Supported experimental encoding conditions include:

```text
none
degree
random
uid
rwse
lappe
```

Do not claim GraphGPS is automatically “beyond 1-WL” simply because it is transformer-style.

## `src/gnn_expressivity/models/__init__.py`

Exports the model classes and model factory.

Prefer the common factory over ad-hoc model construction.

---

# 9. BREC evaluation

## `src/gnn_expressivity/training/brec_evaluation.py`

This is one of the core files in the repository.

It handles:

1. dataset loading
2. structural encoding
3. per-pair model initialization/training
4. graph embedding generation
5. BREC Reliable Paired Comparisons statistics
6. reliability controls
7. category summaries
8. timing/resource metadata
9. result validity flags
10. optional embedding storage

---

## BREC validity rules

A development subset is useful for debugging but is **not** a valid full benchmark result.

A full result should satisfy:

```text
development_subset = False
full_benchmark_completed = True
reliability_valid = True
benchmark_valid = True
```

Partial development runs intentionally do not receive a valid full-benchmark primary metric.

---

## RPC interpretation

BREC compares graph embeddings across relabelings of graph A and graph B.

A pair is counted as distinguished only when:

- the candidate comparison passes the distinction criterion
- the reliability control remains valid

This prevents unstable model behavior from being counted as genuine graph discrimination.

---

# 10. Reproducibility and resource tracking

## `src/gnn_expressivity/training/reproducibility.py`

Responsibilities:

- random seed control
- device selection
- reproducible experiment behavior

Use explicit seeds for headline experiments.

## `src/gnn_expressivity/training/logging.py`

Responsibilities include:

- run identifiers
- Git metadata
- experiment metadata

A research result should be traceable to the code/config that produced it.

## `src/gnn_expressivity/training/profiling.py`

Tracks resource metrics such as:

- wall-clock time
- parameter count
- process memory
- accelerator memory where available
- system metadata

Resource use is part of the research question, not merely engineering bookkeeping.

---

# 11. Analysis

## `src/gnn_expressivity/analysis/brec_analysis.py`

General BREC result analysis.

Responsibilities include:

- checking run consistency
- validating complete runs
- multi-seed aggregation
- rejecting mismatched experiment metadata
- producing summary statistics

## `src/gnn_expressivity/analysis/wp3_1.py`

> Current development branch: `analysis/wp3-1-brec-baselines`

WP3.1-specific analysis for:

```text
GIN + constant features
vs
GIN + degree features
```

The analysis should reject:

- development subsets
- incomplete pair runs
- wrong models
- wrong encodings
- reliability-invalid runs
- mismatched configurations
- missing/malformed metadata

This prevents invalid smoke runs from being silently averaged into the final table.

---

# 12. Scripts

## `scripts/reproduce_gin_brec.py`

Primary CLI for running GIN on BREC.

Full run:

```powershell
uv run python scripts/reproduce_gin_brec.py --seed 0
```

Smoke run:

```powershell
uv run python scripts/reproduce_gin_brec.py --seed 0 --max-pairs 2
```

## `scripts/run_gin_brec_seeds.py`

Runs repeated GIN/BREC seeds.

Used for multi-seed reproducibility.

## `scripts/aggregate_gin_brec.py`

Aggregates valid GIN/BREC runs into compact summary files.

Currently committed examples:

```text
results/summaries/gin_brec_summary.csv
results/summaries/gin_brec_by_seed.csv
```

## `scripts/reproduce_graphgps_brec.py`

Runs GraphGPS through the same shared BREC protocol.

## `scripts/analyze_wp3_1_brec_baselines.py`

> Current development branch: `analysis/wp3-1-brec-baselines`

Audits and summarizes the WP3.1 constant-vs-degree comparison.

Do not feed smoke/development outputs into a final research table.

## `scripts/download_brec.py`

Utility for obtaining BREC data.

Downloaded/raw benchmark data should remain outside normal Git tracking.

## `scripts/inspect_brec.py`

Utility for inspecting BREC contents and categories.

Useful for debugging and benchmark understanding.

---

# 13. Results

## `results/summaries/`

Use this directory for compact, final aggregated outputs that are useful to collaborators and readers.

Current examples:

```text
gin_brec_summary.csv
gin_brec_by_seed.csv
```

## Raw runs

Per-run JSON/NPZ outputs can become large and numerous.

They should generally remain local.

Do not routinely commit:

```text
results/runs/
large embedding files
temporary smoke outputs
cached intermediate artifacts
```

---

# 14. Reports

## `reports/gin_brec_reproduction.md`

Documents the GIN/BREC reproduction layer.

Use reports to record:

- protocol decisions
- deviations from reference settings
- reproduction results
- limitations
- prior work vs project contribution

---

# 15. Tests

The `tests/` directory is part of the scientific validation layer.

Important groups include:

```text
test_brec.py
test_brec_analysis.py
test_brec_evaluation.py
test_brec_seeds.py
test_encodings.py
test_experiment_infrastructure.py
test_gin.py
test_graphgps.py
test_lappe.py
test_rwse.py
test_wp3_1_analysis.py
```

Before opening or merging a PR:

```powershell
uv run pytest
```

Current WP3.1 development state has passed:

```text
291 passed
4 skipped
0 failed
```

The skipped tests are hardware-dependent cases where CUDA is unavailable.

---

# 16. Work-package map

## WP0 — Reproducibility / infrastructure

Main components:

```text
pyproject.toml
uv.lock
configs/
training/logging.py
training/profiling.py
training/reproducibility.py
tests/test_experiment_infrastructure.py
```

## WP1 — Reproduction layer

Main components:

```text
data/brec.py
models/gin.py
training/brec_evaluation.py
scripts/reproduce_gin_brec.py
scripts/run_gin_brec_seeds.py
scripts/aggregate_gin_brec.py
analysis/brec_analysis.py
reports/gin_brec_reproduction.md
results/summaries/gin_brec_*.csv
```

The graph-theory-side WP1.1/WP1.2 deliverables—minimal 1-WL implementation and explanatory graph pairs—should be maintained/merged by the theory lead if not already present.

## WP2 — Structural information / model ladder

```text
encodings/base.py
encodings/degree.py
encodings/random_features.py
encodings/uid.py
encodings/rwse.py
encodings/lappe.py
models/graphgps.py
scripts/reproduce_graphgps_brec.py
```

## WP3.1 — Constant vs degree GIN

Current status:

```text
constant-feature GIN:
    full BREC complete
    seeds 0–4 complete

degree-feature GIN:
    smoke test exists
    full matched runs still required
```

WP3.1 is complete only after both conditions have matched valid full runs and a final analysis table.

---

# 17. Current GIN/BREC baseline

Current project condition:

```text
model: GIN
encoding: constant / none
full BREC: 400 pairs
seeds: 0, 1, 2, 3, 4
```

Observed result:

```text
0 / 400 pairs distinguished
for all five project seeds
```

with:

```text
reliability failures = 0
benchmark_valid = True
```

Interpretation:

This is a **project-specific realized result under the current training/configuration**.

Do not rewrite it as:

> “GIN theoretically distinguishes 0/400 BREC pairs.”

The theoretical and realized questions are different.

---

# 18. Result hierarchy

Use this hierarchy when reporting progress.

### Level 1 — Code exists

Example:

```text
RWSE class implemented
```

Not yet a scientific result.

### Level 2 — Unit tests pass

Validates software behavior, but still not an experiment.

### Level 3 — Smoke run

Example:

```text
2 BREC pairs complete
```

Validates end-to-end execution only.

### Level 4 — Full valid run

Example:

```text
400 / 400 BREC pairs
benchmark_valid = True
```

Now it can be reported as one experiment.

### Level 5 — Matched multi-seed comparison

Example:

```text
constant vs degree
same GIN
same training config
same seed set
```

Now it can support WP3.1 analysis.

### Level 6 — Scientific interpretation

Add:

```text
category breakdown
seed variability
resource costs
graph-theoretic interpretation
```

This is the level needed for a research conclusion.

---

# 19. Git workflow

Do not develop directly on `main`.

Recommended workflow:

```text
main
  ↓
create focused branch
  ↓
implement / test
  ↓
commit
  ↓
push
  ↓
pull request
  ↓
review
  ↓
merge into main
```

Suggested branch names:

```text
analysis/wp3-1-brec-baselines
experiment/wp3-2-rwse-lappe
experiment/wp3-3-graphgps-brec
feature/lappe-cache
```

Before committing:

```powershell
git status
git diff --check
```

Before opening a PR:

```powershell
uv run pytest
git status
```

Prefer explicit `git add` paths over:

```powershell
git add .
```

when generated experiment files are present.

---

# 20. Where new work should go

## New model

```text
src/gnn_expressivity/models/
configs/models/
tests/
scripts/   # only if a dedicated CLI is needed
```

## New encoding

```text
src/gnn_expressivity/encodings/
configs/encodings/
tests/
```

## New analysis

```text
src/gnn_expressivity/analysis/
scripts/
tests/
```

## New benchmark/data interface

```text
src/gnn_expressivity/data/
configs/tasks/
tests/
```

## Final compact results

```text
results/summaries/
```

## Research explanation

```text
reports/
README.md
```

---

# 21. What not to commit

Normally keep these out of Git:

```text
.venv/
.tmp/
raw BREC data
results/runs/
large embeddings
checkpoints
cache directories
local secrets
machine-specific configs
temporary notebooks
```

See `.gitignore` for the authoritative list.

---

# 22. Current priorities

Recommended order:

1. commit/push the WP3.1 analysis branch
2. run full degree-feature GIN BREC using the same seed set as the constant baseline
3. generate the matched constant-vs-degree table
4. add category/resource interpretation
5. resolve whether LapPE eigendecomposition caching is part of the current scope
6. move to WP3.2: RWSE/LapPE BREC comparison
7. then WP3.3: GraphGPS BREC comparison

Do not open many new experiment tracks before WP3.1 has a clean scientific deliverable.

---

# 23. Project-sync reporting format

For each work package, report four things.

## Completed

What artifact is actually finished?

## Evidence

How do we know?

Examples:

```text
tests pass
full benchmark valid
PR merged
```

## Blocker

What prevents scientific completion?

## Acceptance criterion

Exactly what must happen before calling the task done?

This is clearer than saying only:

> “I finished the code.”

---

# 24. Ownership reminder

### Person A — graph theory / combinatorics

- 1-WL theory
- explanatory hard graph pairs
- BREC category interpretation
- controlled structural task design
- graph-theoretic failure analysis
- theoretical writing/review

### Person B — ML / statistics / infrastructure

- reproducible repository
- BREC integration
- GIN / GraphGPS
- structural encodings
- experiment execution
- resource profiling
- repeated seeds
- statistical summaries
- reproducibility

Both collaborators should cross-train enough to explain the full project.

---

## Final rule

A result should only be called **complete** when:

> **the code exists, the tests pass, the experiment satisfies validity criteria, the comparison is scientifically matched, and the team can explain why the result matters.**
