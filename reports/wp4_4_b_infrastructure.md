# WP4.4 Person B: generalization runner infrastructure

Status: review-ready interface and mocked protocol; scientific WP4.4 is NOT complete.
Task definitions/datasets/approved experiments are absent. All new code lives in
this separate WP4 worktree; WP3 scientific files and the running experiment are
untouched. The existing Python installation is read only: no install, update,
editable-install, dependency or environment-variable change was performed.

## Task and model interfaces

`training/controlled_task_runner.py::TaskSpec` explicitly identifies classification
or scalar regression, target description and task approval reference. Classification
uses approved integer IDs 0..num_classes-1; regression requires finite scalar
targets. This defines an interface, NOT a final label function.

`TorchGraphBackend` adapts graph records into fresh PyG Data/Batch objects and
reuses build_encoding/build_model without editing their implementations. Supported
conditions: GIN constant/degree/RWSE/LapPE and GraphGPS with these existing PE
choices. Constant features must be explicitly requested with node_features=constant;
node_features=record requires supplied finite rectangular features. The existing
none encoder preserves provided features, so the explicit policy prevents silently
calling a feature-rich graph a constant-feature baseline. Encoding widths must
be identical across graphs. Full model/encoding configuration, including LapPE k,
is preserved in identities. Unsupported or omitted encoding knobs are rejected;
RWSE walk_length and LapPE k/eigen_tol are explicit.

The first adapter is CPU-only. Directed records are passed as directed arcs;
mathematical suitability of each encoding for that domain requires task approval.
No claim is made that all encoders support every directed/weighted graph domain.
Data-independent encodings operate on each graph without fitting on test labels
or across datasets. Person A must approve feature/encoding admissibility.

## Protocol and checkpoints

Approved immutable manifests and task evidence are mandatory to execute.
Training uses exactly approved training IDs or a validated training-only subset.
Task, dataset/version/hash, manifest/hash, exact IDs, model/encoding/training
config, seed, runtime/backend/version/source identities and subset design are
saved. Training knobs are explicit: Adam, epochs, batch_size, learning_rate,
weight_decay, patience and min_delta. Unsupported knobs fail; no scheduler or
unrecorded tuning mechanism is introduced. Existing controlled.yaml is not a
final approved WP4 experiment configuration.

Each epoch trains only on training IDs, evaluates only ID validation loss,
selects a best checkpoint by min_delta and stops by patience. Only after training
is complete are IID and OOD test pools predicted using that checkpoint. Test
metrics never select epochs. OOD validation is not supported by this initial
protocol; an experiment using OOD validation would answer a different, partially
adapted-domain question and needs an explicit future protocol revision.

Epoch-boundary checkpoints retain current and best backend/model/optimizer state,
RNG state, history, epoch/stale counts and committed timing. Identity checks reject
changes to task/split/config/seed/runtime or implementation. CPU RNG restoration
is covered with mocked stochastic backends and a native GIN checkpoint round-trip. Local torch checkpoints use
weights_only=True and must still come from trusted local sources. Completed JSON
writes refuse overwrite; --resume validates/reuses complete results. Partial
snapshots are replaced atomically and remain as audit artifacts after completion.
A per-run exclusive lock prevents competing writers. An abrupt process death can
leave a stale lock: confirm no writer remains before manual recovery; do not delete
locks or overwrite partial outputs automatically. This is epoch-level, not
mid-epoch recovery. A failed run retains its last committed checkpoint and is not
reported as a valid completed observation.

Resource measurements separate preprocessing, training, validation and final test
inference seconds, plus trainable parameter count. Times describe committed work;
restart setup, interrupted work and wall downtime are not reconstructed. Actual
trained epochs and selected epoch are retained alongside maximum budget. The
first adapter does not promise CUDA support or measured peak memory.

## Metrics and output

Classification reports cross-entropy loss, accuracy, balanced accuracy and AUROC.
Balanced accuracy averages recall over PRESENT classes; inspect label coverage.
AUROC is NA with an explicit reason unless all configured classes occur. Binary
AUROC uses class-1 probabilities; multiclass uses macro one-vs-rest. Logits are
converted to probabilities for AUROC. Regression reports MSE loss, MAE and RMSE.
Nonfinite/malformed predictions fail. There is no composite score.

Each named IID/OOD test pool has absolute metrics. Every OOD metric also has
OOD-minus-IID, with NA preserved: negative accuracy delta indicates worse OOD
accuracy, while positive MAE/RMSE delta indicates worse OOD error. Do not confuse
these directions. Different split types may be analyzed as separate manifests;
this implementation does not silently merge their different IID pools.

`analysis/wp4_generalization.py` produces seed-level and mean/sample-SD long
CSV tables, exact/missing seeds, available metric counts, delta means/SD and costs.
One-seed SD is NA. Identity grouping keeps different tasks, exact evaluation and
training IDs, models, encodings/k, hyperparameters, environments and implementations
separate. Missing AUROC stays missing; an entirely empty result set produces
header-only tables and a DATA-BLOCKED availability JSON. Only validated completed
results enter summaries. No real performance values were fabricated.

## Future commands and execution gate

The following are interfaces for future approved inputs, NOT jobs run here:

```powershell
python scripts/run_wp4_generalization.py --records <records.json> --manifest <approved_manifest.json> --task <approved_task.json> --config <approved_run.yaml> --seeds <approved_seeds> --output-dir <new_results_dir>
# The command above is PLAN ONLY. Only after explicit experiment approval:
python scripts/run_wp4_generalization.py --records <records.json> --manifest <approved_manifest.json> --task <approved_task.json> --config <approved_run.yaml> --seeds <approved_seeds> --output-dir <new_results_dir> --execute --resume
python scripts/analyze_wp4_generalization.py --results-dir <completed_results_dir> --output-dir <fresh_analysis_dir> --expected-seeds <approved_seeds>
```

No valid scientific example configs/targets/grids are invented. All required paths
must refer to externally approved material. Use this worktree's src first on the
import path; do not repoint the original editable installation while its experiment
runs. The default CPU adapter reuses the existing environment read-only.

## Dependencies on Person A / WP4.1?WP4.2

Needed before a real run: formal prediction targets/class mapping or regression
units; approved generators and independent ground-truth validation; allowed graph
domains/features; intended structural controls and ID/OOD question; approved
manifests/validation policy; task/metric/loss relevance, training budget and seed
protocol. Person B can run engineering fixtures now but cannot certify the
mathematical task properties or choose final experimental ranges.

No theoretical beyond-1-WL claim, strict WL hierarchy of tasks, novelty claim for
triangle counting/size generalization, or BREC-to-OOD implication is asserted.

## Lightweight validation and limitations

Tests use tiny synthetic records and stochastic FakeBackend instances. They check
classification/regression metrics, task/approval gates, exact split membership,
validation-only selection, early stopping, seed reproducibility, epoch resume,
overwrite/identity checks, model/encoding separation, missing data and OOD deltas.
Adapter feature-shape tests use existing encoders on two tiny graphs and a mocked
linear pooled head. Focused review added a native two-graph GIN optimization and
checkpoint round-trip, plus six-graph, one-epoch native GIN classification and
regression smoke tests. All run with one CPU thread. Native GraphGPS supervised
training integration, other real-model/encoding combinations and real-dataset
performance testing remain deferred until approved tasks and a safe resource
window are available. These are tiny synthetic unit tests, not scientific runs.

Tests are run with the existing interpreter read-only, bytecode writes disabled,
this WP4 src prepended, one Torch/BLAS thread in the test process, and worktree-local
temporary/cache directories. These in-process test limits do not change the active
GraphGPS environment. Only the three new WP4 test modules are selected. Initial
runs exposed a missing local temporary-directory parent and two fixture assertions;
those were corrected. Final focused result: **71 passed**, one existing Torch JIT deprecation warning (26.18 seconds). No full WP3 suite was run.


Exact final verification command, run from this worktree (the existing interpreter
is reused read-only; imports are asserted to resolve to this worktree):

```powershell
New-Item -ItemType Directory -Path .tmp -Force | Out-Null
& '..\gnn-expressivity\.venv\Scripts\python.exe' -B -c "import sys; from pathlib import Path; root=Path.cwd(); sys.path.insert(0,str(root/'src')); import gnn_expressivity; assert Path(gnn_expressivity.__file__).is_relative_to(root); import torch; torch.set_num_threads(1); from threadpoolctl import threadpool_limits; ctx=threadpool_limits(limits=1); import pytest; raise SystemExit(pytest.main(['tests/test_wp4_splits.py','tests/test_wp4_generalization.py','tests/test_wp4_sample_complexity.py','--basetemp=.tmp/wp4_corrections_final','-o','cache_dir=.tmp/wp4_review_cache','-q']))"
```

Worktree verification shows only 15 new WP4 files and no tracked-file modifications.
Branch remains analysis/wp4-b-infrastructure at the original base commit. No real
scientific manifest, dataset, result table or experiment was produced. Per-process
thread limits and disabled bytecode writes keep this verification isolated from
the existing environment and the active GraphGPS process.


## Focused review corrections

- Encoding validation now delegates parameter bounds to the existing constructor
  implementations after checking the supported field set: RWSE walk_length and
  LapPE k must be positive integers excluding bool; eigen_tol must be a finite,
  positive real excluding bool. There is no invented upper bound on k/tolerance.
  NaN/infinity and unsupported encoding options are rejected without encoding a
  graph or consuming RNG. Actual encoding algorithms were not edited.
- Model configuration is checked before execution: supported fields, positive
  integer widths/layers/heads, finite nonboolean dropout in [0,1), supported
  activation/pooling, boolean train_eps and GraphGPS head divisibility. Task-owned
  in_dim/out_dim overrides and unknown fields are rejected. No model is
  constructed merely to validate its configuration.
- Result validation requires accuracy, balanced_accuracy and present AUROC in
  [0,1], and finite nonnegative CE/MSE loss, MAE and RMSE. Valid missing AUROC is
  still None. OOD-minus-IID remains OOD absolute minus IID absolute: lower OOD
  accuracy is negative, while higher OOD error is positive. Tests reject swapped
  signs or differences inconsistent with the absolute metrics.
- The inspected existing rng_state already converts NumPy's MT state array to
  a list. Native backend state including model weights, populated Adam state
  and RNG metadata round-trips through torch.save and
  torch.load(weights_only=True). Python, NumPy and CPU Torch streams restore
  exactly. No serialization workaround, unsafe loading or WP3 checkpoint edit
  was required. GPU/MPS RNG paths were not exercised.
- Both runner dispatch and native preprocessing sort GraphRecord inputs by
  graph_id before stochastic preprocessing/model initialization. Reversing input
  order retains the same dataset hash AND identical per-graph encoded features,
  model parameters and predictions, even with a stochastic encoder stand-in.
- Audited native factories accept PyG Data/Batch and produce graph-level logits
  via configurable sum/mean/max pooling plus the requested linear head. The
  classification adapter requests num_classes outputs; regression requests one
  output and flattens it. Tiny native GIN tests verify forward shape, gradients/
  optimizer update, weights-only checkpoint loading, final evaluation and
  read-only completed-result reuse. The integration test is deliberately small;
  it is not certification of large-scale training or native GraphGPS supervision.

Final complete lightweight WP4 result: **71 passed**, one existing Torch JIT
warning, **26.18 seconds**. Only the three WP4 test modules were selected with one
Torch/BLAS thread and worktree-local temporary/cache paths. The slowest call was
metric dependency startup (7.94 seconds); native smoke checks were not among the
five slowest tests (the fifth was 0.29 seconds). No full WP3 test suite, BREC
experiment or expensive training was executed. All prior approval, leakage,
validation-only selection, subset, identity, overwrite and missing-data tests pass.

Changes are confined to this worktree's WP4 runner, its generalization tests and
these WP4 reports. Existing WP3 model/encoder/checkpoint implementations, dependency
files and the original repository/active GraphGPS artifacts were not modified.
No commit or push was performed. Scientific WP4 completion remains blocked on
Person A and approved datasets/experiments as described above.
