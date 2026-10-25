# WP4.5 Person B: sample-complexity infrastructure

Status: framework ready for review; scientific WP4.5 is NOT complete. No scientific
learning curve, approved sample-size grid or real sample-complexity run exists.

## Deterministic nested training subsets

`data/controlled_splits.py::nested_training_subsets` requires an approved manifest
and explicit increasing feasible sample counts. It uses ONLY approved training IDs.
Sorted IDs and a local seed define an ordering; each requested subset is a prefix,
so smaller sets nest in larger sets. No validation/IID/OOD IDs can enter.

Optional balanced scheduling shuffles per-class pools then interleaves classes.
It is approximately balanced while class supplies permit; imbalance after a class
is exhausted is explicit in actual label histograms. It is not a substitute for
an approved class-sampling policy. Every subset records requested/actual count,
subset seed, balanced flag, IDs, label distribution, parent manifest hash and
fixed evaluation-membership hash. No default final sizes are proposed.

The runner independently reconstructs supplied nested designs and rejects IDs
that disagree with their declared seed/balance policy. Multiple training seeds
can use the same subset schedule. A different subset seed is a different sampling
cohort, not an unlabelled initialization change. Pure explicit-ID subsets are
allowed for ordinary generalization runs, but are conservatively isolated in
learning curves unless accompanied by a verified nested design.

## Analysis and safeguards

`analysis/wp4_sample_complexity.py` reuses generalization result validation and
statistics. It preserves training sample count, task, full model, encoding/k,
training seed, IID/OOD absolute metrics, OOD-minus-IID, mean/sample SD and available
metric/seed counts, costs and actual/maximum epochs. Curve identity keeps exact
validation/test IDs, parent manifest/dataset, sampling design, implementation,
environment and training hyperparameters fixed. Different maximum epochs/lr/etc.
form separate curves; model/encoding changes are separate curves too. It checks
nesting across observed sizes for each comparable training seed. Missing seeds
are explicit and never zero-filled; missing sample-size grids cannot be enumerated
until approved. One seed yields NA SD, not zero variability.

Matching splits and protocols are enforced through identities rather than
silently pooling incomparable models/tasks/budgets. Budget controls must still
be chosen deliberately: equal epoch count means different update counts as n
changes. Actual epochs and resource costs reveal some of that distinction, but
this infrastructure does not identify a causal pure sample-size effect by itself.
A paired statistical significance claim is not supplied by descriptive means/SDs.

Outputs are curve-ready long CSVs plus an availability JSON. Empty inputs give
header-only tables and DATA-BLOCKED status; there are no fabricated observations.

## Future commands (not executed on real data)

```powershell
python scripts/create_wp4_training_subsets.py --records <records.json> --manifest <approved_manifest.json> --sizes <approved_sizes> --subset-seed <approved_subset_seed> --balanced --output <new_subsets.json>
python scripts/run_wp4_generalization.py --records <records.json> --manifest <approved_manifest.json> --task <approved_task.json> --config <approved_run.yaml> --subsets <subsets.json> --seeds <approved_training_seeds> --output-dir <new_results_dir>
# Planning only above; --execute is reserved for explicitly approved experiments.
python scripts/analyze_wp4_sample_complexity.py --results-dir <completed_results_dir> --output-dir <fresh_curve_dir> --expected-seeds <approved_seeds>
```

No final numerical sizes/budgets/alternative settings are selected here. Exact
commands become concrete only after dataset feasibility and scientific approval.

## Dependencies on Person A / WP4.1?WP4.2

Person A must provide the formal task/target definitions; approved graph
generators/versions and ground-truth oracles; structural and feature controls;
ID/OOD regimes and validated dataset/splits; class coverage and feasible training
sizes; and the scientific question for fixed-epoch versus fixed-update/compute
budget comparisons. Sample-size and seed grids need approval after that review.
No mathematical hierarchy or theory-to-generalization implication is invented.

## Validation and safety

Synthetic tests cover training-only nested membership, class balance where
feasible, reproducibility, fixed evaluation IDs, curve identity, missing seeds,
NA sample SD, budget separation and non-nested rejection. Training outcomes come
from a tiny mocked backend, never a real sample-complexity experiment.

All work is in the separate WP4 worktree. Existing WP3 loaders/models/training
files, dependencies and active GraphGPS result/checkpoint files are untouched.
No branch switch in the original repo, training job, commit or push was performed.

Combined focused WP4 verification: **71 passed**, one existing warning. See the WP4.4 report for the exact one-thread, worktree-local command.


Focused review: corrected shared WP4 result validation now rejects impossible
classification rates and negative/nonfinite losses/errors before curve analysis.
Legitimate None AUROC remains missing, and OOD-minus-IID is still calculated from
absolute metrics with its metric-specific interpretation. Nested subset and
curve-identity implementations were unchanged. All sample-complexity safeguard
tests pass within the final lightweight suite of 71 tests (one existing warning,
26.18 seconds). Native GraphGPS and real scientific learning curves remain pending.
