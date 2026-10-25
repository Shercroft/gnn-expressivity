# WP4.3 Person B: split infrastructure

Status: engineering infrastructure ready for review; scientific WP4.3 is NOT complete.
All work is isolated to `gnn-expressivity-wp4-b`, branch `analysis/wp4-b-infrastructure`.
No active GraphGPS checkpoint/result was read or changed. WP3 loaders, models,
encoders, evaluator, dependencies and environments were not modified.

## Records and adapters

`data/controlled_splits.py::GraphRecord` stores graph_id, canonical edge pairs,
num_nodes, directed, graph_family, target, generation_seed, generator_version,
generator_config, optional structural_metadata and optional rectangular finite
node_features. `num_edges` and `graph_density` are computed and serialized, and
supplied redundant values are checked on load. `target=None` is allowed until a
scientific target exists; training rejects unknown targets. No final task or
label function is implemented.

The first schema is for simple, loop-free graphs. Each undirected edge appears
once in records; the PyG adapter emits both orientations. Directed arcs appear
once. Self-loops, parallel/duplicate edges and out-of-range endpoints fail.
Density is m/[n(n-1)/2] undirected or m/[n(n-1)] directed; n=1 has density 0 by
explicit convention. Multigraph, weighted-graph or loop-inclusive density would
need a separately approved schema. Existing BREC ingestion is not involved.

## Strategies and diagnostics

- IID: deterministic shuffle of sorted graph IDs using a local random.Random(seed),
  optional target stratification, explicit nonempty train/validation/iid_test.
  Too-small strata fail rather than silently lose validation/test classes.
- OOD-size: explicit disjoint half-open ID and OOD node-count ranges. ID graphs
  provide train/validation/IID test; held-out graphs go to ood_size_test.
- OOD-family: explicit disjoint ID/held-out family lists; OOD families never enter
  training, ID validation or IID test.
- OOD-density: explicit disjoint half-open density ranges with the same ID versus
  OOD separation. Boundary membership, especially density=1, must be reviewed.

Records outside specified regimes are recorded as excluded, not reused elsewhere.
The initial implementation rejects overlapping regimes; it does not implement an
implicit overlap exception. Fractions and regimes are supplied externally, not
chosen as a final scientific design. Default fractions are engineering defaults
and need approval with the manifest, not a prescribed experiment.

Every split records counts, target histograms, size/density min/mean/max, families,
generator versions/configurations, generation seeds and exact IDs. OOD manifests
also warn that shifts can correlate and record observed mean size/density changes.
These diagnostics do not assert causal isolation. Label/family distributions and
range metadata must be inspected before approving size/density/family experiments.

## Leakage, immutability and approval

Unique IDs and disjoint split memberships are checked. Topologically isomorphic
duplicates under different IDs are rejected across the ENTIRE input dataset,
including within a split and excluded pools. This is deliberately conservative:
node/edge attributes are ignored for duplicate detection, so equal topology with
different attributes is rejected too. Exact NetworkX isomorphism follows coarse
invariant bucketing (direction, n, m, degree signatures); an edge-list hash is
never claimed to detect isomorphism. An explicit comparison budget fails closed
when exhausted. Exact isomorphism may be costly; this initial implementation is
not a scalable graph-canonicalization service, and needs feasibility review on
future generators. The configured audit budget is preserved in the manifest.

Dataset hashes include sorted full records: topology, target, features, metadata
and generator configuration/seeds. Configuration/manifest hashes detect changes.
Validation recomputes deterministic memberships and distributions, so rehashing
an internally inconsistent split does not make it valid. JSON writes use mode x,
refuse overwrite and preserve approved files. Approved files should be stored in
an externally controlled read-only/versioned location. Hashes are tamper-evident
consistency checks, NOT cryptographic approval signatures or filesystem ACLs.

Creation always produces a DRAFT. `approve_manifest` requires a reviewer identity
and evidence reference, returns a separately sealed approved copy, and never
rewrites a file. No approval or final manifest was created here outside temporary
synthetic tests. Human approval evidence must be authentic; a local string is not
an authorization service. The runner rejects drafts and requires a task approval
reference before training. This enforces the software order split-before-tuning;
prior tuning outside this tool cannot be detected. Only ID validation is supported;
OOD test sets must not guide manual tuning or task revision.

## Future commands (not scientific experiments)

From this worktree, with an existing compatible interpreter and this worktree's
src on the import path, after approved generators export graph-record JSON:

```powershell
python scripts/create_wp4_split_manifest.py --records <records.json> --spec <split_spec.json> --output <draft_manifest.json>
```

The spec contains task_id, dataset_version, split_type, seed, optional fractions,
stratify and explicit regimes. Person A/reviewer must inspect and approve it via
`approve_manifest`, recording evidence and writing a NEW approved file. There is
no auto-approval CLI. No proposed numerical task ranges or final sizes are supplied.

## Dependencies on Person A / WP4.1?WP4.2

Before real splits/experiments, Person A must provide and approve:

1. Formal task ID/version, graph domain, scalar labels/targets and invariance
   requirements, including classification class mapping or regression units.
2. Generator implementations/versions, graph families, parameter support, seeds,
   feasibility and sampling protocol; proof/checks that generated graphs meet
   the intended structural properties.
3. Independent exact/validated ground-truth oracles, small-instance sanity cases,
   label-noise policy, duplicate/isomorphism policy and allowed node/edge features.
4. Intended ID/OOD regimes, split fractions/strata, label coverage and explicit
   controls for size, density, family and correlated target changes.
5. Reviewed dataset/version identity and signed-off split membership before
   model tuning; approval evidence and rules for any future manifest revisions.

No local/motif/global strict WL hierarchy, novelty claim, or BREC-to-OOD guarantee
is assumed. Mathematical task definitions and structural controls remain blocked.

## Validation

Tiny synthetic records test IID/OOD strategies, directed/undirected density,
stratification, relabelled duplicates, range overlap, hash consistency, exclusive
writes and training-only nested subsets. See the WP4.4 report for the combined
lightweight test command and result. No full WP3 test suite or real training run
was launched while GraphGPS was active.

Combined focused WP4 verification: **71 passed**, one existing warning. See the WP4.4 report for the exact one-thread, worktree-local command.


Focused review: runner/native preprocessing now canonically orders records by
graph_id, matching the order-invariant dataset identity. A stochastic-encoder
regression test verifies unchanged per-graph features, model initialization and
predictions under input permutation. Split/manifest/isomorphism code and approval
rules were not changed by this correction. Final lightweight suite: 71 passed,
one existing warning (26.18 seconds). See WP4.4 for native integration limits.
