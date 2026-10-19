# WP3.3 runtime diagnosis and pair-boundary recovery

After `Evaluated 320/400`, IDs 0–319 have completed. The next iteration is
zero-based ID 320 (human pair 321), in CFI. It loads/encodes 64 test graphs and
64 reliability graphs, creates the same fresh GraphGPS model, trains with the
unchanged stopping rule, performs inference and the existing RPC/reliability tests.
The old progress cadence then remained silent until 330 completed pairs.

Inspection of the local pinned BREC data, without training:

| Zero-based ID | Category | Nodes per A/B graph | Stored directed edges per A/B graph |
|---:|---|---:|---:|
| 319 | CFI | 60 / 60 | 180 / 180 |
| 320 | CFI | 80 / 80 | 300 / 300 |
| 321 | CFI | 82 / 82 | 324 / 324 |
| 324 | CFI | 118 / 118 | 684 / 684 |
| 329 | CFI | 86 / 86 | 380 / 380 |

These larger graphs require more attention and LapPE work. Sizes alone do not
explain hours of runtime, and the previous log cannot identify which of IDs
320–329 was slow or which phase dominated. No full training run was launched
to diagnose this change. The cause remains to be measured, not asserted.

The WP3.3 CLI now flushes START/DONE messages per pair, giving one-based sequential
index, zero-based ID, category, A/B node and edge counts, elapsed wall time,
epochs, distinction and reliability. Intermediate phase messages distinguish
preprocessing, training, inference and RPC. Counts refer to representative
relabeling 0; edge counts are PyG directed entries. The same diagnostic sizes
and pair wall time are stored with the pair records. No model computation changes.

## Diagnose human pairs 321–330

Ranges are zero-based, start inclusive, end exclusive. From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_wp3_3_graphgps_brec.py --encoding lappe --k 4 --seed 0 --threads 1 --start-pair 320 --end-pair 330 --resume
```

This uses the same pair preparation, training, inference and decisions. A standalone
range starts its RNG stream at seed 0, rather than consuming the random draws from
the omitted prefix. Thus its numbers are diagnostic, not interchangeable with
those pairs from the full sequential seed-0 run. Range artifacts have a distinct
`_range320_330` suffix, partial benchmark flags and no primary benchmark metric.
They are never merged into a full run or selected by the full-run analyzer.

## Resume the full experiment

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_wp3_3_graphgps_brec.py --encoding lappe --k 4 --seed 0 --threads 1 --resume
```

Default directory: `results/wp3_3/runs`. After each successful pair, the runner
atomically replaces `brec_graphgps_lappe_seed0_k4.partial.pt`. The initial empty
checkpoint is also saved. It contains only tensors and basic data structures and
loads with `weights_only=True`. It explicitly records partial/invalid status and
is not a result JSON. It includes completed decisions (including reliability
failures), embeddings when enabled, accumulated timings, parameter count and
Python, NumPy, CPU PyTorch and available CUDA/MPS RNG states.

On resume the exact config, seed, selected IDs, dataset hash, protocol constants,
implementation source digest, software/device metadata and thread count must
match. The completed IDs must form a contiguous prefix of the selected range.
The saved RNG state is restored before the next model is constructed. Completed
pairs are retained regardless of their result; only an interrupted uncommitted
pair restarts from its prior boundary. There is no epoch-level checkpoint.
Changing code or environment between interrupted runs intentionally fails identity
validation rather than silently producing a mixed experiment.

Once all selected pairs finish, embeddings and the normal result JSON are written
atomically per file and the checkpoint removed. A crash between final embedding
publication and JSON publication can be recovered from the checkpoint without
training again. Only the full 400-ID sequence can receive full-benchmark status,
and any reliability failure still invalidates its primary score. Finished-result
resume retains the previous full-run validation behavior. Multiple writers for
one run ID are unsupported; JSON/NPZ publication is not a multi-file transaction.

The full run's RNG/training/evaluation semantics are unchanged. Diagnostic output,
checkpoint I/O and wall/resource measurements add overhead. Accumulated timings
cover committed work plus the final session; abandoned work inside an interrupted
pair is not included. Process/peak-memory fields describe the final process.
Cross-device/version bitwise reproducibility is still not promised.

The earlier manually interrupted attempts predate checkpoints and cannot be
recovered from their progress lines. The first run using this implementation
starts from pair 0 if no matching checkpoint exists. Use the range command first
to locate the slow phase. No full 400-pair experiment is automatically started.
