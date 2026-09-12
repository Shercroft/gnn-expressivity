# GIN on BREC — Reproduction Report

## Objective

Reproduce the expected qualitative behavior of a standard GIN
baseline on the BREC realized-expressivity benchmark.

This experiment is a reproduction and pipeline-validation result,
not a novel expressivity claim.

## Benchmark

- Dataset: BREC
- Total graph pairs: 400
- Total graphs: 800

Category breakdown:

- Basic: 60
- Regular: 100
- Extension: 100
- CFI: 100
- 4-Vertex Condition: 20
- Distance Regular: 20

## Model

- Model: GIN
- Hidden dimension: 128
- Number of layers: 4
- Activation: ReLU
- Dropout: 0.1
- Pooling: sum
- Trainable epsilon: true
- Encoding: none
- Parameters: 117,908

The baseline uses constant-one input node features.

## RPC Protocol

- Threshold: 72.34
- Relabelings: 32
- Output dimension: 16
- Loss threshold: 0.2
- Loss margin: 0.0
- Statistic device: CPU
- Statistic dtype: float32
- atol: 1e-6
- rtol: 1e-5

The project uses a five-seed reproducibility sweep and does not
claim to reproduce every detail of the official BREC final
seed/search procedure.

## Results

| Seed | Pairs | Distinguished | Reliability Failures | Valid | Training Time (s) | Inference Time (s) |
|---:|---:|---:|---:|---|---:|---:|
| 0 | 400 | 0 | 0 | True | 3385.71 | 20.91 |
| 1 | 400 | 0 | 0 | True | 9675.61 | 16.87 |
| 2 | 400 | 0 | 0 | True | 11845.01 | 16.85 |
| 3 | 400 | 0 | 0 | True | 2196.79 | 17.91 |
| 4 | 400 | 0 | 0 | True | 2075.37 | 17.61 |

All five seeds produced 0 distinguished pairs out of 400 and
zero reliability failures.

## Resource Measurements

For seed 0:

- Preprocessing time: 68.46 s
- Training time: 3385.71 s
- Inference time: 20.91 s
- Total evaluation time: 3475.66 s
- Parameters: 117,908
- Device: CPU
- PyTorch: 2.14.0+cpu
- Torch threads: 1
- Process memory at completion: 490.81 MB

GPU peak memory is not applicable because these runs used
CPU-only PyTorch.

Training time varied substantially across seeds and should be
treated as an observed runtime variation rather than a stable
hardware cost estimate at this stage.

## Interpretation

The five-seed experiment consistently produced no distinguished
BREC pairs while passing the reliability checks.

This result is treated as a qualitative baseline reproduction
rather than a new theoretical result.

Its purpose is to validate the standard GIN/BREC pipeline before
introducing stronger structural information and positional
encodings in later work packages.

## Protocol Deviations

The recorded experiment metadata notes the following deviations
from the official BREC reference procedure:

- Single project-level seeded runs rather than the official full
  seed/search procedure.
- Training epochs, learning rate, weight decay, and batch size
  follow the repository configuration rather than reference defaults.
- Existing GIN encoder plus a linear 16-dimensional RPC head.
- Constant-one input node features.
- Ordered PyG batches cached per pair.
- No DataLoader RNG consumption or worker processes.

## Validation

The BREC and GIN targeted test suite completed successfully:

- 81 tests passed
- 0 tests failed
- 1 non-failing PyTorch deprecation warning

## Current Status

- Official BREC data integrated.
- Full 400-pair benchmark completed.
- Five project seeds completed.
- Reliability checks passed.
- Structured JSON outputs saved.
- Aggregate CSV outputs generated.
- BREC / GIN targeted tests passed.

WP1.3 BREC integration is complete.

WP1.4 GIN baseline reproduction is complete at the BREC
experiment level; explanatory positive/negative controls can be
added later when the WP1.1/WP1.2 graph-pair work is available.
