"""Run the lightweight WP5.1 synthetic forward-pass diagnostics, never training."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='Fresh output directory; no overwrite')
    parser.add_argument('--permutation-seeds', nargs='+', type=int, default=[0, 1])
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory already exists')
    import torch
    from threadpoolctl import threadpool_limits
    from gnn_expressivity.analysis.wp5_permutation import run_diagnostics, write_diagnostics
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        rows = run_diagnostics(tuple(args.permutation_seeds))
        path = write_diagnostics(rows, args.output_dir)
    from collections import Counter
    print(dict(Counter(row['status'] for row in rows)))
    print(path)
    return int(any(row['status'] == 'fail' for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
