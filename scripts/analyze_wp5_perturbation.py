"""WP5.4 synthetic edge-perturbation diagnostics: forward passes only."""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1])
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory must be fresh')
    import torch
    from threadpoolctl import threadpool_limits
    from gnn_expressivity.analysis.wp5_perturbation import run_diagnostics, save_diagnostics
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        rows = run_diagnostics(tuple(args.seeds))
        path = save_diagnostics(rows, args.output_dir)
    print(dict(Counter(row['status'] for row in rows)))
    print(path)
    return int(any(row['status'] == 'fail' for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
