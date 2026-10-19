"""Aggregate full WP3.2 runs; optionally include saved vanilla GIN baselines."""

import argparse
from pathlib import Path

from gnn_expressivity.analysis.wp3_2 import load_runs, matrix, save_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('results/wp3_2/runs'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/wp3_2/analysis'))
    parser.add_argument('--baseline-dir', type=Path)
    parser.add_argument('--encoding', choices=['rwse', 'lappe'])
    parser.add_argument('--k', type=int)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--allow-missing', action='store_true')
    args = parser.parse_args()
    runs, missing = load_runs(args.results_dir, matrix(args.encoding, args.k, args.seeds),
                             args.baseline_dir, args.allow_missing)
    if missing:
        print('INCOMPLETE MATRIX; missing runs:\n' + '\n'.join(missing))
    for path in save_tables(runs, args.output_dir):
        print(path)


if __name__ == '__main__':
    main()
