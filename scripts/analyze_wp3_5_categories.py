"""Analyze saved full BREC runs by category without training."""
import argparse
import json
from pathlib import Path

from gnn_expressivity.analysis.wp3_5 import discover, save_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/wp3_5/analysis'))
    parser.add_argument('--allow-missing', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--condition', action='append', help='Repeat MODEL:ENCODING[:K] to request an explicit matrix')
    args = parser.parse_args()
    conditions = None
    if args.condition:
        conditions = []
        for value in args.condition:
            parts = value.split(':')
            if len(parts) not in (2, 3):
                parser.error('Use MODEL:ENCODING[:K]')
            model, encoding = parts[:2]
            k = int(parts[2]) if len(parts) == 3 else None
            if model not in ('gin','graphgps') or encoding not in ('none','degree','rwse','lappe') or ((encoding == 'lappe') != (k is not None)) or (k is not None and k <= 0):
                parser.error('Invalid condition or missing/extra LapPE k')
            conditions.append((model, encoding, k))
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Duplicate requested seeds')
    runs, audit = discover(args.results_dir, conditions=conditions, seeds=args.seeds, allow_missing=args.allow_missing)
    print(json.dumps(audit, indent=2))
    for path in save_tables(runs, args.output_dir, audit):
        print(path)


if __name__ == '__main__':
    main()
