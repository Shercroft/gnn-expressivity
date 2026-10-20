"""Measure seed and training variability using saved finalized BREC runs only."""
import argparse
import json
from pathlib import Path

from gnn_expressivity.analysis.wp3_6 import load_results, save_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/wp3_6/analysis'))
    parser.add_argument('--allow-missing', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--condition', action='append', help='Repeat MODEL:ENCODING[:K]; default: project matrices')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error('Seeds must be unique integers in [0, 2**32)')
    conditions = None
    if args.condition:
        conditions = []
        for value in args.condition:
            parts = value.split(':')
            try:
                model, encoding = parts[:2]
                k = int(parts[2]) if len(parts) == 3 else None
                if (len(parts) not in (2, 3) or model not in ('gin', 'graphgps')
                        or encoding not in ('none', 'degree', 'rwse', 'lappe')
                        or ((encoding == 'lappe') != (k is not None))
                        or (k is not None and k <= 0)):
                    raise ValueError()
            except ValueError:
                parser.error('Expected MODEL:ENCODING[:K], with positive k for LapPE only')
            conditions.append((model, encoding, k))
    tables, audit = load_results(args.results_dir, seeds=args.seeds, conditions=conditions,
                                allow_missing=args.allow_missing)
    print(json.dumps(audit, indent=2))
    for path in save_tables(tables, audit, args.output_dir):
        print(path)


if __name__ == '__main__':
    main()
