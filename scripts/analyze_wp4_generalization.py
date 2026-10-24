"""Analyze only supplied completed WP4 results; no training."""
import argparse
import json
from pathlib import Path
from gnn_expressivity.analysis.wp4_generalization import save_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-seeds', nargs='+', type=int)
    args = parser.parse_args()
    paths = sorted(args.results_dir.glob('wp4_*.json'))
    results = [json.loads(path.read_text()) for path in paths]
    for path in save_analysis(results, args.output_dir, expected_seeds=args.expected_seeds):
        print(path)


if __name__ == '__main__':
    main()
