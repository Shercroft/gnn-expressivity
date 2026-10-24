"""Generate explicit nested training-only subsets from an approved manifest."""
import argparse
import json
from pathlib import Path
from gnn_expressivity.data.controlled_splits import load_records, nested_training_subsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--sizes', nargs='+', type=int, required=True)
    parser.add_argument('--subset-seed', type=int, required=True)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    subsets = nested_training_subsets(json.loads(args.manifest.read_text()), load_records(args.records), args.sizes,
        seed=args.subset_seed, balanced=args.balanced)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(subsets, file, indent=2)


if __name__ == '__main__':
    main()
