"""Create a DRAFT split manifest from externally supplied graph records."""
import argparse
import json
from pathlib import Path

from gnn_expressivity.data.controlled_splits import create_manifest, load_records, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True, help='JSON kwargs: task_id,dataset_version,split_type,seed,fractions,regimes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    manifest = create_manifest(load_records(args.records), **spec)
    write_manifest(manifest, args.output)
    print(f'DRAFT manifest only: {args.output}; requires Person A/task and split approval before training')


if __name__ == '__main__':
    main()
