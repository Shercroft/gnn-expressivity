"""Plan controlled experiments; --execute additionally requires approved inputs."""
import argparse
import json
from pathlib import Path
from gnn_expressivity.data.controlled_splits import load_records, validate_manifest
from gnn_expressivity.training.config import load_yaml
from gnn_expressivity.training.controlled_task_runner import TaskSpec, run_experiment, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('records', 'manifest', 'task', 'config', 'output-dir'):
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--subsets', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**32 for s in args.seeds):
        parser.error('Unique NumPy-compatible nonnegative integer seeds required')
    records = load_records(args.records)
    manifest = json.loads(args.manifest.read_text())
    task = TaskSpec(**json.loads(args.task.read_text()))
    config = load_yaml(args.config)
    validate_manifest(manifest, records, require_approved=args.execute)
    task.validate(records)
    validate_config(config)
    subsets = json.loads(args.subsets.read_text()) if args.subsets else [None]
    if not subsets:
        raise ValueError('Empty subset schedule')
    for subset in subsets:
        if subset and (subset['parent_manifest_hash'] != manifest['manifest_hash'] or subset['actual_sample_count'] != len(subset['train_ids'])):
            raise ValueError('Subset parent/count mismatch')
        for seed in args.seeds:
            if not args.execute:
                print(json.dumps(dict(status='PLAN ONLY; no training', seed=seed,
                    sample_count=len(subset['train_ids']) if subset else len(manifest['splits']['train']),
                    manifest_hash=manifest['manifest_hash'], config=config)))
                continue
            result = run_experiment(records, manifest, task, config, seed, args.output_dir,
                train_ids=subset['train_ids'] if subset else None,
                subset_design={'policy':'nested','subset_seed':subset['subset_seed'],'balanced':subset['balanced']} if subset else None,
                resume=args.resume)
            print(result['run_id'])


if __name__ == '__main__':
    main()
