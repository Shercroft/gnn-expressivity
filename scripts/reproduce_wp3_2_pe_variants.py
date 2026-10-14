"""Run the controlled WP3.2 matrix using the existing GIN/BREC evaluator."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gnn_expressivity.analysis.wp3_2 import configuration, matrix, run_id, validate_run
from gnn_expressivity.training.brec_evaluation import evaluate_gin_brec
from gnn_expressivity.training.logging import save_run


def run_one(config, seed, results_dir, *, resume=False, evaluator=evaluate_gin_brec,
            validator=validate_run):
    directory = Path(results_dir)
    identity = run_id(config, seed)
    path = directory / f'{identity}.json'
    embeddings = directory / f'{identity}_embeddings.npz'
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        saved = json.loads(path.read_text(encoding='utf-8'))
        validator(saved)
        if saved['config'] != config or saved['seed'] != seed:
            raise ValueError(f'Existing configuration mismatch: {path}')
        if config['evaluation']['save_embeddings']:
            if not embeddings.is_file():
                raise ValueError(f'Missing embeddings: {embeddings}')
            with np.load(embeddings, allow_pickle=False) as arrays:
                if not np.array_equal(arrays['pair_ids'], np.arange(400)) or any(
                    arrays[key].shape != (400, 64, 16) or not np.isfinite(arrays[key]).all()
                    for key in ('test', 'reliability')
                ):
                    raise ValueError(f'Invalid embeddings: {embeddings}')
        print(f'Reusing {path}', flush=True)
        return path
    if embeddings.exists():
        raise FileExistsError(f'Orphan embeddings; use another directory: {embeddings}')
    result = evaluator(config, seed=seed,
        embeddings_path=embeddings if config['evaluation']['save_embeddings'] else None)
    result['run_id'] = identity
    validator(result)
    return save_run(result, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--encoding', choices=['rwse', 'lappe'])
    selection.add_argument('--all', action='store_true')
    parser.add_argument('--k', type=int)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument('--seed', type=int)
    seeds.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--results-dir', '--output-dir', type=Path, default=Path('results/wp3_2/runs'))
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--resume', action='store_true', help='Validate and skip completed runs.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    seed_values = args.seeds if args.seeds is not None else ([args.seed] if args.seed is not None else list(range(5)))
    try:
        specs = matrix(args.encoding, args.k, seed_values)
        if args.threads <= 0:
            raise ValueError('threads must be positive')
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(args.threads)
    for encoding, k, seed in specs:
        print(f'Running {encoding}, k={k}, seed={seed}', flush=True)
        print(run_one(configuration(root, encoding, k), seed, args.results_dir, resume=args.resume))


if __name__ == '__main__':
    main()
