"""Run fixed GraphGPS settings across none, RWSE and LapPE conditions."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from reproduce_wp3_2_pe_variants import run_one as shared_run_one

from gnn_expressivity.analysis.wp3_3 import configuration, matrix, run_id, validate_run
from gnn_expressivity.training.brec_evaluation import evaluate_graphgps_brec
from gnn_expressivity.training.pair_checkpoint import atomic_write


def run_checkpointed(config, seed, results_dir, *, resume=False, start_pair=0, end_pair=None):
    directory = Path(results_dir)
    identity = run_id(config, seed)
    ranged = start_pair != 0 or end_pair is not None
    if ranged:
        identity += f'_range{start_pair}_{end_pair if end_pair is not None else 400}'
    path = directory / f'{identity}.json'
    checkpoint = directory / f'{identity}.partial.pt'
    embeddings = directory / f'{identity}_embeddings.npz'
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        if not ranged:
            return shared_run_one(config, seed, directory, resume=True,
                                  evaluator=evaluate_graphgps_brec, validator=validate_run)
        saved = json.loads(path.read_text(encoding='utf-8'))
        expected_range = [start_pair, end_pair if end_pair is not None else 400]
        from gnn_expressivity.training.brec_evaluation import summarize_results
        if (saved['config'] != config or saved['seed'] != seed or
                saved['pair_range'] != expected_range or saved['run_id'] != identity):
            raise ValueError('Completed range identity mismatch')
        summarize_results(saved['pairs'], expected_range[1] - start_pair, start_pair=start_pair)
        return path
    if embeddings.exists() and not (resume and checkpoint.exists()):
        raise FileExistsError(f'Orphan embeddings without resumable checkpoint: {embeddings}')
    result = evaluate_graphgps_brec(config, seed=seed,
        embeddings_path=embeddings if config['evaluation']['save_embeddings'] else None,
        progress=lambda message: print(message, flush=True),
        start_pair=start_pair, end_pair=end_pair, checkpoint_path=checkpoint,
        resume=resume, diagnostics=True)
    result['run_id'] = identity
    if not result['development_subset']:
        # Range 0..400 is still a full evaluation, but retain its distinct filename.
        validate_run(result | {'run_id': run_id(config, seed)})
    payload = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
    atomic_write(path, lambda file: file.write(payload))
    checkpoint.unlink(missing_ok=True)
    return path


def run_one(config, seed, results_dir, *, resume=False, overwrite=False,
            evaluator=evaluate_graphgps_brec, start_pair=0, end_pair=None):
    if resume and overwrite:
        raise ValueError('resume and overwrite are mutually exclusive')
    if evaluator is evaluate_graphgps_brec:
        if not overwrite:
            return run_checkpointed(config, seed, results_dir, resume=resume,
                                    start_pair=start_pair, end_pair=end_pair)
        if start_pair != 0 or end_pair is not None:
            raise ValueError('Use another directory for range diagnostics instead of overwrite')
        # A persistent replacement directory preserves partial work and the old
        # published result. Repeating --overwrite resumes this replacement.
        directory = Path(results_dir).resolve()
        staging = directory / '.replacements' / run_id(config, seed)
        path = run_checkpointed(config, seed, staging, resume=True)
        result = json.loads(path.read_text(encoding='utf-8'))
        if config['evaluation']['save_embeddings']:
            source = Path(result['embeddings_file'])
            target = directory / source.name
            # Keep staging intact until both final artifacts are published.
            atomic_write(target, lambda file: file.write(source.read_bytes()))
            result['embeddings_file'] = str(target)
        destination = directory / path.name
        payload = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
        atomic_write(destination, lambda file: file.write(payload))
        path.unlink()
        if config['evaluation']['save_embeddings']:
            source.unlink()
        return destination
    if start_pair != 0 or end_pair is not None:
        raise ValueError('Range diagnostics support safe resume, not overwrite; use another directory')
    kwargs = {'evaluator': evaluator, 'validator': validate_run}
    if not overwrite:
        return shared_run_one(config, seed, results_dir, resume=resume, **kwargs)
    # Train and validate before replacing a prior result. Only this exact run's
    # two artifacts are replaced; a failed evaluation leaves old outputs intact.
    directory = Path(results_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    identity = run_id(config, seed)
    with TemporaryDirectory(prefix='wp3_3_', dir=directory) as staging:
        path = shared_run_one(config, seed, staging, **kwargs)
        result = json.loads(path.read_text(encoding='utf-8'))
        if config['evaluation']['save_embeddings']:
            target = directory / f'{identity}_embeddings.npz'
            source = Path(staging) / target.name
            if not source.is_file():
                raise ValueError('Evaluator did not save configured embeddings')
            result['embeddings_file'] = str(target)
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            source.replace(target)
        destination = directory / path.name
        path.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--encoding', choices=['none', 'rwse', 'lappe'])
    selection.add_argument('--all', action='store_true')
    parser.add_argument('--k', type=int)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument('--seed', type=int)
    seeds.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--results-dir', '--output-dir', type=Path, default=Path('results/wp3_3/runs'))
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--start-pair', type=int, default=0, help='Zero-based inclusive pair ID.')
    parser.add_argument('--end-pair', type=int, help='Zero-based exclusive pair ID (default 400).')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--overwrite', action='store_true', help='Replace selected runs after successful evaluation.')
    args = parser.parse_args()
    seed_values = args.seeds if args.seeds is not None else ([args.seed] if args.seed is not None else list(range(5)))
    try:
        specs = matrix(args.encoding, args.k, seed_values)
        if args.threads <= 0:
            raise ValueError('threads must be positive')
        if not 0 <= args.start_pair < (args.end_pair if args.end_pair is not None else 400) <= 400:
            raise ValueError('Require 0 <= start-pair < end-pair <= 400')
        if args.overwrite and (args.start_pair != 0 or args.end_pair is not None):
            raise ValueError('Use a new output directory for range diagnostics instead of overwrite')
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(args.threads)
    root = Path(__file__).resolve().parents[1]
    for encoding, k, seed in specs:
        print(f'GraphGPS {encoding}, k={k}, seed={seed}', flush=True)
        print(run_one(configuration(root, encoding, k), seed, args.results_dir,
                      resume=args.resume, overwrite=args.overwrite,
                      start_pair=args.start_pair, end_pair=args.end_pair))


if __name__ == '__main__':
    main()
