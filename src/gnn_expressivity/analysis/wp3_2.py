"""Controlled encoding matrix and BREC curve tables (sample SD)."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from gnn_expressivity.training.config import load_yaml, merge_configs
from gnn_expressivity.training.logging import build_run_id

from .brec_analysis import SEEDS, experiment_signature, validate_full_run

KS = (1, 2, 4, 8, 16)


def matrix(encoding=None, k=None, seeds=SEEDS):
    """Return (encoding, k, seed); None selects the complete six-condition matrix."""
    if encoding not in (None, "rwse", "lappe"):
        raise ValueError("Expected rwse or lappe")
    if k is not None and (encoding != "lappe" or type(k) is not int or k <= 0):
        raise ValueError("k requires lappe and a positive integer")
    if not seeds or len(set(seeds)) != len(seeds) or any(
        type(s) is not int or not 0 <= s < 2**32 for s in seeds
    ):
        raise ValueError("Seeds must be unique integers in [0, 2**32)")
    conditions = ([('rwse', None)] if encoding != 'lappe' else [])
    if encoding != 'rwse':
        conditions += [('lappe', value) for value in ((k,) if k is not None else KS)]
    return [(name, value, seed) for name, value in conditions for seed in seeds]


def configuration(root, encoding, k=None, *, model_name='gin'):
    config = merge_configs(*(load_yaml(Path(root) / 'configs' / p) for p in
        (f'models/{model_name}.yaml', 'tasks/brec.yaml', f'encodings/{encoding}.yaml')))
    if encoding == 'lappe':
        if type(k) is not int or k <= 0:
            raise ValueError('k must be positive')
        config['encoding']['k'] = k
    return config


def run_id(config, seed):
    suffix = f"_k{config['encoding']['k']}" if config['encoding']['name'] == 'lappe' else ''
    return build_run_id(config, seed) + suffix


def validate_run(run, *, model_name='gin'):
    """Reuse full-benchmark checks while allowing dimension IDs and arbitrary seeds."""
    try:
        encoding, seed = run['encoding'], run['seed']
        if encoding == 'none':
            matrix('rwse', seeds=[seed])
        else:
            matrix(encoding, run['config']['encoding'].get('k'), [seed])
        if run['run_id'] != run_id(run['config'], seed):
            raise ValueError('Run identity disagrees with configuration')
        # The WP3.1 validator fixes seed IDs to 0..4; adapt only its identity view.
        validate_full_run(run | {'seed': 0, 'run_id': f'brec_{model_name}_{encoding}_seed0'},
                          0, encoding, model_name=model_name)
        from gnn_expressivity.training.brec_evaluation import summarize_results

        summary = summarize_results(run['pairs'], 400)
        if any(summary[key] != run[key] for key in ('overall', 'by_category')):
            raise ValueError('Pair records disagree with summary')
        if run['reliability_valid'] != run['benchmark_valid']:
            raise ValueError('Inconsistent reliability status')
    except (KeyError, TypeError) as error:
        raise ValueError(f'Missing or malformed run field: {error}') from error


def load_runs(directory, specs, baseline_dir=None, allow_missing=False, *, model_name='gin'):
    runs, missing = [], []
    requests = [(Path(directory), e, k, s) for e, k, s in specs]
    if baseline_dir is not None:
        requests += [(Path(baseline_dir), 'none', None, s) for s in sorted({s for _, _, s in specs})]
    for folder, encoding, k, seed in requests:
        config = {'task': {'name': 'brec'}, 'model': {'name': model_name},
                  'encoding': {'name': encoding, 'k': k}}
        path = folder / (run_id(config, seed) + '.json')
        if not path.exists():
            missing.append(str(path))
            continue
        run = json.loads(path.read_text(encoding='utf-8'))
        validate_run(run, model_name=model_name)
        if (run['encoding'], run['config']['encoding'].get('k'), run['seed']) != (encoding, k, seed):
            raise ValueError(f'Filename/config mismatch: {path}')
        runs.append(run)
    if missing and not allow_missing:
        raise ValueError(f'Missing runs: {missing}')
    return runs, missing


def aggregate(runs, *, model_name='gin'):
    if not runs:
        raise ValueError('No complete runs')
    groups, seen, reference, by_seed = defaultdict(list), set(), None, []
    for run in runs:
        validate_run(run, model_name=model_name)
        encoding = run['encoding']
        k = run['config']['encoding'].get('k')
        key = encoding, k, run['seed']
        if key in seen:
            raise ValueError(f'Duplicate run: {key}')
        seen.add(key)
        signature = experiment_signature(run)
        signature.pop('encoding')
        signature.pop('num_parameters')
        signature['evaluation'] = run['config']['evaluation']
        if reference is not None and signature != reference:
            raise ValueError('Incompatible model/training/source/protocol/evaluation')
        reference = signature
        group = groups[encoding, k]
        if group and (group[0]['config']['encoding'] != run['config']['encoding'] or
                      group[0]['num_parameters'] != run['num_parameters']):
            raise ValueError('Incompatible encoding configuration/parameter count')
        group.append(run)
        by_seed.append({"encoding": encoding, "k": k, "seed": run['seed'],
            "distinguished_count": run['overall']['distinguished'], "total_pairs": 400,
            "distinguished_fraction": run['overall']['distinction_rate'],
            "benchmark_valid": run['benchmark_valid'],
            "reliability_failures": run['overall']['reliability_failures']})
    summary = []
    for (encoding, k), group in sorted(groups.items()):
        values = [r['overall']['distinguished'] for r in group]
        sd = stdev(values) if len(values) > 1 else None
        summary.append({"encoding": encoding, "k": k, "seeds": json.dumps(sorted(r['seed'] for r in group)),
            "n_seeds": len(group), "mean_distinguished": mean(values), "std_distinguished": sd,
            "min_distinguished": min(values), "max_distinguished": max(values), "total_pairs": 400,
            "mean_fraction": mean(values)/400, "std_fraction": sd/400 if sd is not None else None,
            "all_benchmarks_valid": all(r['benchmark_valid'] for r in group),
            "reliability_failures": sum(r['overall']['reliability_failures'] for r in group)})
    return by_seed, summary


def save_tables(runs, output_dir):
    by_seed, summary = aggregate(runs)
    paths = [Path(output_dir) / name for name in
             ('wp3_2_by_seed.csv', 'wp3_2_encoding_summary.csv', 'wp3_2_lappe_curve.csv')]
    if any(p.exists() for p in paths):
        raise FileExistsError('WP3.2 tables already exist; use another output directory')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for path, rows, fields in zip(paths,
            (by_seed, summary, [r for r in summary if r['encoding'] == 'lappe']),
            (list(by_seed[0]), list(summary[0]), list(summary[0]))):
        with path.open('x', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return paths
