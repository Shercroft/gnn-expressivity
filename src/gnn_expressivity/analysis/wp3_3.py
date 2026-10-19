"""Controlled GraphGPS PE comparison using the shared full-run analysis."""

import csv
from pathlib import Path

from . import wp3_2
from .brec_analysis import SEEDS

KS = (4, 8, 16)
run_id = wp3_2.run_id


def matrix(encoding=None, k=None, seeds=SEEDS):
    if encoding not in (None, 'none', 'rwse', 'lappe'):
        raise ValueError('Expected none, rwse or lappe')
    # Reuse seed/k validation, including rejecting k for non-LapPE conditions.
    wp3_2.matrix('rwse' if encoding == 'none' else encoding, k, seeds)
    conditions = [('none', None), ('rwse', None), *[('lappe', value) for value in KS]]
    if encoding is not None:
        conditions = [(e, value) for e, value in conditions if e == encoding]
    if k is not None:
        conditions = [('lappe', k)]
    return [(e, value, seed) for e, value in conditions for seed in seeds]


def configuration(root, encoding, k=None):
    matrix(encoding, k, [0])
    return wp3_2.configuration(root, encoding, k, model_name='graphgps')


def validate_run(run):
    wp3_2.validate_run(run, model_name='graphgps')
    settings = run['config']['model']
    heads = settings.get('num_heads')
    if type(heads) is not int or heads <= 0 or settings['hidden_dim'] % heads:
        raise ValueError('Missing or invalid GraphGPS num_heads')
    encoding = run['config']['encoding']
    name = encoding['name']
    if name == 'lappe':
        matrix(name, encoding.get('k'), [run['seed']])
        if type(encoding.get('k')) is not int or encoding.get('eigen_tol', 0) <= 0:
            raise ValueError('Missing LapPE parameters')
    elif name == 'rwse' and (type(encoding.get('walk_length')) is not int or encoding['walk_length'] <= 0):
        raise ValueError('Missing RWSE parameters')


def load_runs(directory, specs, *, allow_missing=False):
    runs, missing = wp3_2.load_runs(directory, specs, allow_missing=allow_missing, model_name='graphgps')
    for run in runs:
        validate_run(run)
    return runs, missing


def aggregate(runs):
    for run in runs:
        validate_run(run)
    by_seed, summary = wp3_2.aggregate(runs, model_name='graphgps')
    for row in [*by_seed, *summary]:
        row['pe_type'] = row.pop('encoding')
        row['model'] = 'graphgps'
    return by_seed, summary


def save_tables(runs, output_dir):
    tables = aggregate(runs)
    paths = [Path(output_dir) / name for name in
             ('wp3_3_graphgps_by_seed.csv', 'wp3_3_graphgps_summary.csv')]
    if any(path.exists() for path in paths):
        raise FileExistsError('WP3.3 tables exist; use another output directory')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for path, rows in zip(paths, tables):
        with path.open('x', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return paths
