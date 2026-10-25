"""Nested training-subset validation and curve-ready, budget-separated reporting."""
import csv
import json
from pathlib import Path

from gnn_expressivity.analysis import wp4_generalization
from gnn_expressivity.data.controlled_splits import digest


def analyze(results, *, expected_seeds=None):
    # Validate every raw result and preserve exact sample/seed/model/config identities.
    rows, summary = wp4_generalization.analyze(results, expected_seeds=expected_seeds)
    cohorts = {}
    row_lookup = {}
    for result in results:
        identity = result['identity']
        cohort = wp4_generalization.condition_identity(result, sample_curve=True)
        curve_id = digest(cohort)
        key = (curve_id, identity['training_seed'])
        cohort_runs = cohorts.setdefault(key, {})
        count = result['sample_count']
        if count in cohort_runs:
            raise ValueError('Duplicate sample count for curve/seed')
        cohort_runs[count] = set(identity['train_ids'])
        row_lookup[digest(wp4_generalization.condition_identity(result))] = curve_id
    for key, by_count in cohorts.items():
        previous = set()
        for count, ids in sorted(by_count.items()):
            if not previous <= ids:
                raise ValueError('Training subsets are not nested for a comparable seed/curve')
            previous = ids
    for table in (rows, summary):
        for row in table:
            row['curve_id'] = row_lookup[row['condition_id']]
    # configuration/budget remains in curve_id: changed epochs/lr cannot masquerade as size effects.
    return rows, summary


def save_analysis(results, directory, *, expected_seeds=None):
    rows, summary = analyze(results, expected_seeds=expected_seeds)
    directory = Path(directory)
    paths = [directory/'wp4_sample_complexity_by_seed.csv', directory/'wp4_sample_complexity_curves.csv', directory/'wp4_sample_complexity_availability.json']
    if any(p.exists() for p in paths):
        raise FileExistsError('Sample-complexity outputs exist')
    directory.mkdir(parents=True, exist_ok=True)
    for path, table in zip(paths, (rows, summary)):
        with path.open('x', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(table[0]) if table else ['curve_id','sample_count','metric','value'])
            writer.writeheader()
            writer.writerows(table)
    paths[2].write_text(json.dumps(dict(n_runs=len(results), status='descriptive observations' if results else 'DATA-BLOCKED: no approved experiment results',
        expected_seeds=expected_seeds, actual_curve_count=len({r['curve_id'] for r in rows}),
        note='Exact fixed evaluation IDs and training budget preserved; no fabricated sample sizes or missing results'), indent=2))
    return paths
