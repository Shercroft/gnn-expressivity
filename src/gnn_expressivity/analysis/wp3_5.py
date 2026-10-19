"""Category diagnostics from finalized saved BREC runs; no model evaluation."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from .brec_analysis import CATEGORIES, experiment_signature


def identity(run):
    return (run['model'], run['encoding'],
            run['config']['encoding'].get('k') if run['encoding'] == 'lappe' else None)


def category_rows(run):
    """Validate full coverage and reconcile pair-derived counts with saved summaries."""
    try:
        return _category_rows(run)
    except (KeyError, TypeError) as error:
        raise ValueError(f'Missing/invalid result schema: {error}') from error


def _category_rows(run):
    model, encoding, k = identity(run)
    if run['task'] != 'brec' or run['dataset'] != 'BREC':
        raise ValueError('Expected BREC result')
    source = run['source_sha256']
    if (len(source) != 64 or any(c not in '0123456789abcdef' for c in source)
            or not run['protocol']):
        raise ValueError('Missing source/protocol provenance')
    if model not in ('gin', 'graphgps') or encoding not in ('none', 'degree', 'rwse', 'lappe'):
        raise ValueError('Unsupported model/encoding')
    if (run['config']['model']['name'] != model
            or run['config']['encoding']['name'] != encoding
            or type(run['seed']) is not int
            or (encoding == 'lappe' and (type(k) is not int or k <= 0))):
        raise ValueError('Inconsistent model/encoding/k/seed identity')
    n = run['full_benchmark_pairs']
    if (n != sum(CATEGORIES.values()) or run['evaluated_pairs'] != n
            or run['full_benchmark_completed'] is not True
            or run['development_subset'] is not False
            or run.get('pair_range', [0, n]) != [0, n]
            or '_range' in run['run_id']):
        raise ValueError('Incomplete or diagnostic range run')
    pairs = run['pairs']
    if (len(pairs) != n or any(type(p['pair_id']) is not int for p in pairs)
            or {p['pair_id'] for p in pairs} != set(range(n))):
        raise ValueError('Pairs must cover full benchmark exactly once')
    groups = defaultdict(list)
    for pair in pairs:
        if pair['category'] not in CATEGORIES:
            raise ValueError(f"Unknown category: {pair['category']}")
        if any(type(pair[key]) is not bool for key in ('distinguished', 'reliable', 'candidate_distinguished')):
            raise ValueError('Pair decisions must be boolean')
        if pair['distinguished'] != (pair['candidate_distinguished'] and pair['reliable']):
            raise ValueError('Inconsistent saved pair decision')
        groups[pair['category']].append(pair)
    if set(groups) != set(CATEGORIES) or set(run['by_category']) != set(groups):
        raise ValueError('Missing/unexpected categories')
    rows = []
    totals = defaultdict(int)
    for category in CATEGORIES:
        group = groups[category]
        counts = {'pairs': len(group), 'distinguished': sum(p['distinguished'] for p in group),
                  'candidate_distinguished': sum(p['candidate_distinguished'] for p in group),
                  'reliability_failures': sum(not p['reliable'] for p in group)}
        saved = run['by_category'][category]
        for key, value in counts.items():
            if saved[key] != value:
                raise ValueError(f'Category count mismatch: {category}/{key}')
            totals[key] += value
        fraction = counts['distinguished'] / counts['pairs']
        if not math.isclose(saved['distinction_rate'], fraction):
            raise ValueError('Category fraction mismatch')
        rows.append(dict(model=model, encoding=encoding, k=k, seed=run['seed'], category=category,
                         distinguished_count=counts['distinguished'], total_pairs=counts['pairs'],
                         distinguished_fraction=fraction, benchmark_valid=run['benchmark_valid'],
                         reliability_failures=counts['reliability_failures'],
                         result_path=run.get('result_path', '')))
    if any(run['overall'][key] != value for key, value in totals.items()) or totals['pairs'] != n:
        raise ValueError('Overall/category totals disagree')
    valid = totals['reliability_failures'] == 0
    if (type(run['benchmark_valid']) is not bool or run['benchmark_valid'] != valid
            or run['reliability_valid'] is not valid):
        raise ValueError('Validity disagrees with reliability failures')
    fraction = totals['distinguished'] / n
    if (not math.isclose(run['overall']['distinction_rate'], fraction)
            or run['primary_metric'] != (fraction if valid else None)):
        raise ValueError('Invalid overall/primary metric')
    return rows


def aggregate(runs):
    """Return per-seed, long summary, wide comparison, and LapPE curve rows."""
    by_seed, groups, signatures, seen = [], defaultdict(list), {}, set()
    reference_membership = None
    for run in sorted(runs, key=lambda r: (*map(str, identity(r)), r['seed'])):
        rows = category_rows(run)
        membership = {p['pair_id']: p['category'] for p in run['pairs']}
        if reference_membership is not None and membership != reference_membership:
            raise ValueError('Category membership differs across runs')
        reference_membership = membership
        condition = identity(run)
        key = (*condition, run['seed'])
        if key in seen:
            raise ValueError(f'Duplicate run: {key}')
        seen.add(key)
        signature = experiment_signature(run)
        signature['category_membership'] = sorted((p['pair_id'], p['category']) for p in run['pairs'])
        if condition in signatures and signatures[condition] != signature:
            raise ValueError(f'Incompatible runs: {condition}')
        signatures[condition] = signature
        by_seed.extend(rows)
        for row in rows:
            groups[(*condition, row['category'])].append(row)
    summary = []
    for (model, encoding, k, category), rows in groups.items():
        counts = [r['distinguished_count'] for r in rows]
        fractions = [r['distinguished_fraction'] for r in rows]
        if len({r['total_pairs'] for r in rows}) != 1:
            raise ValueError('Category size changed across seeds')
        summary.append(dict(model=model, encoding=encoding, k=k, category=category,
            n_seeds=len(rows), seeds=json.dumps([r['seed'] for r in rows]),
            mean_distinguished=mean(counts), std_distinguished=stdev(counts) if len(rows)>1 else None,
            min_distinguished=min(counts), max_distinguished=max(counts),
            mean_fraction=mean(fractions), std_fraction=stdev(fractions) if len(rows)>1 else None,
            category_pair_count=rows[0]['total_pairs'],
            all_benchmarks_valid=all(r['benchmark_valid'] for r in rows),
            total_reliability_failures=sum(r['reliability_failures'] for r in rows)))
    wide = []
    for condition in signatures:
        entries = [r for r in summary if (r['model'], r['encoding'], r['k']) == condition]
        first = entries[0]
        wide.append({key: first[key] for key in ('model', 'encoding', 'k', 'n_seeds', 'seeds', 'all_benchmarks_valid')} |
                    {r['category']: r['mean_distinguished'] for r in entries} |
                    {'Overall': sum(r['mean_distinguished'] for r in entries),
                     'total_reliability_failures': sum(r['total_reliability_failures'] for r in entries)})
    return by_seed, summary, wide, [r for r in summary if r['encoding'] == 'lappe']


def discover(results_dir, *, conditions=None, seeds=range(5), allow_missing=False):
    """Audit all BREC artifacts; skipped subsets and missing requested runs are explicit."""
    runs, skipped = [], []
    for path in sorted(Path(results_dir).rglob('brec_*.json')):
        if any(token in path.stem for token in ('_range', '_first', '_checkpoint')):
            skipped.append({'path': str(path), 'reason': 'diagnostic/subset/checkpoint filename'})
            continue
        run = json.loads(path.read_text(encoding='utf-8'))
        if run.get('model') not in ('gin', 'graphgps') or run.get('encoding') not in ('none','degree','rwse','lappe'):
            continue
        if run.get('seed') not in seeds or (conditions is not None and identity(run) not in conditions):
            skipped.append({'path': str(path), 'reason': 'outside requested seeds/conditions'})
            continue
        if run.get('full_benchmark_completed') is not True or run.get('development_subset') is not False:
            skipped.append({'path': str(path), 'reason': 'incomplete run'})
            continue
        category_rows(run)  # Malformed claimed full runs fail loudly.
        runs.append(run | {'result_path': str(path)})
    requested = conditions if conditions is not None else sorted({identity(r) for r in runs}, key=str)
    available = {(*identity(r), r['seed']) for r in runs}
    missing = [(*c, s) for c in requested for s in seeds if (*c, s) not in available]
    if missing and not allow_missing:
        raise ValueError(f'Missing configurations/seeds: {missing}')
    return runs, {'requested_conditions': requested, 'requested_seeds': list(seeds),
                  'available': sorted(available, key=str), 'missing': missing, 'skipped': skipped,
                  'scope': 'explicit conditions' if conditions is not None else 'observed conditions only; unobserved conditions unknown'}


def save_tables(runs, output_dir, audit):
    tables = aggregate(runs)
    if not runs:
        raise ValueError('No finalized full runs available')
    directory = Path(output_dir)
    names = ('by_seed_category', 'category_summary', 'category_comparison', 'lappe_category_curves')
    paths = [directory / f'wp3_5_{name}.csv' for name in names]
    audit_path = directory / 'wp3_5_availability.json'
    if any(p.exists() for p in [*paths, audit_path]):
        raise FileExistsError('WP3.5 outputs exist; choose a new output directory')
    directory.mkdir(parents=True, exist_ok=True)
    for path, rows in zip(paths, tables):
        fields = list(rows[0]) if rows else list(tables[1][0])
        with path.open('x', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return [*paths, audit_path]
