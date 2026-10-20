"""Descriptive realized-expressivity variability from saved full BREC runs."""

import csv
import hashlib
import json
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

from . import wp3_2, wp3_3, wp3_5
from .brec_analysis import ENVIRONMENT, experiment_signature


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def identifier(prefix, value):
    return prefix + hashlib.sha256(canonical(value).encode()).hexdigest()


def scientific_condition(run):
    """All model and encoding settings define a condition, not a training sweep."""
    return {key: run['config'][key] for key in ('model', 'encoding')}


def training_variant(run):
    """Keep the complete training config and exposed batch-size knob together."""
    config = run['config']
    return {'training': config['training'], 'batch_size': config['task']['batch_size'],
            **({'optimizer': config['optimizer']} if 'optimizer' in config else {})}


COMMON = ['model', 'encoding', 'k', 'condition_id', 'hyperparameter_variant']
STATS = ['n_seeds', 'exact_seed_list', 'mean_distinguished', 'std_distinguished',
         'min_distinguished', 'max_distinguished', 'mean_fraction', 'std_fraction',
         'all_benchmarks_valid', 'total_reliability_failures', 'n_available_runs',
         'available_seeds', 'invalid_seeds', 'missing_seeds', 'environment_count',
         'environment_warning', 'interpretation_status']
SCHEMAS = {
    'seed_variability': COMMON + STATS,
    'pair_stability': COMMON + ['pair_id', 'category', 'n_seeds', 'exact_seed_list',
        'distinguished_count', 'distinguished_rate', 'classification'],
    'seed_agreement': COMMON + ['seed_a', 'seed_b', 'n_seeds', 'intersection_size',
        'union_size', 'jaccard_similarity'],
    'category_variability': COMMON + ['category', 'n_seeds', 'exact_seed_list',
        'mean_distinguished', 'std_distinguished', 'min_distinguished', 'max_distinguished',
        'mean_fraction', 'std_fraction', 'category_pair_count', 'all_benchmarks_valid',
        'total_reliability_failures'],
    'hyperparameter_by_seed': COMMON + ['seed', 'distinguished_count',
        'distinguished_fraction', 'benchmark_valid', 'reliability_failures',
        'training_json', 'environment_json', 'result_path', 'mean_pair_epochs',
        'max_pair_epochs', 'pairs_at_epoch_budget'],
    'hyperparameter_summary': COMMON + STATS,
    'theory_vs_realized': COMMON + ['architecture_json', 'theoretical_status_note',
        'realized_brec_mean', 'realized_brec_std', 'n_seeds', 'exact_seed_list',
        'all_benchmarks_valid', 'total_reliability_failures', 'interpretation_status'],
}


def default_conditions():
    """Project matrices from WP3.1–WP3.3; no invented training variants."""
    return [('gin', 'none', None), ('gin', 'degree', None),
            *[('gin', e, k) for e, k, _ in wp3_2.matrix(seeds=[0])],
            *[('graphgps', e, k) for e, k, _ in wp3_3.matrix(seeds=[0])]]


def _stats(runs, requested_seeds):
    valid = [r for r in runs if r['benchmark_valid']]
    counts = [r['overall']['distinguished'] for r in valid]
    fractions = [r['overall']['distinction_rate'] for r in valid]
    environments = {canonical({key: r.get(key) for key in ENVIRONMENT}) for r in runs}
    environment_warning = ''
    if len(environments) > 1:
        environment_warning = 'Recorded environments differ; comparison may be confounded'
    elif any(key not in r for r in runs for key in ENVIRONMENT):
        environment_warning = 'Environment metadata incomplete; compatibility unknown'
    if environment_warning:
        warnings.warn(environment_warning, UserWarning, stacklevel=2)
    return dict(n_seeds=len(valid), exact_seed_list=canonical(sorted(r['seed'] for r in valid)),
        mean_distinguished=mean(counts) if counts else None,
        std_distinguished=stdev(counts) if len(counts) > 1 else None,
        min_distinguished=min(counts) if counts else None,
        max_distinguished=max(counts) if counts else None,
        mean_fraction=mean(fractions) if fractions else None,
        std_fraction=stdev(fractions) if len(fractions) > 1 else None,
        all_benchmarks_valid=all(r['benchmark_valid'] for r in runs),
        total_reliability_failures=sum(r['overall']['reliability_failures'] for r in runs),
        n_available_runs=len(runs), available_seeds=canonical(sorted(r['seed'] for r in runs)),
        invalid_seeds=canonical(sorted(r['seed'] for r in runs if not r['benchmark_valid'])),
        missing_seeds=canonical(sorted(set(requested_seeds) - {r['seed'] for r in runs})),
        environment_count=len(environments), environment_warning=environment_warning,
        interpretation_status=('no valid seeds' if not valid else
            'single seed; seed stability not established' if len(valid) == 1 else
            'descriptive variability across available valid seeds'))


def analyze(runs, *, requested_seeds=range(5)):
    """Never pool training variants or invalid primary metrics into seed statistics."""
    tables = {name: [] for name in SCHEMAS}
    groups, invariants, conditions, variants = defaultdict(list), {}, {}, {}
    membership = None
    seen = set()
    for run in runs:
        wp3_5.category_rows(run)
        current = {p['pair_id']: p['category'] for p in run['pairs']}
        if membership is not None and membership != current:
            raise ValueError('Category membership differs across runs')
        membership = current
        condition = scientific_condition(run)
        variant = training_variant(run)
        cid, vid = identifier('condition_', condition), identifier('training_', variant)
        key = (cid, vid, run['seed'])
        if key in seen:
            raise ValueError(f'Duplicate condition/training/seed: {key}')
        seen.add(key)
        # Changes outside declared model/encoding/training knobs must not be pooled.
        signature = experiment_signature(run)
        signature.pop('training')
        signature['task'] = {k: v for k, v in signature['task'].items() if k != 'batch_size'}
        signature['other_config'] = {k: v for k, v in run['config'].items()
            if k not in ('model', 'encoding', 'training', 'task', 'evaluation', 'optimizer')}
        if cid in invariants and invariants[cid] != signature:
            raise ValueError(f'Incompatible source/protocol/task/parameter metadata: {cid}')
        invariants[cid] = signature
        conditions[cid], variants[vid] = condition, variant
        groups[cid, vid].append(run)

    availability = []
    for (cid, vid), group in sorted(groups.items()):
        group.sort(key=lambda r: r['seed'])
        model, encoding, k = wp3_5.identity(group[0])
        common = dict(model=model, encoding=encoding, k=k, condition_id=cid,
                      hyperparameter_variant=vid)
        stats = _stats(group, requested_seeds)
        summary = common | stats
        tables['seed_variability'].append(summary)
        tables['hyperparameter_summary'].append(summary.copy())
        valid = [r for r in group if r['benchmark_valid']]
        for run in group:
            epochs = [p['epochs'] for p in run['pairs'] if 'epochs' in p]
            tables['hyperparameter_by_seed'].append(common | dict(seed=run['seed'],
                distinguished_count=run['overall']['distinguished'],
                distinguished_fraction=run['overall']['distinction_rate'],
                benchmark_valid=run['benchmark_valid'],
                reliability_failures=run['overall']['reliability_failures'],
                training_json=canonical(variants[vid]),
                environment_json=canonical({key: run.get(key) for key in ENVIRONMENT}),
                result_path=run.get('result_path', ''),
                mean_pair_epochs=mean(epochs) if epochs else None,
                max_pair_epochs=max(epochs) if epochs else None,
                pairs_at_epoch_budget=sum(e == run['config']['training']['epochs'] for e in epochs)
                    if len(epochs) == len(run['pairs']) else None))
        if valid:
            # WP3.5 retains all six categories and validates each within-variant signature.
            _, categories, _, _ = wp3_5.aggregate(valid)
            for row in categories:
                row = row.copy()
                row['exact_seed_list'] = row.pop('seeds')
                tables['category_variability'].append(common | row)
            decisions = [{p['pair_id'] for p in r['pairs'] if p['distinguished']} for r in valid]
            for pair_id, category in sorted(membership.items()):
                count = sum(pair_id in selected for selected in decisions)
                classification = ('insufficient_seeds' if len(valid) < 2 else
                    'never_distinguished' if count == 0 else
                    'always_distinguished' if count == len(valid) else 'seed_sensitive')
                tables['pair_stability'].append(common | dict(pair_id=pair_id, category=category,
                    n_seeds=len(valid), exact_seed_list=stats['exact_seed_list'],
                    distinguished_count=count, distinguished_rate=count / len(valid),
                    classification=classification))
            for a, b in combinations(range(len(valid)), 2):
                intersection = len(decisions[a] & decisions[b])
                union = len(decisions[a] | decisions[b])
                tables['seed_agreement'].append(common | dict(seed_a=valid[a]['seed'],
                    seed_b=valid[b]['seed'], n_seeds=len(valid), intersection_size=intersection,
                    union_size=union, jaccard_similarity=intersection / union if union else 1.0))
        tables['theory_vs_realized'].append(common | dict(
            architecture_json=canonical(conditions[cid]),
            theoretical_status_note='not established in this project',
            realized_brec_mean=stats['mean_distinguished'], realized_brec_std=stats['std_distinguished'],
            **{key: stats[key] for key in ('n_seeds', 'exact_seed_list', 'all_benchmarks_valid',
                                         'total_reliability_failures', 'interpretation_status')}))
        availability.append(common | {key: stats[key] for key in
            ('available_seeds', 'exact_seed_list', 'invalid_seeds', 'missing_seeds',
             'environment_count', 'environment_warning')} | {'training': variants[vid]})

    comparisons = []
    for cid in conditions:
        entries = [(vid, group) for (condition_id, vid), group in groups.items() if condition_id == cid]
        environments = {canonical({key: r.get(key) for key in ENVIRONMENT})
                        for _, group in entries for r in group}
        environment_note = ('Recorded environments differ across training variants; '
                            'comparison may be confounded' if len(environments) > 1 else '')
        if environment_note and len(entries) > 1:
            warnings.warn(environment_note, UserWarning, stacklevel=2)
        fields = {}
        for vid, _ in entries:
            value = variants[vid]
            flattened = {f'training.{key}': val for key, val in value['training'].items()}
            flattened['batch_size'] = value['batch_size']
            if 'optimizer' in value:
                flattened['optimizer'] = value['optimizer']
            fields[vid] = flattened
        differing = [field for field in sorted(set().union(*(set(v) for v in fields.values())))
            if len({canonical((field in v, v.get(field))) for v in fields.values()}) > 1]
        overlaps = []
        for (va, ga), (vb, gb) in combinations(entries, 2):
            shared = sorted({r['seed'] for r in ga if r['benchmark_valid']} &
                            {r['seed'] for r in gb if r['benchmark_valid']})
            overlaps.append(dict(variant_a=va, variant_b=vb, shared_valid_seeds=shared))
        runnable = any(pair['shared_valid_seeds'] for pair in overlaps)
        comparisons.append(dict(condition_id=cid, variants=[v for v, _ in entries],
            environment_count=len(environments), environment_warning=environment_note,
            differing_training_fields=differing, pairwise_seed_overlap=overlaps,
            status='RUNNABLE' if runnable else 'DATA-BLOCKED',
            reason='controlled matched-seed comparison available' if runnable else
                'no alternative training variant' if len(entries) < 2 else
                'variants have no shared valid seeds; seed and hyperparameter effects confounded'))
    audit = dict(conditions=conditions, training_variants=variants, coverage=availability,
        hyperparameter_comparisons=comparisons,
        hyperparameter_status='RUNNABLE' if any(c['status'] == 'RUNNABLE' for c in comparisons) else 'DATA-BLOCKED',
        missing_hyperparameter_variants='No numerical grid documented; alternative settings require approval',
        invalid_run_policy='Invalid full runs retained by seed; excluded from all primary statistics, pair rates and agreement',
        empty_union_jaccard='1.0 by convention for two empty distinguished sets',
        theory_policy='Exact encoded architectures: not established in this project')
    return tables, audit


def load_results(results_dir, *, seeds=range(5), conditions=None, allow_missing=False):
    conditions = default_conditions() if conditions is None else conditions
    runs, discovery = wp3_5.discover(results_dir, conditions=conditions, seeds=seeds, allow_missing=True)
    tables, audit = analyze(runs, requested_seeds=seeds)
    audit['discovery'] = discovery
    missing = discovery['missing'] or any(json.loads(row['missing_seeds']) for row in audit['coverage'])
    if missing and not allow_missing:
        raise ValueError('Missing conditions or training-variant seeds; use --allow-missing for partial analysis')
    return tables, audit


def save_tables(tables, audit, output_dir):
    directory = Path(output_dir)
    paths = [directory / f'wp3_6_{name}.csv' for name in SCHEMAS]
    audit_path = directory / 'wp3_6_availability.json'
    if any(path.exists() for path in [*paths, audit_path]):
        raise FileExistsError('WP3.6 outputs exist; choose another directory')
    directory.mkdir(parents=True, exist_ok=True)
    for (name, fields), path in zip(SCHEMAS.items(), paths):
        with path.open('x', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(tables[name])
    audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return [*paths, audit_path]
