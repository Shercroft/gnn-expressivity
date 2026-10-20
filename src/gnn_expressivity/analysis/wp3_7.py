"""Expressivity and resource cost from existing finalized BREC telemetry."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from . import wp3_5, wp3_6
from .brec_analysis import ENVIRONMENT, experiment_signature


TOTALS = {'preprocessing_total_sec': 'preprocessing_time_sec',
          'training_total_sec': 'training_time_sec',
          'inference_total_sec': 'inference_time_sec',
          'wall_clock_total_sec': 'evaluation_time_sec'}
METRICS = ['parameter_count', *TOTALS, 'preprocessing_mean_sec', 'training_mean_sec',
           'training_median_sec', 'inference_mean_sec', 'mean_pair_elapsed_sec',
           'total_sec_per_pair', 'process_rss_snapshot_bytes', 'peak_cpu_rss_bytes',
           'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes', 'mean_pair_epochs',
           'distinguished_pairs_per_hour']
IDENTITY = ['model', 'encoding', 'k', 'configuration_id', 'environment_id',
            'measurement_id', 'environment_json', 'measurement_json']
BY_RUN_FIELDS = IDENTITY + ['seed', 'run_id', 'result_path', 'git_commit',
    'parameter_count', 'total_pairs', *list(TOTALS), 'preprocessing_mean_sec',
    'training_mean_sec', 'training_median_sec', 'inference_mean_sec',
    'mean_pair_elapsed_sec', 'total_sec_per_pair', 'mean_pair_epochs',
    'process_rss_snapshot_bytes', 'peak_cpu_rss_bytes', 'peak_cuda_allocated_bytes',
    'peak_cuda_reserved_bytes', 'cpu_memory_kind', 'cuda_memory_kind',
    *ENVIRONMENT, 'distinguished_count', 'distinguished_fraction', 'benchmark_valid',
    'reliability_failures', 'distinguished_pairs_per_hour']
SUMMARY_FIELDS = IDENTITY + ['n_seeds', 'exact_seeds', 'n_valid_seeds', 'valid_seeds',
    'mean_distinguished_count', 'std_distinguished_count', 'mean_distinguished_fraction',
    'all_benchmarks_valid', 'total_reliability_failures'] + [
        field for metric in METRICS for field in (f'mean_{metric}', f'std_{metric}', f'n_{metric}')]


def number(value, field):
    """Missing stays missing; malformed or nonfinite telemetry fails loudly."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'Invalid resource measurement: {field}')
    return value


def extract(run):
    """Map historical telemetry with original units/scope preserved."""
    wp3_5.category_rows(run)
    model, encoding, k = wp3_5.identity(run)
    n = run['evaluated_pairs']
    parameters = number(run.get('num_parameters'), 'num_parameters')
    if parameters is not None and (int(parameters) != parameters or parameters <= 0):
        raise ValueError('Parameter count must be a positive integer')
    environment = {key: run.get(key) for key in ENVIRONMENT}
    # Missing environment details cannot prove compatibility, even with another unknown run.
    incomplete = [key for key in ENVIRONMENT if key not in run or
                  (run[key] is None and key != 'accelerator')]
    if str(run.get('device', '')).startswith('cuda') and not run.get('accelerator'):
        incomplete.append('accelerator')
    environment_identity = environment | ({'unknown_environment_run': run['run_id'],
        'result_path': run.get('result_path', '')} if incomplete else {})
    pair_elapsed = [number(p.get('pair_elapsed_sec'), 'pair_elapsed_sec') for p in run['pairs']]
    complete_pair_elapsed = all(v is not None for v in pair_elapsed)
    measurement = {
        'pair_elapsed_records': ('complete' if complete_pair_elapsed else
                                 'partial' if any(v is not None for v in pair_elapsed) else 'absent'),
        'resource_profile': run.get('resource_profile', 'not recorded'),
        'cpu_memory_kind': 'final process RSS snapshot' if run.get('process_memory_mb') is not None else 'unavailable',
        'environment_incomplete': incomplete,
    }
    signature = experiment_signature(run | {'num_parameters': parameters})
    signature.pop('num_parameters')
    signature['optimizer'] = run['config'].get('optimizer')
    signature['other_config'] = {key: value for key, value in run['config'].items()
        if key not in ('model', 'encoding', 'training', 'task', 'evaluation', 'optimizer')}
    row = dict(model=model, encoding=encoding, k=k,
        configuration_id=wp3_6.identifier('configuration_', signature),
        environment_id=wp3_6.identifier('environment_', environment_identity),
        measurement_id=wp3_6.identifier('measurement_', measurement),
        environment_json=wp3_6.canonical(environment_identity),
        measurement_json=wp3_6.canonical(measurement),
        seed=run['seed'], run_id=run['run_id'], result_path=run.get('result_path', ''),
        git_commit=run.get('git_commit', ''), parameter_count=parameters, total_pairs=n,
        **{target: number(run.get(source), source) for target, source in TOTALS.items()},
        **environment, distinguished_count=run['overall']['distinguished'],
        distinguished_fraction=run['overall']['distinction_rate'], benchmark_valid=run['benchmark_valid'],
        reliability_failures=run['overall']['reliability_failures'])
    for phase in ('preprocessing', 'training', 'inference'):
        total = row[f'{phase}_total_sec']
        row[f'{phase}_mean_sec'] = total / n if total is not None else None
    # No per-pair training duration was stored: a median cannot be reconstructed.
    row['training_median_sec'] = None
    row['mean_pair_elapsed_sec'] = mean(pair_elapsed) if complete_pair_elapsed else None
    wall = row['wall_clock_total_sec']
    row['total_sec_per_pair'] = wall / n if wall is not None else None
    epochs = [number(p.get('epochs'), 'epochs') for p in run['pairs']]
    row['mean_pair_epochs'] = mean(epochs) if all(v is not None for v in epochs) else None
    rss = number(run.get('process_memory_mb'), 'process_memory_mb')
    row['process_rss_snapshot_bytes'] = rss * 1024**2 if rss is not None else None
    row['peak_cpu_rss_bytes'] = None  # Historical RSS is not a peak.
    row['cpu_memory_kind'] = measurement['cpu_memory_kind']
    cuda = str(run.get('device', '')).split(':')[0] == 'cuda'
    allocated = number(run.get('peak_gpu_memory_mb'), 'peak_gpu_memory_mb') if cuda else None
    row['peak_cuda_allocated_bytes'] = allocated * 1024**2 if allocated is not None else None
    row['peak_cuda_reserved_bytes'] = None  # Not measured by the current evaluator.
    row['cuda_memory_kind'] = 'framework peak allocated; final process since reset' if allocated is not None else 'unavailable'
    row['distinguished_pairs_per_hour'] = (row['distinguished_count'] * 3600 / wall
        if run['benchmark_valid'] and wall is not None and wall > 0 else None)
    return row, signature


def aggregate(runs):
    rows, groups, parameters, seen, catalog = [], defaultdict(list), {}, set(), {}
    membership = None
    for run in runs:
        row, signature = extract(run)
        current = {p['pair_id']: p['category'] for p in run['pairs']}
        if membership is not None and current != membership:
            raise ValueError('Category membership differs across runs')
        membership = current
        cid = row['configuration_id']
        key = (cid, row['environment_id'], row['measurement_id'], row['seed'])
        if key in seen:
            raise ValueError('Duplicate configuration/environment/measurement/seed')
        seen.add(key)
        if row['parameter_count'] is not None:
            if cid in parameters and parameters[cid] != row['parameter_count']:
                raise ValueError('Parameter count changed within exact configuration')
            parameters[cid] = row['parameter_count']
        catalog[cid] = signature
        rows.append(row)
        groups[key[:3]].append(row)
    summary = []
    for _, group in sorted(groups.items()):
        group.sort(key=lambda r: r['seed'])
        valid = [r for r in group if r['benchmark_valid']]
        counts = [r['distinguished_count'] for r in valid]
        row = {key: group[0][key] for key in IDENTITY}
        row.update(n_seeds=len(group), exact_seeds=wp3_6.canonical([r['seed'] for r in group]),
            n_valid_seeds=len(valid), valid_seeds=wp3_6.canonical([r['seed'] for r in valid]),
            mean_distinguished_count=mean(counts) if counts else None,
            std_distinguished_count=stdev(counts) if len(counts) > 1 else None,
            mean_distinguished_fraction=mean(r['distinguished_fraction'] for r in valid) if valid else None,
            all_benchmarks_valid=all(r['benchmark_valid'] for r in group),
            total_reliability_failures=sum(r['reliability_failures'] for r in group))
        # Costs remain useful diagnostic measurements even when reliability failed.
        for metric in METRICS:
            values = [r[metric] for r in group if r[metric] is not None]
            row[f'mean_{metric}'] = mean(values) if values else None
            row[f'std_{metric}'] = stdev(values) if len(values) > 1 else None
            row[f'n_{metric}'] = len(values)
        summary.append(row)
    audit = dict(configuration_catalog=catalog,
        resources=[dict(result_path=r['result_path'], run_id=r['run_id'], seed=r['seed'],
            available=[m for m in METRICS if r[m] is not None],
            unavailable=[m for m in METRICS if r[m] is None]) for r in rows],
        memory_policy='RSS snapshot is not peak CPU RSS; CUDA allocated peak only on CUDA; reserved peak unavailable',
        metric_provenance={
            'recorded_totals': TOTALS,
            'parameter_count': 'num_parameters: trainable parameters, existing count_parameters()',
            'phase_mean_sec': 'recorded phase total / evaluated_pairs; amortized, not separately timed',
            'total_sec_per_pair': 'evaluation_time_sec / evaluated_pairs',
            'mean_pair_elapsed_sec': 'mean of saved pair_elapsed_sec, only with complete coverage',
            'mean_pair_epochs': 'mean of saved epochs, only with complete coverage',
            'process_rss_snapshot_bytes': 'process_memory_mb * 1024**2; snapshot, not peak',
            'peak_cuda_allocated_bytes': 'peak_gpu_memory_mb * 1024**2, CUDA only',
            'distinguished_pairs_per_hour': 'valid distinguished_count * 3600 / positive recorded wall time',
            'unmeasured': ['training_median_sec', 'peak_cpu_rss_bytes', 'peak_cuda_reserved_bytes']},
        grouping_policy='Exact scientific/training/source/protocol config, environment, and measurement coverage/profile',
        validity_policy='Costs include all finalized runs; primary expressivity means include only valid seeds; denominators explicit',
        limitations=['CPU model, physical RAM and system load not recorded historically',
            'Missing environment fields force separate per-run groups',
            'Pair elapsed coverage is a proxy for diagnostics, not a measurement of profiling overhead',
            'Resume timing covers committed work plus final session, not abandoned pair work or downtime'])
    return rows, summary, audit


def load_results(results_dir, *, seeds=range(5), conditions=None, allow_missing=False):
    conditions = wp3_6.default_conditions() if conditions is None else conditions
    runs, discovery = wp3_5.discover(results_dir, conditions=conditions, seeds=seeds, allow_missing=True)
    rows, summary, audit = aggregate(runs)
    audit['discovery'] = discovery
    audit['coverage'] = [dict(configuration_id=row['configuration_id'],
        environment_id=row['environment_id'], measurement_id=row['measurement_id'],
        available_seeds=json.loads(row['exact_seeds']),
        missing_seeds=sorted(set(seeds) - set(json.loads(row['exact_seeds'])))) for row in summary]
    if not allow_missing and (discovery['missing'] or any(r['missing_seeds'] for r in audit['coverage'])):
        raise ValueError('Missing runs in requested conditions/resource groups; use --allow-missing')
    return rows, summary, audit


def save_tables(rows, summary, audit, output_dir):
    directory = Path(output_dir)
    paths = [directory / f'wp3_7_{name}' for name in
             ('resource_by_run.csv', 'expressivity_cost.csv', 'resource_audit.json')]
    if any(p.exists() for p in paths):
        raise FileExistsError('WP3.7 outputs exist; choose another output directory')
    directory.mkdir(parents=True, exist_ok=True)
    for path, fields, table in zip(paths, (BY_RUN_FIELDS, SUMMARY_FIELDS), (rows, summary)):
        with path.open('x', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
    paths[2].write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return paths
