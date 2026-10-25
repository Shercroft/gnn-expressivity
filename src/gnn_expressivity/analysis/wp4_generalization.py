"""Seed-level supervised metrics, preserving task/split/configuration identities."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from gnn_expressivity.data.controlled_splits import canonical, digest
from gnn_expressivity.training.controlled_task_runner import validate_result


def condition_identity(result, *, sample_curve=False):
    identity = dict(result['identity'])
    identity.pop('training_seed')
    if sample_curve:
        identity.pop('train_ids')
        if identity['subset_design']['policy'] == 'explicit_ids':
            identity['subset_design'] = identity['subset_design'] | {'explicit_ids_hash': digest(result['identity']['train_ids'])}
    return identity


def analyze(results, *, expected_seeds=None):
    rows, groups, seen = [], defaultdict(list), set()
    for result in results:
        validate_result(result)
        identity = result['identity']
        condition = condition_identity(result)
        cid, seed = digest(condition), identity['training_seed']
        if (cid, seed) in seen:
            raise ValueError('Duplicate comparable run/seed')
        seen.add((cid, seed))
        config = identity['config']
        common = dict(task_id=identity['task_id'], model=config['model']['name'], encoding=config['encoding']['name'],
            k=config['encoding'].get('k') if config['encoding']['name'] == 'lappe' else None,
            condition_id=cid, condition_json=canonical(condition), seed=seed, sample_count=result['sample_count'],
            training_sec=result['resources']['training_sec'], parameter_count=result['resources']['parameter_count'],
            preprocessing_sec=result['resources']['preprocessing_sec'], inference_sec=result['resources']['inference_sec'],
            planned_max_epochs=config['training']['epochs'], actual_epochs=len(result['history']), selected_epoch=result['selected_epoch'])
        for pool, values in result['performance'].items():
            for metric, value in values.items():
                if metric == 'auroc_unavailable_reason':
                    continue
                delta = result['ood_minus_iid'].get(pool, {}).get(metric)
                row = common | dict(pool=pool, metric=metric, value=value, ood_minus_iid=delta)
                rows.append(row)
                groups[cid, pool, metric].append(row)
    summary = []
    for (cid, pool, metric), group in sorted(groups.items()):
        group.sort(key=lambda r: r['seed'])
        values = [r['value'] for r in group if r['value'] is not None]
        deltas = [r['ood_minus_iid'] for r in group if r['ood_minus_iid'] is not None]
        seeds = [r['seed'] for r in group]
        summary.append({key: group[0][key] for key in ('task_id','model','encoding','k','condition_id','condition_json','sample_count')} |
            dict(pool=pool, metric=metric, n_seeds=len(seeds), exact_seeds=canonical(seeds),
                missing_seeds=canonical(sorted(set(expected_seeds or [])-set(seeds))), n_observations=len(values),
                mean=mean(values) if values else None, sample_std=stdev(values) if len(values)>1 else None,
                n_delta_observations=len(deltas), mean_ood_minus_iid=mean(deltas) if deltas else None,
                sample_std_ood_minus_iid=stdev(deltas) if len(deltas)>1 else None,
                mean_training_sec=mean(r['training_sec'] for r in group),
                mean_parameter_count=mean(r['parameter_count'] for r in group),
                mean_preprocessing_sec=mean(r['preprocessing_sec'] for r in group),
                mean_inference_sec=mean(r['inference_sec'] for r in group),
                planned_max_epochs=group[0]['planned_max_epochs'],
                mean_actual_epochs=mean(r['actual_epochs'] for r in group)))
    return rows, summary


def save_analysis(results, directory, *, expected_seeds=None):
    rows, summary = analyze(results, expected_seeds=expected_seeds)
    directory = Path(directory)
    paths = [directory/'wp4_generalization_by_seed.csv', directory/'wp4_generalization_summary.csv', directory/'wp4_generalization_availability.json']
    if any(p.exists() for p in paths):
        raise FileExistsError('Analysis outputs exist')
    directory.mkdir(parents=True, exist_ok=True)
    for path, table in zip(paths, (rows, summary)):
        with path.open('x', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(table[0]) if table else ['task_id','condition_id','pool','metric','value'])
            writer.writeheader()
            writer.writerows(table)
    paths[2].write_text(json.dumps(dict(n_runs=len(results), expected_seeds=expected_seeds,
        status='available observations only' if results else 'DATA-BLOCKED: no approved completed observations',
        coverage=[{k:r[k] for k in ('condition_id','pool','metric','exact_seeds','missing_seeds','n_observations')} for r in summary]), indent=2))
    return paths
