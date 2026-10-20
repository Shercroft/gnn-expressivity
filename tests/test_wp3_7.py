import copy
import csv
import json

import pytest

from gnn_expressivity.analysis.wp3_7 import aggregate, extract, load_results, save_tables
from test_brec_analysis import runs  # noqa: F401
from test_wp3_5 import values  # noqa: F401


def test_extraction_and_units(values):
    run = values[0]
    for pair in run['pairs']:
        pair['pair_elapsed_sec'] = 2.0
        pair['epochs'] = 3
    row, _ = extract(run)
    assert row['parameter_count'] == 100 and row['total_pairs'] == 400
    assert row['preprocessing_total_sec'] == 10
    assert row['training_mean_sec'] == row['inference_mean_sec'] == 10/400
    assert row['preprocessing_mean_sec'] == row['total_sec_per_pair'] == 10/400
    assert row['mean_pair_elapsed_sec'] == 2
    assert row['mean_pair_epochs'] == 3
    assert row['training_median_sec'] is None
    assert row['process_rss_snapshot_bytes'] == 10*1024**2
    assert row['distinguished_pairs_per_hour'] == 0


def test_aggregation_and_sample_sd(values):
    _, summary, _ = aggregate(values)
    row = summary[0]
    assert row['n_seeds'] == 5 and json.loads(row['exact_seeds']) == list(range(5))
    assert row['mean_distinguished_count'] == 2
    assert row['std_distinguished_count'] == pytest.approx(2.5**.5)
    assert row['mean_training_total_sec'] == 12
    assert row['std_training_total_sec'] == pytest.approx(2.5**.5)
    assert row['n_training_total_sec'] == 5
    assert aggregate(values[:1])[1][0]['std_training_total_sec'] is None


def test_missing_metrics_are_not_zero(values):
    for key in ('training_time_sec', 'process_memory_mb', 'num_parameters'):
        values[0].pop(key)
    rows, summary, audit = aggregate(values[:2])
    assert rows[0]['training_total_sec'] is None
    assert rows[0]['training_mean_sec'] is None
    assert rows[0]['parameter_count'] is None
    # RSS availability differs, so measurement groups remain explicit.
    assert len(summary) == 2
    assert 'training_total_sec' in audit['resources'][0]['unavailable']
    values[1].pop('training_time_sec')
    assert all(r['mean_training_total_sec'] is None for r in aggregate(values[:2])[1])


def test_partial_metric_denominator(values):
    values[0].pop('training_time_sec')
    _, summary, _ = aggregate(values[:2])
    assert len(summary) == 1
    assert summary[0]['mean_training_total_sec'] == 11
    assert summary[0]['n_training_total_sec'] == 1
    assert summary[0]['std_training_total_sec'] is None


def test_parameter_consistency(values):
    values[1]['num_parameters'] += 1
    values[1]['device'] = 'different'  # Consistency also enforced across environments.
    with pytest.raises(ValueError, match='Parameter count changed'):
        aggregate(values[:2])


def test_different_k_can_have_different_parameters(values):
    for run, k, parameters in [(values[0], 4, 100), (values[1], 16, 200)]:
        run['encoding'] = run['config']['encoding']['name'] = 'lappe'
        run['config']['encoding']['k'] = k
        run['num_parameters'] = parameters
    _, summary, _ = aggregate(values[:2])
    assert {r['k'] for r in summary} == {4,16}
    assert {r['mean_parameter_count'] for r in summary} == {100,200}


@pytest.mark.parametrize('field', ['device', 'torch_num_threads', 'os', 'torch_version'])
def test_environment_grouping(values, field):
    values[1][field] = 'different'
    _, summary, _ = aggregate(values[:2])
    assert len(summary) == 2
    assert len({r['environment_id'] for r in summary}) == 2


def test_unknown_environment_not_pooled(values):
    for run in values[:2]:
        run.pop('device')
    assert len(aggregate(values[:2])[1]) == 2


def test_measurement_profile_separation(values):
    values[1]['resource_profile'] = {'rss_sampling': True}
    assert len(aggregate(values[:2])[1]) == 2
    values[1].pop('resource_profile')
    values[1]['pairs'][0]['pair_elapsed_sec'] = 1
    assert len(aggregate(values[:2])[1]) == 2
    assert extract(values[1])[0]['mean_pair_elapsed_sec'] is None


def test_training_variant_separation(values):
    values[1]['config']['training']['epochs'] *= 2
    assert len(aggregate(values[:2])[1]) == 2


def test_memory_semantics(values):
    values[0]['peak_gpu_memory_mb'] = 123
    row, _ = extract(values[0])
    assert row['peak_cpu_rss_bytes'] is None
    assert row['peak_cuda_allocated_bytes'] is None
    assert row['peak_cuda_reserved_bytes'] is None
    assert 'snapshot' in row['cpu_memory_kind']
    values[0]['device'] = 'cuda:0'
    values[0]['accelerator'] = 'synthetic GPU'
    row, _ = extract(values[0])
    assert row['peak_cuda_allocated_bytes'] == 123*1024**2
    assert row['peak_cuda_reserved_bytes'] is None
    assert 'framework peak allocated' in row['cuda_memory_kind']


def test_zero_wall_time_efficiency(values):
    values[1]['evaluation_time_sec'] = 0
    assert extract(values[1])[0]['distinguished_pairs_per_hour'] is None
    assert extract(values[0])[0]['distinguished_pairs_per_hour'] == 0


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True])
def test_bad_telemetry(values, value):
    values[0]['training_time_sec'] = value
    with pytest.raises(ValueError, match='Invalid resource'):
        extract(values[0])


def test_diagnostics_and_missing_runs(values, tmp_path):
    (tmp_path/'brec_gin_none_seed0.json').write_text(json.dumps(values[0]))
    (tmp_path/'brec_gin_none_seed1_range0_10.json').write_text('range')
    (tmp_path/'brec_gin_none_seed1_checkpoint.json').write_text('checkpoint')
    (tmp_path/'brec_gin_none_seed1_first2.json').write_text('smoke')
    partial = copy.deepcopy(values[1])
    partial['full_benchmark_completed'] = False
    (tmp_path/'brec_gin_none_seed1.json').write_text(json.dumps(partial))
    with pytest.raises(ValueError, match='Missing runs'):
        load_results(tmp_path)
    rows, summary, audit = load_results(tmp_path, allow_missing=True)
    assert len(rows) == 1 and len(audit['discovery']['skipped']) == 4
    assert audit['discovery']['missing']
    assert audit['coverage'][0]['missing_seeds'] == [1,2,3,4]
    paths = save_tables(rows, summary, audit, tmp_path/'analysis')
    assert len(paths) == 3 and all(p.exists() for p in paths)
    with paths[0].open(newline='') as file:
        row = next(csv.DictReader(file))
        assert row['peak_cpu_rss_bytes'] == ''
    with pytest.raises(FileExistsError):
        save_tables(rows, summary, audit, tmp_path/'analysis')


def test_range_and_incomplete_rejected(values):
    values[0]['pair_range'] = [0,10]
    with pytest.raises(ValueError, match='diagnostic'):
        aggregate(values[:1])
    values[0].pop('pair_range')
    values[0]['full_benchmark_completed'] = False
    with pytest.raises(ValueError, match='Incomplete'):
        aggregate(values[:1])


def test_duplicate_rejected(values):
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate([values[0],values[0]])


def test_invalid_expressivity_not_averaged(values):
    run = values[0]
    run['pairs'][0]['reliable'] = False
    run['overall']['reliability_failures'] = 1
    run['by_category']['Basic']['reliability_failures'] = 1
    run['benchmark_valid'] = run['reliability_valid'] = False
    run['primary_metric'] = None
    rows, summary, _ = aggregate(values[:2])
    assert rows[0]['benchmark_valid'] is False
    assert rows[0]['distinguished_pairs_per_hour'] is None
    row = summary[0]
    assert row['n_seeds'] == 2 and row['n_valid_seeds'] == 1
    assert row['mean_distinguished_count'] == 1
    assert row['mean_training_total_sec'] == 10.5
    assert row['all_benchmarks_valid'] is False
    assert row['total_reliability_failures'] == 1
