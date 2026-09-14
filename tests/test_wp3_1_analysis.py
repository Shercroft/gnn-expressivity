import copy
import json

import pytest

from gnn_expressivity.analysis.wp3_1 import aggregate, audit, save_tables, validate_run
from test_brec_analysis import runs  # noqa: F401 -- shared synthetic fixture


def both(runs):
    result = []
    for encoding in ('none', 'degree'):
        for original in runs:
            run = copy.deepcopy(original)
            run['encoding'] = encoding
            run['config']['encoding']['name'] = encoding
            run['run_id'] = f"brec_gin_{encoding}_seed{run['seed']}"
            run['reliability_valid'] = True
            run['num_parameters'] = 100 if encoding == 'none' else 110
            result.append(run)
    return result


def test_statistics_and_outputs(runs, tmp_path):
    values = both(runs)
    by_seed, summary = aggregate(values)
    assert len(by_seed) == 10
    assert [r['encoding'] for r in summary] == ['none', 'degree']
    for r in summary:
        assert r['distinguished_mean'] == 2
        assert r['distinguished_std'] == pytest.approx(2.5**.5)
        assert r['distinction_rate_mean'] == 2/400
        assert r['total_reliability_failures'] == 0
    assert summary[1]['num_parameters_mean'] == 110
    paths = save_tables(values, tmp_path)
    assert all(p.exists() for p in paths)
    with pytest.raises(FileExistsError):
        save_tables(values, tmp_path)


@pytest.mark.parametrize('key,value', [('development_subset', True), ('model','graphgps'),
    ('encoding','rwse'), ('reliability_valid', False), ('evaluated_pairs', 2)])
def test_invalid_runs(runs, key, value):
    values = both(runs)
    values[0][key] = value
    with pytest.raises(ValueError):
        aggregate(values)


def test_failure_is_not_silently_averaged(runs):
    values = both(runs)
    r = values[0]
    r['overall']['reliability_failures'] = 1
    r['by_category']['Basic']['reliability_failures'] = 1
    r['reliability_valid'] = r['benchmark_valid'] = False
    r['primary_metric'] = None
    with pytest.raises(ValueError, match='Unreliable'):
        aggregate(values)


def test_missing_field_and_mismatched_configs(runs):
    values = both(runs)
    del values[0]['training_time_sec']
    with pytest.raises(ValueError, match='Missing required'):
        validate_run(values[0], 'none', 0)
    values = both(runs)
    values[-1]['config']['training']['epochs'] += 1
    with pytest.raises(ValueError, match='Incompatible'):
        aggregate(values)


def test_audit_missing_and_reuse(runs, tmp_path):
    values = both(runs)
    for r in values[:5]:
        (tmp_path / (r['run_id']+'.json')).write_text(json.dumps(r))
    found, missing = audit(tmp_path, values[0]['config'])
    assert len(found) == 5
    assert missing == [('degree', seed) for seed in range(5)]
    wrong = tmp_path / 'brec_gin_none_seed0.json'
    r = values[0] | {'seed': 1}
    wrong.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        audit(tmp_path, values[0]['config'])
