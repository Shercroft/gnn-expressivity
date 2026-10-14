# ruff: noqa: F811 -- imported shared pytest fixture
import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_brec_analysis import runs  # noqa: F401

from gnn_expressivity.analysis.brec_analysis import CATEGORIES
from gnn_expressivity.analysis.wp3_2 import (
    KS,
    aggregate,
    configuration,
    load_runs,
    matrix,
    run_id,
    save_tables,
)

ROOT = Path(__file__).resolve().parents[1]


def variants(runs):
    result = []
    for encoding, k, seed in matrix():
        run = copy.deepcopy(runs[seed])
        run['encoding'] = encoding
        run['config']['encoding'] = configuration(ROOT, encoding, k)['encoding']
        run['run_id'] = run_id(run['config'], seed)
        run['reliability_valid'] = True
        run['pairs'] = []
        for category, count in CATEGORIES.items():
            for _ in range(count):
                pair_id = len(run['pairs'])
                run['pairs'].append({'pair_id': pair_id, 'category': category,
                    'distinguished': pair_id < seed, 'candidate_distinguished': pair_id < seed,
                    'reliable': True})
        result.append(run)
    return result


def test_matrix_and_configs():
    assert len(matrix()) == 30
    assert matrix('rwse', seeds=[42]) == [('rwse', None, 42)]
    assert matrix('lappe', 8, [0]) == [('lappe', 8, 0)]
    for k in KS:
        config = configuration(ROOT, 'lappe', k)
        assert config['encoding'] == {'name': 'lappe', 'k': k, 'eigen_tol': 1e-8}
        assert run_id(config, 0).endswith(f'_k{k}')
    assert configuration(ROOT, 'rwse')['encoding'] == {'name': 'rwse', 'walk_length': 16}
    assert configuration(ROOT, 'rwse')['training'] == configuration(ROOT, 'none')['training']


@pytest.mark.parametrize('args', [('rwse', 8, [0]), ('lappe', 0, [0]),
                                 ('lappe', 1, [0, 0]), ('lappe', 1, [-1])])
def test_invalid_matrix(args):
    with pytest.raises(ValueError):
        matrix(*args)


def test_aggregation_schema_and_sd(runs, tmp_path):
    values = variants(runs)
    rows, summary = aggregate(values)
    assert len(rows) == 30 and len(summary) == 6
    assert set(rows[0]) >= {'encoding', 'k', 'seed', 'distinguished_count',
                            'total_pairs', 'distinguished_fraction'}
    for row in summary:
        assert row['mean_distinguished'] == 2
        assert row['std_distinguished'] == pytest.approx(2.5**0.5)
        assert row['min_distinguished'] == 0 and row['max_distinguished'] == 4
    paths = save_tables(values, tmp_path)
    assert len(paths[2].read_text().splitlines()) == 6
    with pytest.raises(FileExistsError):
        save_tables(values, tmp_path)
    assert aggregate(values[:1])[1][0]['std_distinguished'] is None


@pytest.mark.parametrize('section,key,value', [('training', 'epochs', 7),
    ('encoding', 'walk_length', 8), ('evaluation', 'save_embeddings', True)])
def test_incompatible(runs, section, key, value):
    values = variants(runs)
    values[0]['config'][section][key] = value
    with pytest.raises(ValueError, match='Incompatible'):
        aggregate(values)


def test_missing_incomplete_duplicate(runs, tmp_path):
    run = variants(runs)[0]
    with pytest.raises(ValueError, match='Missing runs'):
        load_runs(tmp_path, matrix())
    assert len(load_runs(tmp_path, matrix(), allow_missing=True)[1]) == 30
    path = tmp_path / (run['run_id'] + '.json')
    path.write_text(json.dumps(run))
    loaded, missing = load_runs(tmp_path, matrix(), allow_missing=True)
    assert len(loaded) == 1 and len(missing) == 29
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate([run, run])
    run['full_benchmark_completed'] = False
    with pytest.raises(ValueError):
        aggregate([run])


def test_invalid_benchmark_is_visible(runs):
    run = variants(runs)[0]
    run['overall']['reliability_failures'] = 1
    run['by_category']['Basic']['reliability_failures'] = 1
    run['reliability_valid'] = run['benchmark_valid'] = False
    run['primary_metric'] = None
    run['pairs'][0]['reliable'] = False
    row = aggregate([run])[1][0]
    assert row['all_benchmarks_valid'] is False
    assert row['reliability_failures'] == 1


def test_runner_propagation_resume_and_overwrite(runs, tmp_path):
    spec = importlib.util.spec_from_file_location('wp32_runner', ROOT / 'scripts/reproduce_wp3_2_pe_variants.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = variants(runs)[5]
    evaluator = Mock(return_value=run)
    path = module.run_one(run['config'], 0, tmp_path, evaluator=evaluator)
    evaluator.assert_called_once_with(run['config'], seed=0, embeddings_path=None)
    assert json.loads(path.read_text())['config']['encoding']['k'] == 1
    with pytest.raises(FileExistsError):
        module.run_one(run['config'], 0, tmp_path, evaluator=evaluator)
    module.run_one(run['config'], 0, tmp_path, resume=True, evaluator=evaluator)
    assert evaluator.call_count == 1
    changed = copy.deepcopy(run['config'])
    changed['training']['epochs'] += 1
    with pytest.raises(ValueError, match='mismatch'):
        module.run_one(changed, 0, tmp_path, resume=True, evaluator=evaluator)
