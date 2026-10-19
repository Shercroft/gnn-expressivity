# ruff: noqa: F811 -- shared pytest fixture
import copy
import csv
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from test_brec_analysis import runs  # noqa: F401
from test_wp3_2 import variants

from gnn_expressivity.analysis import wp3_2, wp3_3
from gnn_expressivity.encodings import build_encoding

ROOT = Path(__file__).resolve().parents[1]


def gps_runs(runs):
    originals = variants(runs)[:5]
    result = []
    for encoding, k, seed in wp3_3.matrix():
        run = copy.deepcopy(originals[seed])
        run['model'] = run['config']['model']['name'] = 'graphgps'
        run['config']['model']['num_heads'] = 2
        run['encoding'] = encoding
        run['config']['encoding'] = wp3_3.configuration(ROOT, encoding, k)['encoding']
        run['run_id'] = wp3_3.run_id(run['config'], seed)
        result.append(run)
    return result


def test_matrix_pe_and_isolation():
    specs = wp3_3.matrix()
    assert len(specs) == 25
    assert {(e, k) for e, k, _ in specs} == {('none', None), ('rwse', None),
                                                           ('lappe', 4), ('lappe', 8), ('lappe', 16)}
    baseline = wp3_3.configuration(ROOT, 'none')
    for encoding, k, _ in specs:
        config = wp3_3.configuration(ROOT, encoding, k)
        assert config['model'] == baseline['model']
        assert config['training'] == baseline['training']
        assert config['evaluation'] == baseline['evaluation']
        encoder = build_encoding(config)
        assert encoder.name == encoding
        if encoding == 'lappe':
            assert encoder.k == k
        if encoding == 'rwse':
            assert encoder.walk_length == 16
    baseline['model']['num_heads'] = 99
    assert wp3_3.configuration(ROOT, 'none')['model']['num_heads'] == 4
    assert wp3_2.configuration(ROOT, 'rwse')['model']['name'] == 'gin'
    assert wp3_3.matrix('lappe', 8, [42]) == [('lappe', 8, 42)]


@pytest.mark.parametrize('encoding,k', [('none', 4), ('rwse', 4), ('lappe', 0), ('degree', None)])
def test_invalid_selection(encoding, k):
    with pytest.raises(ValueError):
        wp3_3.matrix(encoding, k)


def test_statistics_and_schema(runs, tmp_path):
    values = gps_runs(runs)
    rows, summary = wp3_3.aggregate(values)
    assert len(rows) == 25 and len(summary) == 5
    assert set(rows[0]) >= {'model', 'pe_type', 'k', 'seed', 'distinguished_count',
                            'total_pairs', 'distinguished_fraction'}
    for row in summary:
        assert row['mean_distinguished'] == 2
        assert row['std_distinguished'] == pytest.approx(2.5**.5)
        assert row['min_distinguished'] == 0 and row['max_distinguished'] == 4
        assert row['model'] == 'graphgps'
    paths = wp3_3.save_tables(values, tmp_path)
    with paths[1].open(newline='') as file:
        assert len(list(csv.DictReader(file))) == 5
    with pytest.raises(FileExistsError):
        wp3_3.save_tables(values, tmp_path)


@pytest.mark.parametrize('mutation', ['incomplete', 'gin', 'heads', 'training', 'duplicate', 'pairs', 'source'])
def test_reject_incompatible(runs, mutation):
    values = gps_runs(runs)
    if mutation == 'incomplete':
        values[0]['full_benchmark_completed'] = False
    elif mutation == 'gin':
        values[0]['model'] = 'gin'
    elif mutation == 'heads':
        values[0]['config']['model']['num_heads'] = 4
    elif mutation == 'training':
        values[0]['config']['training']['epochs'] += 1
    elif mutation == 'duplicate':
        values.append(values[0])
    elif mutation == 'pairs':
        values[0]['pairs'].pop()
    else:
        values[0]['source_sha256'] = 'b' * 64
    with pytest.raises(ValueError):
        wp3_3.aggregate(values)


def test_loading_missing_and_identity(runs, tmp_path):
    run = gps_runs(runs)[0]
    path = tmp_path / (run['run_id'] + '.json')
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match='Missing runs'):
        wp3_3.load_runs(tmp_path, wp3_3.matrix())
    found, missing = wp3_3.load_runs(tmp_path, wp3_3.matrix(), allow_missing=True)
    assert len(found) == 1 and len(missing) == 24
    run['seed'] = 1
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError):
        wp3_3.load_runs(tmp_path, wp3_3.matrix(), allow_missing=True)


def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('wp33_cli', ROOT / 'scripts/reproduce_wp3_3_graphgps_brec.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_resume_overwrite_and_failure(runs, tmp_path, monkeypatch):
    cli = runner(monkeypatch)
    run = gps_runs(runs)[10]
    evaluator = Mock(return_value=run)
    path = cli.run_one(run['config'], 0, tmp_path, evaluator=evaluator)
    evaluator.assert_called_once_with(run['config'], seed=0, embeddings_path=None)
    assert 'graphgps_lappe' in path.name and path.stem.endswith('_k4')
    before = path.read_bytes()
    cli.run_one(run['config'], 0, tmp_path, resume=True, evaluator=evaluator)
    assert evaluator.call_count == 1
    with pytest.raises(FileExistsError):
        cli.run_one(run['config'], 0, tmp_path, evaluator=evaluator)
    with pytest.raises(RuntimeError):
        cli.run_one(run['config'], 0, tmp_path, overwrite=True,
                    evaluator=Mock(side_effect=RuntimeError('training failed')))
    assert path.read_bytes() == before
    cli.run_one(run['config'], 0, tmp_path, overwrite=True, evaluator=evaluator)
    assert evaluator.call_count == 2


def test_reliability_failure_is_retained(runs):
    run = gps_runs(runs)[0]
    run['pairs'][0]['reliable'] = False
    run['overall']['reliability_failures'] = run['by_category']['Basic']['reliability_failures'] = 1
    run['benchmark_valid'] = run['reliability_valid'] = False
    run['primary_metric'] = None
    row = wp3_3.aggregate([run])[1][0]
    assert row['all_benchmarks_valid'] is False
    assert row['std_distinguished'] is None


def test_embedding_overwrite_and_resume(runs, tmp_path, monkeypatch):
    cli = runner(monkeypatch)
    run = gps_runs(runs)[0]
    run['config']['evaluation']['save_embeddings'] = True

    def evaluate(config, *, seed, embeddings_path):
        np.savez(embeddings_path, pair_ids=np.arange(400),
                 test=np.zeros((400, 64, 16)), reliability=np.zeros((400, 64, 16)))
        return run | {'embeddings_file': str(embeddings_path)}

    path = cli.run_one(run['config'], 0, tmp_path, overwrite=True, evaluator=evaluate)
    saved = json.loads(path.read_text())
    artifact = Path(saved['embeddings_file'])
    assert artifact.is_file() and artifact.parent == tmp_path
    unexpected = Mock(side_effect=AssertionError('must reuse completed result'))
    cli.run_one(run['config'], 0, tmp_path, resume=True, evaluator=unexpected)
    artifact.write_bytes(b'corrupted')
    with pytest.raises(ValueError):
        cli.run_one(run['config'], 0, tmp_path, resume=True, evaluator=unexpected)
