"""Exercise the real evaluator loop with cheap stochastic stand-ins for training."""

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gnn_expressivity.analysis.wp3_3 import validate_run
from gnn_expressivity.training import brec_evaluation as evaluation
from gnn_expressivity.training.config import load_yaml, merge_configs


@pytest.fixture
def harness(monkeypatch):
    config = merge_configs(load_yaml('configs/models/graphgps.yaml'),
                           load_yaml('configs/tasks/brec.yaml'), load_yaml('configs/encodings/lappe.yaml'))
    config['encoding']['k'] = 4
    seen = []
    control = {'fail': None}

    class Dataset:
        sha256 = 'a' * 64

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return SimpleNamespace(pair_id=index, category='Basic', family='Basic',
                                   graphs=[SimpleNamespace(num_nodes=3, num_edges=4)] * 2)

    monkeypatch.setattr(evaluation.BRECDataset, 'from_config', lambda _: Dataset())
    monkeypatch.setattr(evaluation, 'validate_brec', lambda _: None)
    monkeypatch.setattr(evaluation, 'get_device', lambda: torch.device('cpu'))

    def batches(dataset, index, *args):
        seen.append(index)
        if index == control['fail']:
            # Consume random numbers before failing: resume must restore the last boundary.
            torch.rand(5)
            random.random()
            np.random.rand()
            raise RuntimeError('simulated interruption')
        return dataset[index], [SimpleNamespace(x=torch.ones(3, 1))], []

    monkeypatch.setattr(evaluation, 'pair_batches', batches)
    monkeypatch.setattr(evaluation, 'build_model', lambda *a, **k: torch.nn.Linear(1, 16))
    monkeypatch.setattr(evaluation, 'train_pair',
                        lambda *a: (float(torch.rand(()) + random.random() + np.random.rand()), 2))
    monkeypatch.setattr(evaluation, 'embed_batches', lambda *a: torch.rand(64, 16))
    monkeypatch.setattr(evaluation, 'rpc_statistic', lambda a: a.mean())
    return config, seen, control


def run(config, **kwargs):
    return evaluation.evaluate_graphgps_brec(config, seed=0, progress=lambda _: None, **kwargs)


def test_resume_rng_equivalence_finalization_and_diagnostics(harness, tmp_path):
    config, seen, control = harness
    reference = run(config, embeddings_path=tmp_path / 'reference.npz')
    seen.clear()
    checkpoint = tmp_path / 'run.partial.pt'
    embeddings = tmp_path / 'resumed.npz'
    control['fail'] = 2
    with pytest.raises(RuntimeError, match='interruption'):
        run(config, checkpoint_path=checkpoint, diagnostics=True, embeddings_path=embeddings)
    state = torch.load(checkpoint, weights_only=True)
    assert state['status'] == 'partial' and state['benchmark_valid'] is False
    assert [r['pair_id'] for r in state['records']] == [0, 1]
    control['fail'] = None
    seen.clear()
    resumed = run(config, checkpoint_path=checkpoint, resume=True, diagnostics=True, embeddings_path=embeddings)
    assert seen == [2, 3]
    assert resumed['full_benchmark_completed'] and resumed['benchmark_valid']
    with np.load(embeddings) as actual, np.load(tmp_path / 'reference.npz') as expected:
        for key in ('test', 'reliability'):
            np.testing.assert_array_equal(actual[key], expected[key])
    for actual, expected in zip(resumed['pairs'], reference['pairs']):
        for key in expected:
            assert actual[key] == expected[key]
        assert actual['num_nodes'] == [3, 3] and actual['num_edges'] == [4, 4]
        assert actual['pair_elapsed_sec'] >= 0
        assert actual['sequential_pair_index'] == actual['pair_id'] + 1
    # Failure after the last checkpoint but before final JSON: no training repeats.
    seen.clear()
    again = run(config, checkpoint_path=checkpoint, resume=True, embeddings_path=embeddings)
    assert seen == [] and again['pairs'] == resumed['pairs']


def test_range_and_embedding_resume(harness, tmp_path):
    config, seen, control = harness
    checkpoint = tmp_path / 'range.partial.pt'
    embeddings = tmp_path / 'range.npz'
    control['fail'] = 2
    with pytest.raises(RuntimeError):
        run(config, start_pair=1, end_pair=3, checkpoint_path=checkpoint, embeddings_path=embeddings)
    seen.clear()
    control['fail'] = None
    result = run(config, start_pair=1, end_pair=3, checkpoint_path=checkpoint,
                 embeddings_path=embeddings, resume=True)
    assert seen == [2]
    assert result['pair_range'] == [1, 3]
    assert result['evaluated_pairs'] == 2 and result['development_subset']
    assert result['primary_metric'] is None and not result['full_benchmark_completed']
    with np.load(embeddings) as arrays:
        assert arrays['pair_ids'].tolist() == [1, 2]
        assert arrays['test'].shape == (2, 64, 16)
    with pytest.raises(ValueError):
        validate_run(result)


@pytest.mark.parametrize('change', ['config', 'seed', 'range', 'embeddings', 'threads', 'protocol', 'source'])
def test_identity_mismatch(harness, tmp_path, change):
    config, seen, control = harness
    checkpoint = tmp_path / 'run.partial.pt'
    control['fail'] = 1
    with pytest.raises(RuntimeError):
        run(config, checkpoint_path=checkpoint)
    kwargs = {'checkpoint_path': checkpoint, 'resume': True}
    altered = copy.deepcopy(config)
    if change == 'config':
        altered['model']['num_heads'] = 8
    elif change == 'range':
        kwargs['end_pair'] = 3
    elif change == 'embeddings':
        kwargs['embeddings_path'] = tmp_path / 'different.npz'
    elif change in ('seed', 'threads'):
        state = torch.load(checkpoint, weights_only=True)
        state['identity'][change] += 1
        torch.save(state, checkpoint)
    elif change in ('protocol', 'source'):
        state = torch.load(checkpoint, weights_only=True)
        state['identity']['protocol' if change == 'protocol' else 'source_sha256'] = 'changed'
        torch.save(state, checkpoint)
    seen.clear()
    with pytest.raises(ValueError, match='mismatch'):
        run(altered, **kwargs)
    assert seen == []


@pytest.mark.parametrize('start,end', [(-1, 2), (2, 2), (3, 2), (0, 5)])
def test_bad_range(harness, start, end):
    with pytest.raises(ValueError, match='range'):
        run(harness[0], start_pair=start, end_pair=end)


def test_checkpoint_not_overwritten_without_resume(harness, tmp_path):
    config, _, _ = harness
    path = tmp_path / 'run.partial.pt'
    run(config, checkpoint_path=path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        run(config, checkpoint_path=path)
    assert path.read_bytes() == original


def test_runner_range_publication_and_resume(harness, tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('checkpoint_runner', scripts / 'reproduce_wp3_3_graphgps_brec.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    config, seen, control = harness
    control['fail'] = 2
    with pytest.raises(RuntimeError):
        cli.run_checkpointed(config, 0, tmp_path, start_pair=1, end_pair=3)
    assert not list(tmp_path.glob('*.json'))
    assert len(list(tmp_path.glob('*.partial.pt'))) == 1
    control['fail'] = None
    seen.clear()
    path = cli.run_checkpointed(config, 0, tmp_path, start_pair=1, end_pair=3, resume=True)
    assert seen == [2] and '_range1_3' in path.name
    assert not list(tmp_path.glob('*.partial.pt'))
    seen.clear()
    assert cli.run_checkpointed(config, 0, tmp_path, start_pair=1, end_pair=3, resume=True) == path
    assert seen == []
