import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import GPSConv

from gnn_expressivity.encodings import build_encoding
from gnn_expressivity.models import GIN, GraphGPS, build_model
from gnn_expressivity.training import brec_evaluation as evaluation
from gnn_expressivity.training.config import load_yaml
from gnn_expressivity.training.profiling import count_parameters
from test_gin import path_graph
from test_encodings import permute_graph
from test_brec_evaluation import ToyDataset


def tiny(**kwargs):
    return GraphGPS(**({"in_dim": 3, "hidden_dim": 16, "num_layers": 2,
                       "num_heads": 4, "dropout": 0, "out_dim": 8} | kwargs))


def graphs():
    return [path_graph(n, torch.arange(n * 3).float().reshape(n, 3) / 10)
            for n in (3, 7)]


def test_config_factory_and_count():
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs/models/graphgps.yaml")
    model = GraphGPS.from_config(config, in_dim=3, out_dim=8)
    assert (model.hidden_dim, model.num_layers, model.num_heads, model.dropout, model.pooling) == (128, 4, 4, .1, 'sum')
    assert all(isinstance(layer, GPSConv) and layer.norm1.mode == 'node'
               and layer.conv.eps.requires_grad for layer in model.layers)
    assert count_parameters(model) == count_parameters(build_model(config, in_dim=3, out_dim=8)) > 0
    assert isinstance(build_model({'model': {'name': 'gin'}}), GIN)
    with pytest.raises(ValueError, match='Unsupported model'):
        build_model({'model': {'name': 'unknown'}})


@pytest.mark.parametrize('pooling', ['sum', 'mean', 'max'])
def test_forward_isolation_permutation_and_determinism(pooling):
    torch.manual_seed(3)
    model = tiny(pooling=pooling).eval()
    a, b = graphs()
    batch = Batch.from_data_list([a, b])
    result = model(batch)
    assert result.shape == (2, 8) and torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.cat([model(a), model(b)]))
    torch.testing.assert_close(result, model(batch))
    changed = b.clone()
    changed.x = changed.x * 100 - 50
    torch.testing.assert_close(result[:1], model(Batch.from_data_list([a, changed]))[:1])
    torch.testing.assert_close(model(a), model(permute_graph(a, torch.tensor([2, 0, 1]))))
    torch.testing.assert_close(model.head(model.encode(batch)), result)


@pytest.mark.parametrize('train_eps', [True, False])
def test_gradients_and_reset(train_eps):
    model = tiny(train_eps=train_eps)
    batch = Batch.from_data_list(graphs())
    model(batch).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(layer.conv.eps.requires_grad == train_eps for layer in model.layers)
    before = model.layers[0].attn.out_proj.weight.detach().clone()
    model.reset_parameters()
    assert not torch.equal(before, model.layers[0].attn.out_proj.weight)
    assert torch.isfinite(model(batch)).all()


@pytest.mark.parametrize('name', ['none', 'degree', 'random', 'uid', 'rwse', 'lappe'])
def test_external_encodings(name):
    encoder = build_encoding({'name': name, 'walk_length': 4, 'k': 4})
    batch = Batch.from_data_list([encoder(path_graph(n)) for n in (3, 5)])
    model = tiny(in_dim=batch.x.shape[1])
    assert model.input_projection.in_features == batch.x.shape[1]
    result = model(batch)
    assert result.shape == (2, 8) and torch.isfinite(result).all()


@pytest.mark.parametrize('settings', [
    {'hidden_dim': 0}, {'hidden_dim': True}, {'num_layers': 0}, {'num_layers': 1.5},
    {'num_heads': 0}, {'num_heads': True}, {'num_heads': 3}, {'in_dim': 0},
    {'out_dim': 0}, {'dropout': -.1}, {'dropout': 1.0}, {'dropout': float('nan')},
    {'dropout': '0.1'}, {'pooling': 'unknown'}, {'activation': 'gelu'}, {'train_eps': 1},
])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        tiny(**settings)


def test_invalid_features_and_cross_graph_edges():
    with pytest.raises(ValueError, match='x must'):
        tiny()(path_graph())
    batch = Batch.from_data_list(graphs())
    batch.edge_index[:, 0] = torch.tensor([0, 4])
    with pytest.raises(ValueError, match='different graphs'):
        tiny()(batch)


@pytest.mark.parametrize('name,width', [('none', 1), ('rwse', 5), ('lappe', 5)])
def test_evaluator_metadata_seed_and_parameters(monkeypatch, name, width):
    class Dataset(ToyDataset):
        sha256 = 'a' * 64

        def __len__(self):
            return 400

    monkeypatch.setattr(evaluation.BRECDataset, 'from_config', lambda config: Dataset())
    monkeypatch.setattr(evaluation, 'validate_brec', lambda data: None)
    monkeypatch.setattr(evaluation, 'get_device', lambda: torch.device('cpu'))
    seed = Mock(wraps=evaluation.set_seed)
    monkeypatch.setattr(evaluation, 'set_seed', seed)
    config = {'model': {'name': 'graphgps', 'hidden_dim': 8, 'num_layers': 1,
                        'num_heads': 2, 'dropout': 0},
              'encoding': {'name': name, 'walk_length': 4, 'k': 4},
              'task': {'name': 'brec', 'dataset': 'BREC', 'batch_size': 32},
              'training': {'epochs': 1, 'learning_rate': .001, 'weight_decay': 0}}
    result = evaluation.evaluate_graphgps_brec(config, seed=0, max_pairs=2, progress=lambda msg: None)
    seed.assert_called_once_with(0)
    assert result['model'] == 'graphgps' and result['encoding'] == name
    assert result['run_id'] == f'brec_graphgps_{name}_seed0'
    assert result['evaluated_pairs'] == 2 and result['development_subset'] is True
    assert result['benchmark_valid'] is False
    assert result['num_parameters'] == count_parameters(build_model(config, in_dim=width, out_dim=16))


def test_cli_safeguards(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('graphgps_cli', scripts / 'reproduce_graphgps_brec.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    counts = {'pairs': 2, 'distinguished': 0, 'distinction_rate': 0., 'reliability_failures': 0}
    evaluator = Mock(return_value={'run_id': 'unused', 'by_category': {}, 'overall': counts,
        'evaluated_pairs': 2, 'full_benchmark_pairs': 400, 'development_subset': True,
        'benchmark_valid': False, 'embeddings_file': None,
        'evaluation_time_sec': 1., 'process_memory_mb': 1.})
    monkeypatch.setattr(cli, 'evaluate_graphgps_brec', evaluator)
    monkeypatch.setattr('sys.argv', ['script', '--seed', '0', '--max-pairs', '2',
                                    '--encoding', 'rwse', '--results-dir', str(tmp_path)])
    cli.main()
    assert evaluator.call_args.kwargs['seed'] == 0
    assert evaluator.call_args.kwargs['max_pairs'] == 2
    assert evaluator.call_args.args[0]['model']['name'] == 'graphgps'
    assert evaluator.call_args.args[0]['encoding']['name'] == 'rwse'
    assert evaluator.call_args.kwargs['embeddings_path'].name == 'brec_graphgps_rwse_seed0_first2_embeddings.npz'
    path = tmp_path / 'brec_graphgps_rwse_seed0_first2.json'
    assert json.loads(path.read_text())['run_id'] == path.stem
    with pytest.raises(FileExistsError):
        cli.main()
    assert evaluator.call_count == 1
