from collections import Counter
import json

import pytest
import torch
from torch_geometric.data import Batch, Data

from gnn_expressivity.analysis.wp5_permutation import (
    error_record, lappe_checks, model_checks, native_outputs, permutation,
    relabel, run_diagnostics, small_graphs, validate_permutation, write_diagnostics,
)
from gnn_expressivity.encodings import build_encoding
from gnn_expressivity.models import build_model


@pytest.fixture(scope='module', autouse=True)
def one_thread():
    from threadpoolctl import threadpool_limits
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('bad', [[0, 0, 2], [-1, 1, 2], [0, 1, 3], [0, 1], [0., 1., 2.], [True, False, True]])
def test_invalid_permutations(bad):
    with pytest.raises(ValueError, match='bijection'):
        validate_permutation(bad, 3)


@pytest.mark.parametrize('kind', ['identity', 'reverse', 'random'])
def test_local_deterministic_permutations(kind):
    state = torch.get_rng_state().clone()
    p = permutation(7, kind, 9)
    assert torch.equal(p, permutation(7, kind, 9))
    validate_permutation(p, 7)
    assert torch.equal(state, torch.get_rng_state())


def test_attribute_alignment_and_inverse():
    graph = Data(num_nodes=4, x=torch.arange(8.).reshape(4, 2),
                 edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                 edge_attr=torch.arange(4.).reshape(-1, 1),
                 pos=torch.arange(12.).reshape(4, 3), node_mask=torch.tensor([True, False, True, False]),
                 y=torch.tensor([6]), label='graph-level', custom=torch.arange(4))
    p = torch.tensor([2, 0, 3, 1])
    changed = relabel(graph, p, node_attrs=('custom', 'node_mask'))
    for key in ('x', 'pos', 'custom', 'node_mask'):
        assert torch.equal(changed[key][p], graph[key])
    assert torch.equal(changed.edge_index, p[graph.edge_index])
    assert torch.equal(changed.edge_attr, graph.edge_attr)
    assert changed.label == graph.label
    assert torch.equal(changed.y, graph.y)
    restored = relabel(changed, p.argsort(), node_attrs=('custom', 'node_mask'))
    for key in ('x', 'pos', 'custom', 'node_mask', 'edge_index', 'edge_attr'):
        assert torch.equal(restored[key], graph[key])
    changed.x.zero_()
    assert graph.x.sum() > 0


def test_node_labels_explicit_and_unsupported_indices():
    graph = Data(num_nodes=3, edge_index=torch.empty(2, 0, dtype=torch.long), y=torch.arange(3))
    p = permutation(3, 'reverse')
    assert torch.equal(relabel(graph, p).y, graph.y)
    assert torch.equal(relabel(graph, p, node_attrs=('y',), graph_attrs=()).y[p], graph.y)
    graph.face = torch.tensor([[0], [1], [2]])
    with pytest.raises(ValueError, match='index-bearing'):
        relabel(graph, p)


@pytest.mark.parametrize('name', ['gin', 'graphgps'])
def test_native_batch_invariance_and_isolation(name):
    graphs = [build_encoding({'name':'degree'})(graph) for _, graph in list(small_graphs())[:2]]
    batch = Batch.from_data_list(graphs)
    p = torch.cat([permutation(5, 'reverse'), permutation(8, 'random', 8) + 5])
    changed = relabel(batch, p)
    assert torch.equal(changed.ptr, batch.ptr)
    for old, new in zip(graphs, changed.to_data_list()):
        assert old.num_nodes == new.num_nodes
    config = dict(model=dict(name=name, hidden_dim=8, num_layers=1, dropout=0.))
    model = build_model(config, in_dim=2, out_dim=2).eval()
    assert all(v['status'] == 'pass' for v in model_checks(model, batch, changed, p).values())
    with torch.no_grad():
        torch.testing.assert_close(model(batch), torch.cat([model(g) for g in graphs]), atol=2e-5, rtol=2e-5)
    bad = torch.arange(batch.num_nodes)
    bad[0], bad[5] = 5, 0
    with pytest.raises(ValueError, match='within each graph'):
        relabel(batch, bad)


@pytest.mark.parametrize('name', ['gin', 'graphgps'])
def test_negative_controls(name):
    graph = dict(small_graphs())['star']
    graph.x = torch.arange(1., graph.num_nodes + 1).reshape(-1, 1)
    p = permutation(graph.num_nodes, 'reverse')
    torch.manual_seed(78)
    model = build_model(dict(model=dict(name=name, hidden_dim=8, num_layers=2, dropout=0.)), in_dim=1).eval()
    correct = relabel(graph, p)
    assert all(v['status'] == 'pass' for v in model_checks(model, graph, correct, p).values())
    wrong = correct.clone()
    wrong.x = graph.x.clone()  # deliberately leaves features at old array positions
    assert model_checks(model, graph, wrong, p)['node_equivariance']['status'] == 'fail'
    assert model_checks(model, graph, wrong, p)['graph_invariance']['status'] == 'fail'
    wrong_edges = correct.clone()
    wrong_edges.edge_index = graph.edge_index.clone()
    assert model_checks(model, graph, wrong_edges, p)['node_equivariance']['status'] == 'fail'
    before = len(model.layers[-1]._forward_hooks) if name == 'graphgps' else len(model.dropout._forward_hooks)
    native_outputs(model, graph)
    after = len(model.layers[-1]._forward_hooks) if name == 'graphgps' else len(model.dropout._forward_hooks)
    assert before == after == 0


def test_nonfinite_and_shape_failures():
    assert error_record(torch.ones(2), torch.tensor([1., float('nan')]), diagnostic=True)['status'] == 'fail'
    assert error_record(torch.ones(2), torch.ones(3))['status'] == 'fail'
    assert error_record(torch.zeros(2), torch.zeros(2))['relative_error'] is None


def test_lappe_complete_and_truncated_repeated_blocks():
    # C6 positive spectrum: .5,.5,1.5,1.5,2; k=1 cuts a repeated block.
    graph = dict(small_graphs())['cycle']
    p = permutation(graph.num_nodes, 'random', 7)
    for k, status in [(1, 'unsupported'), (2, 'pass'), (4, 'pass')]:
        encoder = build_encoding(dict(name='lappe', k=k))
        checks = lappe_checks(graph, encoder(graph), encoder(relabel(graph, p)), p, k)
        assert checks['lappe_projector']['status'] == status
        assert checks['lappe_raw_recomputation']['status'] == 'diagnostic'
    # Rotation within an entire repeated block keeps its projector, not columns.
    encoder = build_encoding(dict(name='lappe', k=2))
    original = encoder(graph)
    rotated = relabel(original, p)
    rotated.x[:, -2:] = rotated.x[:, -2:] @ torch.tensor([[0., -1.], [1., 0.]])
    checks = lappe_checks(graph, original, rotated, p, 2)
    assert checks['lappe_projector']['status'] == 'pass'
    assert checks['lappe_raw_recomputation']['max_absolute_error'] > .1


@pytest.mark.parametrize('name', ['uid', 'random'])
def test_carried_vs_recreated_index_tied_features(name):
    graph = dict(small_graphs())['path']
    encoder = build_encoding(dict(name=name))
    encoded = encoder(graph)
    p = permutation(graph.num_nodes, 'reverse')
    assert torch.equal(relabel(encoded, p).x[p], encoded.x)
    assert not torch.equal(encoder(relabel(graph, p)).x[p], encoded.x)
    if name == 'random':
        independent = build_encoding(dict(name='random', seed=17))(relabel(graph, p))
        assert not torch.equal(independent.x[p], encoded.x)


def test_full_lightweight_matrix_and_machine_report(tmp_path):
    state = torch.get_rng_state().clone()
    rows = run_diagnostics(seeds=(0, 1))
    assert torch.equal(state, torch.get_rng_state())
    assert not [row for row in rows if row['status'] == 'fail']
    assert {row['model'] for row in rows} == {'encoding', 'gin', 'graphgps'}
    assert {row['encoding'] for row in rows} == {'none','degree','rwse','lappe','uid','random'}
    assert len({row['graph_family'] for row in rows}) == 7
    assert any(row['status'] == 'unsupported' for row in rows)
    path = write_diagnostics(rows, tmp_path / 'diagnostics')
    payload = json.loads(path.read_text())
    assert payload['counts'] == dict(Counter(row['status'] for row in rows))
    assert len(payload['diagnostics']) == len(rows)
    assert all({'model','encoding','graph_family','num_nodes','permutation_seed','test_type',
                'max_absolute_error','relative_error','status','tolerance','caveat'} <= row.keys() for row in rows)
    with pytest.raises(FileExistsError):
        write_diagnostics(rows, path.parent)

def test_near_repeated_boundary_is_not_certified(monkeypatch):
    import gnn_expressivity.analysis.wp5_permutation as harness
    graph = dict(small_graphs())['path']
    p = permutation(5, 'reverse')
    encoder = build_encoding(dict(name='lappe', k=1))
    monkeypatch.setattr(harness, 'laplacian_spectrum', lambda data: torch.tensor([0., .5, .5 + 1e-9, 1.5, 2.], dtype=torch.float64))
    checks = lappe_checks(graph, encoder(graph), encoder(relabel(graph, p)), p, 1)
    assert checks['lappe_projector']['status'] == 'unsupported'


def test_eval_required_and_unknown_permutation():
    graph = build_encoding({'name':'none'})(dict(small_graphs())['path'])
    model = build_model(dict(model=dict(name='gin', hidden_dim=4, num_layers=1, dropout=0.)), in_dim=1)
    with pytest.raises(ValueError, match='eval'):
        native_outputs(model, graph)
    with pytest.raises(ValueError, match='unknown'):
        permutation(3, 'invalid')
    with pytest.raises(ValueError, match='positive'):
        permutation(True)
