import json
import random

import pytest
import torch
from torch_geometric.data import Batch, Data

from gnn_expressivity.analysis.wp5_permutation import permutation, relabel, small_graphs
from gnn_expressivity.analysis.wp5_perturbation import (
    compare, difference, graph_stats, lap_sensitivity, perturb, run_diagnostics,
    save_diagnostics, simple_edges, spectral_metadata,
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


@pytest.mark.parametrize('family', ['path','cycle','complete','regular','star','disconnected'])
@pytest.mark.parametrize('kind,count', [('add',1),('remove',1),('edits',3)])
def test_valid_exact_edits(family, kind, count):
    graph = dict(small_graphs())[family]
    graph.x = torch.arange(graph.num_nodes).float().reshape(-1,1)
    graph.y = torch.tensor([4])
    before = graph.clone()
    changed, audit = perturb(graph, kind, count, 23)
    if family == 'complete' and kind == 'add':
        assert audit['status'] == 'infeasible'
        assert audit['edit_count'] == 0
    else:
        assert audit['status'] == 'applied'
        old, new = simple_edges(graph), simple_edges(changed)
        assert len(old ^ new) == count
        assert audit['edit_count'] == count
        assert new-old == {tuple(e) for e in audit['added_edges']}
        assert old-new == {tuple(e) for e in audit['removed_edges']}
        assert audit['after']['num_edges'] - audit['before']['num_edges'] == len(new)-len(old)
        assert sum(audit['degree_changes']) == 2*(len(new)-len(old))
        assert audit['density_change'] == pytest.approx(2*(len(new)-len(old))/(graph.num_nodes*(graph.num_nodes-1)))
    assert changed.num_nodes == graph.num_nodes
    assert torch.equal(changed.x, graph.x)
    assert torch.equal(changed.y, graph.y)
    assert torch.equal(graph.edge_index, before.edge_index)


def test_seed_determinism_and_rng_isolation():
    graph = dict(small_graphs())['path_large']
    python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    a, am = perturb(graph,'edits',4,8)
    b, bm = perturb(graph,'edits',4,8)
    assert torch.equal(a.edge_index,b.edge_index) and am == bm
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(),torch_state)
    candidates = {tuple(map(tuple, perturb(graph,'add',1,seed)[1]['added_edges'])) for seed in range(10)}
    assert len(candidates) > 1


def test_noop_infeasible_isolates_and_disconnect():
    graph = Data(num_nodes=1, edge_index=torch.empty(2,0,dtype=torch.long))
    for kind in ('add','remove','edits'):
        changed, meta = perturb(graph,kind,1)
        assert meta['status'] == 'infeasible' and meta['edit_count'] == 0
        assert torch.equal(changed.edge_index,graph.edge_index)
    _, meta = perturb(graph,count=0)
    assert meta['status'] == 'noop'
    assert meta['before']['density'] == 0
    path = dict(small_graphs())['path']
    _, meta = perturb(path,'remove')
    assert meta['after']['components'] == meta['before']['components']+1
    empty = Data(num_nodes=3,edge_index=torch.empty(2,0,dtype=torch.long))
    changed, _ = perturb(empty,'add')
    assert graph_stats(changed)['components'] == 2
    changed, _ = perturb(path,'remove',4)
    assert changed.edge_index.shape == (2,0)


@pytest.mark.parametrize('edges', [torch.tensor([[0],[1]]),torch.tensor([[0,0,1,1],[1,1,0,0]]),torch.tensor([[0],[0]]),torch.tensor([[0,4],[4,0]])])
def test_reject_invalid_graphs(edges):
    with pytest.raises(ValueError):
        perturb(Data(num_nodes=3,edge_index=edges))


@pytest.mark.parametrize('kwargs',[dict(count=True),dict(count=-1),dict(count=1.2),dict(seed=True),dict(seed=-1),dict(kind='bad')])
def test_invalid_edit_parameters(kwargs):
    with pytest.raises(ValueError):
        perturb(dict(small_graphs())['path'],**kwargs)


def test_edge_attributes_and_batch_rejected():
    graph = dict(small_graphs())['path']
    batch = Batch.from_data_list([graph,graph])
    with pytest.raises(ValueError,match='Batch'):
        perturb(batch)
    graph.edge_attr = torch.ones(graph.num_edges,1)
    with pytest.raises(ValueError,match='attribute'):
        perturb(graph)


def test_metrics_exact_values_and_failures():
    before = torch.tensor([3.,4.])
    after = torch.tensor([6.,8.])
    result = difference(before,after)
    assert result['l2_change'] == 5
    assert result['normalized_l2_change'] == 1
    assert result['max_absolute_error'] == 4
    assert result['relative_error'] == 1
    assert result['status'] == 'diagnostic'
    assert difference(before,before,control=True)['status'] == 'pass'
    assert difference(before,after,control=True)['status'] == 'fail'
    assert difference(torch.zeros(2),torch.ones(2))['normalized_l2_change'] is None
    assert difference(before,torch.tensor([float('nan'),4.]))['status'] == 'fail'
    assert difference(before,torch.ones(3))['status'] == 'fail'


def test_lappe_sign_rotation_and_truncation():
    graph = dict(small_graphs())['cycle']
    encoder = build_encoding(dict(name='lappe',k=2))
    encoded = encoder(graph)
    rotated = encoded.clone()
    rotated.x[:,-2:] = rotated.x[:,-2:] @ torch.tensor([[0.,-1.],[1.,0.]])
    metrics = lap_sensitivity(graph,graph,encoded,rotated,2)
    assert metrics['projector']['l2_change'] < 1e-6
    assert metrics['raw_coordinates']['l2_change'] > 1
    assert metrics['sign_aligned']['status'] == 'unsupported'
    assert metrics['before']['repeated_or_near_positive_pairs']
    encoder = build_encoding(dict(name='lappe',k=1))
    encoded = encoder(graph)
    metrics = lap_sensitivity(graph,graph,encoded,encoded,1)
    assert metrics['before']['truncated_block']
    assert metrics['projector']['status'] == 'unsupported'
    path = dict(small_graphs())['path']
    encoder = build_encoding(dict(name='lappe',k=2))
    encoded = encoder(path)
    flipped = encoded.clone()
    flipped.x[:,-2:] *= -1
    metrics = lap_sensitivity(path,path,encoded,flipped,2)
    assert metrics['sign_aligned']['l2_change'] < 1e-6
    assert metrics['sign_aligned']['sign_flipped_columns'] == [0,1]


def test_near_degenerate_and_rank_changing_spectra(monkeypatch):
    import gnn_expressivity.analysis.wp5_perturbation as harness
    graph = dict(small_graphs())['path']
    encoded = build_encoding(dict(name='lappe',k=1))(graph)
    monkeypatch.setattr(harness,'laplacian_spectrum',lambda data: torch.tensor([0.,.5,.5+1e-9,1.5,2.],dtype=torch.float64))
    assert spectral_metadata(graph,encoded,1)['truncated_block']
    monkeypatch.undo()
    disconnected = Data(num_nodes=5,edge_index=torch.empty(2,0,dtype=torch.long))
    other = build_encoding(dict(name='lappe',k=1))(disconnected)
    result = lap_sensitivity(graph,disconnected,encoded,other,1)
    assert result['projector']['status'] == 'unsupported'


@pytest.mark.parametrize('name',['gin','graphgps'])
def test_native_controls_and_sensitive_case(name):
    graph = dict(small_graphs())['star']
    original = build_encoding(dict(name='degree'))(graph)
    torch.manual_seed(73)
    model = build_model(dict(model=dict(name=name,hidden_dim=8,num_layers=2,dropout=0.)),in_dim=2).eval()
    assert compare(model,original,original.clone(),control=True)['status'] == 'pass'
    p = permutation(graph.num_nodes,'reverse')
    assert compare(model,original,relabel(original,p),control=True,undo=p)['status'] == 'pass'
    changed, _ = perturb(graph,'remove',1)
    recomputed = build_encoding(dict(name='degree'))(changed)
    result = compare(model,original,recomputed)
    assert result['metrics']['encoding']['l2_change'] > 0
    assert result['metrics']['node_representation']['l2_change'] > 0
    assert result['metrics']['graph_prediction']['l2_change'] > 0
    tagged = original.clone()
    tagged.x[:,0] = torch.arange(graph.num_nodes).float()
    wrong = relabel(tagged,p)
    wrong.x = tagged.x.clone()
    assert compare(model,tagged,wrong,control=True,undo=p)['metrics']['node_representation']['status'] == 'fail'


def test_matrix_and_json(tmp_path):
    state = torch.get_rng_state().clone()
    rows = run_diagnostics(seeds=(0,1))
    assert torch.equal(state,torch.get_rng_state())
    assert not [r for r in rows if r['status'] == 'fail']
    assert {r['model'] for r in rows} == {'gin','graphgps'}
    assert {r['encoding']['name'] for r in rows} == {'none','degree','rwse','lappe'}
    assert {r['feature_mode'] for r in rows} >= {'frozen','recomputed'}
    assert any(r['status'] == 'infeasible' for r in rows)
    assert all(r['expected_failure_detected'] for r in rows if r['test_type'] == 'negative_alignment')
    path = save_diagnostics(rows,tmp_path/'new')
    payload = json.loads(path.read_text())
    assert payload['counts']['pass'] > 0
    assert len(payload['rows']) == len(rows)
    assert payload['environment']['threads'] == 1
    with pytest.raises(FileExistsError):
        save_diagnostics(rows,path.parent)

@pytest.mark.parametrize('field,value', [
    ('batch', torch.tensor([0,0,0,1,1])),
    ('batch', torch.zeros(5,dtype=torch.long)),
    ('ptr', torch.tensor([0,3,5])),
    ('ptr', torch.tensor([0,5])),
    ('x_batch', torch.zeros(5,dtype=torch.long)),
    ('x_ptr', torch.tensor([0,5])),
    ('adj', torch.eye(5)), ('adj_t', torch.eye(5)),
    ('adjacency', torch.eye(5)), ('adjacency_matrix', torch.eye(5)),
    ('adj_cache', torch.eye(5)), ('alternate_edge_index', torch.empty(2,0,dtype=torch.long)),
])
def test_reject_disguised_batch_and_alternate_adjacency(field,value):
    graph = dict(small_graphs())['path']
    graph[field] = value
    for validate in (simple_edges, perturb):
        with pytest.raises(ValueError,match='Batch metadata or alternate adjacency'):
            validate(graph)
    assert torch.equal(graph[field],value)


def test_native_batch_guard_and_plain_data_behavior():
    graph = dict(small_graphs())['path']
    for validate in (simple_edges,perturb):
        with pytest.raises(ValueError,match='Batch'):
            validate(Batch.from_data_list([graph,graph]))
    assert simple_edges(graph) == {(0,1),(1,2),(2,3),(3,4)}
    changed, audit = perturb(graph,'remove',1,0)
    assert simple_edges(graph)-simple_edges(changed) == {tuple(e) for e in audit['removed_edges']}
    assert len(simple_edges(graph)^simple_edges(changed)) == 1
    assert changed.num_nodes == graph.num_nodes


@pytest.mark.parametrize('scale', [1e-8,1e-20])
def test_near_zero_warnings_keep_valid_ratios_and_absolute_l2(scale):
    before = torch.tensor([3*scale,4*scale],dtype=torch.float64)
    after = before*2
    result = difference(before,after)
    assert result['normalized_l2_change'] == pytest.approx(1.)
    assert result['relative_error'] == pytest.approx(1.)
    assert result['l2_change'] == torch.linalg.vector_norm((after-before).double()).item()
    assert result['l2_change'] == pytest.approx(5*scale,rel=1e-12,abs=0)
    assert result['reference_l2'] == pytest.approx(5*scale,rel=1e-12,abs=0)
    assert len(result['normalization_warnings']) == 2
    assert result['status'] == 'diagnostic'
    assert difference(before,before,control=True)['status'] == 'pass'


def test_zero_and_ordinary_reference_scales():
    zero = difference(torch.zeros(2),torch.ones(2))
    assert zero['relative_error'] is None and zero['normalized_l2_change'] is None
    assert len(zero['normalization_warnings']) == 2
    ordinary = difference(torch.tensor([3.,4.]),torch.tensor([6.,8.]))
    assert ordinary['normalization_warnings'] == []
    assert ordinary['l2_change'] == 5 and ordinary['normalized_l2_change'] == 1
    # Tiny nonzero max but normal L2: warn only for the max-based ratio.
    result = difference(torch.full((100,),1e-5),torch.full((100,),2e-5))
    assert len(result['normalization_warnings']) == 1
    assert 'reference_max_abs' in result['normalization_warnings'][0]


def test_underflowing_reference_norm_is_flagged_not_treated_as_zero():
    before = torch.tensor([1e-200],dtype=torch.float64)
    result = difference(before,before)
    assert result['reference_l2'] > 0
    assert result['normalized_l2_change'] == 0.
    assert result['normalization_warnings']
    assert result['l2_change'] == 0.
