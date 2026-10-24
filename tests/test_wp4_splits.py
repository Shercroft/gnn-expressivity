import copy
import json

import pytest

from gnn_expressivity.data.controlled_splits import (
    GraphRecord, approve_manifest, create_manifest, dataset_hash, digest,
    nested_training_subsets, validate_manifest, validate_records, write_manifest,
)


@pytest.fixture
def records():
    return [GraphRecord(f'g{i}', i+3, tuple((j,j+1) for j in range(i+2)),
        'A' if i < 15 else 'B', i%2, i, 'fixture-v1', {'kind':'test path'}) for i in range(20)]


@pytest.fixture
def manifest(records):
    draft = create_manifest(records, task_id='fixture', dataset_version='fixture-v1', split_type='iid', seed=7, stratify=True)
    return approve_manifest(draft, records, approved_by='synthetic test only', reference='fixture approval; not a scientific task')


def test_density_and_edge_schema():
    undirected = GraphRecord('u',3,((0,1),),'A',None,0,'v',{})
    directed = GraphRecord('d',3,((0,1),),'A',None,0,'v',{},directed=True)
    assert undirected.graph_density == pytest.approx(1/3)
    assert directed.graph_density == pytest.approx(1/6)
    with pytest.raises(ValueError, match='Duplicate'):
        GraphRecord('x',3,((0,1),(1,0)),'A',None,0,'v',{})
    with pytest.raises(ValueError, match='loop'):
        GraphRecord('x',3,((0,0),),'A',None,0,'v',{})
    assert GraphRecord.from_dict(undirected.to_dict()) == undirected


def test_iid_reproducible_membership_and_strata(records, manifest):
    validate_manifest(manifest, records, require_approved=True)
    draft = create_manifest(list(reversed(records)),task_id='fixture',dataset_version='fixture-v1',split_type='iid',seed=7,stratify=True)
    assert draft['splits'] == manifest['splits']
    ids = [i for values in manifest['splits'].values() for i in values]
    assert len(ids) == len(set(ids)) == 20
    assert set(manifest['observed']['train']['label_distribution'].values()) == {6}
    assert dataset_hash(records) == dataset_hash(list(reversed(records)))


@pytest.mark.parametrize('kind,regimes', [
    ('ood_size', {'id':[3,16], 'ood':[16,24]}),
    ('ood_family', {'id':['A'], 'ood':['B']}),
    ('ood_density', {'id':[0.1,0.5], 'ood':[0.5,1.01]}),
])
def test_ood_regimes(records, kind, regimes):
    value = create_manifest(records,task_id='fixture',dataset_version='v',split_type=kind,seed=3,regimes=regimes)
    validate_manifest(value, records)
    lookup = {r.graph_id:r for r in records}
    for i in value['splits'][kind+'_test']:
        record = lookup[i]
        if kind == 'ood_family':
            assert record.graph_family == 'B'
        else:
            measured = record.num_nodes if kind == 'ood_size' else record.graph_density
            assert regimes['ood'][0] <= measured < regimes['ood'][1]
    assert value['shift_warnings'] and value['observed'][kind+'_test']['count'] > 0
    assert value['validation_domain'] == 'id_only'


def test_overlap_and_empty_regimes_rejected(records):
    for regimes in ({'id':[3,18], 'ood':[17,25]}, {'id':[3,24], 'ood':[50,60]}):
        with pytest.raises(ValueError):
            create_manifest(records,task_id='x',dataset_version='v',split_type='ood_size',seed=0,regimes=regimes)


def test_id_and_isomorphic_duplicate_rejection(records):
    with pytest.raises(ValueError, match='unique graph IDs'):
        validate_records([records[0],records[0]])
    graph = GraphRecord('original',4,((0,1),(1,2),(2,3)),'A',0,0,'v',{})
    relabelled = GraphRecord('relabelled',4,((1,0),(0,2),(2,3)),'B',1,1,'v',{})
    assert graph.edges != relabelled.edges
    with pytest.raises(ValueError,match='Isomorphic'):
        validate_records([graph,relabelled])
    with pytest.raises(ValueError,match='budget'):
        validate_records([graph,relabelled],max_isomorphism_checks=0)


def test_tamper_and_immutable_manifest(records, manifest, tmp_path):
    path = tmp_path/'manifest.json'
    write_manifest(manifest,path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_manifest(manifest,path)
    assert path.read_bytes() == before
    changed = copy.deepcopy(manifest)
    changed['splits']['train'][0] = changed['splits']['iid_test'][0]
    with pytest.raises(ValueError,match='modified'):
        validate_manifest(changed, records)
    changed.pop('manifest_hash')
    changed['manifest_hash'] = digest(changed)
    with pytest.raises(ValueError,match='overlapping'):
        validate_manifest(changed,records)
    changed = list(records)
    changed[0] = GraphRecord.from_dict(records[0].to_dict() | {'target':99})
    with pytest.raises(ValueError,match='hash'):
        validate_manifest(manifest,changed)


def test_draft_gate_and_nested_training_only(records,manifest):
    draft = create_manifest(records,task_id='x',dataset_version='v',split_type='iid',seed=0)
    with pytest.raises(ValueError,match='Approved'):
        validate_manifest(draft,records,require_approved=True)
    subsets = nested_training_subsets(manifest,records,[2,6,12],seed=4,balanced=True)
    assert subsets == nested_training_subsets(manifest,records,[2,6,12],seed=4,balanced=True)
    assert set(subsets[0]['train_ids']) < set(subsets[1]['train_ids']) < set(subsets[2]['train_ids'])
    assert all(set(s['train_ids']) <= set(manifest['splits']['train']) for s in subsets)
    assert len({s['fixed_evaluation_hash'] for s in subsets}) == 1
    assert all(max(s['label_distribution'].values()) - min(s['label_distribution'].values()) <= 1 for s in subsets)
    for sizes in ([0], [2,2], [100], [6,2]):
        with pytest.raises(ValueError):
            nested_training_subsets(manifest,records,sizes,seed=0)
