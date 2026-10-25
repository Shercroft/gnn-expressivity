import copy
import json
import math
import random

import pytest

from gnn_expressivity.data.controlled_splits import approve_manifest, create_manifest
from gnn_expressivity.training.controlled_task_runner import TaskSpec, metrics, run_experiment, validate_result
from gnn_expressivity.analysis.wp4_generalization import analyze, save_analysis
from test_wp4_splits import records, manifest  # noqa: F401


@pytest.fixture
def config():
    return dict(model={'name':'gin','hidden_dim':8,'num_layers':1,'dropout':0.0,'activation':'relu','pooling':'sum','train_eps':True},
        encoding={'name':'none'}, node_features='constant',
        training=dict(epochs=3,batch_size=2,patience=2,min_delta=0.0,learning_rate=0.001,weight_decay=0.0,optimizer='Adam'))


@pytest.fixture
def task():
    return TaskSpec('fixture','classification','synthetic parity labels only','test fixture approval',2)


class FakeBackend:
    fail_epoch = None
    events = []
    parameter_count = 17

    def __init__(self, records, task, config, seed):
        self.lookup = {r.graph_id:r for r in records}
        self.task, self.rng, self.step = task, random.Random(seed), 0

    def epoch(self, ids):
        self.rng.random()
        if self.step+1 == self.fail_epoch:
            raise RuntimeError('simulated epoch interruption')
        self.step += 1
        type(self).events.append(('train',self.step,tuple(ids)))
        return self.rng.random()

    def predict(self, ids):
        type(self).events.append(('predict',self.step,tuple(ids)))
        if self.task.task_type == 'regression':
            return [self.lookup[i].target + self.step*.1 for i in ids]
        return [[self.step+self.rng.random()*.01 if c == self.lookup[i].target else 0.0 for c in range(self.task.num_classes)] for i in ids]

    def state(self):
        return dict(step=self.step,rng=self.rng.getstate())

    def restore(self,state):
        self.step = state['step']
        self.rng.setstate(state['rng'])


def test_metric_correctness():
    task = TaskSpec('t','classification','fixture','fixture',2)
    value = metrics(task,[0,0,1,1],[[3,0],[0,3],[0,3],[0,3]])
    assert value['accuracy'] == value['balanced_accuracy'] == .75
    assert value['auroc'] == .75
    assert metrics(task,[0,0],[[1,0],[1,0]])['auroc'] is None
    regression = TaskSpec('t','regression','fixture','fixture')
    value = metrics(regression,[1,3],[2,1])
    assert value['mae'] == 1.5 and value['rmse'] == pytest.approx(math.sqrt(2.5))
    with pytest.raises(ValueError):
        metrics(task,[-1],[[1,0]])


def test_validation_only_selection_and_ood(records,config,task,tmp_path):
    draft = create_manifest(records,task_id='fixture',dataset_version='v',split_type='ood_size',seed=0,regimes={'id':[3,16],'ood':[16,24]})
    manifest = approve_manifest(draft,records,approved_by='fixture',reference='test only')
    FakeBackend.events = []
    result = run_experiment(records,manifest,task,config,0,tmp_path,backend_factory=FakeBackend)
    assert result['experiment_valid'] and set(result['performance']) == {'iid_test','ood_size_test'}
    assert result['selected_epoch'] == 3
    assert result['ood_minus_iid']['ood_size_test']['accuracy'] == 0
    first_test = next(i for i,event in enumerate(FakeBackend.events) if event[0]=='predict' and set(event[2]) == set(manifest['splits']['ood_size_test']))
    assert not any(event[0]=='train' for event in FakeBackend.events[first_test:])
    assert all(set(event[2]) == set(manifest['splits']['validation']) for event in FakeBackend.events[:first_test-1] if event[0]=='predict')
    validate_result(result)


def test_seed_reproducibility_resume_and_overwrite(records,manifest,task,config,tmp_path):
    first = run_experiment(records,manifest,task,config,2,tmp_path/'a',backend_factory=FakeBackend)
    second = run_experiment(records,manifest,task,config,2,tmp_path/'b',backend_factory=FakeBackend)
    assert first['performance'] == second['performance'] and first['history'] == second['history']
    with pytest.raises(FileExistsError):
        run_experiment(records,manifest,task,config,2,tmp_path/'a',backend_factory=FakeBackend)
    assert run_experiment(records,manifest,task,config,2,tmp_path/'a',resume=True,backend_factory=FakeBackend) == first
    FakeBackend.fail_epoch = 2
    try:
        with pytest.raises(RuntimeError,match='interruption'):
            run_experiment(records,manifest,task,config,2,tmp_path/'c',backend_factory=FakeBackend)
    finally:
        FakeBackend.fail_epoch = None
    with pytest.raises(FileExistsError):
        run_experiment(records,manifest,task,config,2,tmp_path/'c',backend_factory=FakeBackend)
    resumed = run_experiment(records,manifest,task,config,2,tmp_path/'c',resume=True,backend_factory=FakeBackend)
    assert resumed['performance'] == first['performance'] and resumed['history'] == first['history']


def test_approval_targets_and_subset_gates(records,manifest,task,config,tmp_path):
    from gnn_expressivity.data.controlled_splits import create_manifest
    draft = create_manifest(records,task_id='fixture',dataset_version='v',split_type='iid',seed=0)
    with pytest.raises(ValueError,match='Approved'):
        run_experiment(records,draft,task,config,0,tmp_path,backend_factory=FakeBackend)
    with pytest.raises(ValueError,match='subset'):
        run_experiment(records,manifest,task,config,0,tmp_path,train_ids=manifest['splits']['iid_test'],backend_factory=FakeBackend)
    with pytest.raises(ValueError,match='approval'):
        TaskSpec('fixture','regression','labels','').validate(records)
    with pytest.raises(ValueError,match='classes'):
        TaskSpec('fixture','classification','labels','fixture',1).validate(records)


def test_analysis_seeds_missing_and_model_separation(records,manifest,task,config,tmp_path):
    results = [run_experiment(records,manifest,task,config,s,tmp_path,backend_factory=FakeBackend) for s in (0,2)]
    rows, summary = analyze(results,expected_seeds=[0,1,2])
    assert summary[0]['n_seeds'] == 2 and json.loads(summary[0]['missing_seeds']) == [1]
    assert summary[0]['sample_std'] == 0
    assert analyze(results[:1])[1][0]['sample_std'] is None
    other = copy.deepcopy(config)
    other['model']['name'] = 'graphgps'
    other['encoding'] = {'name':'lappe','k':4,'eigen_tol':1e-8}
    results.append(run_experiment(records,manifest,task,other,0,tmp_path,backend_factory=FakeBackend))
    assert len({r['condition_id'] for r in analyze(results)[0]}) == 2
    with pytest.raises(ValueError,match='Duplicate'):
        analyze([results[0],results[0]])
    invalid = copy.deepcopy(results[0])
    invalid['experiment_valid'] = False
    with pytest.raises(ValueError,match='Invalid'):
        analyze([invalid])
    paths = save_analysis([],tmp_path/'empty',expected_seeds=[0,1])
    assert 'DATA-BLOCKED' in paths[-1].read_text()
    with pytest.raises(FileExistsError):
        save_analysis([],tmp_path/'empty')


def test_checkpoint_identity_rejection(records,manifest,task,config,tmp_path):
    import torch
    FakeBackend.fail_epoch = 2
    try:
        with pytest.raises(RuntimeError):
            run_experiment(records,manifest,task,config,0,tmp_path,backend_factory=FakeBackend)
    finally:
        FakeBackend.fail_epoch = None
    path = next(tmp_path.glob('*.partial.pt'))
    state = torch.load(path,weights_only=True)
    state['identity']['config']['training']['epochs'] += 1
    torch.save(state,path)
    with pytest.raises(ValueError,match='Checkpoint identity'):
        run_experiment(records,manifest,task,config,0,tmp_path,resume=True,backend_factory=FakeBackend)


def test_regression_early_stopping(records,manifest,config,tmp_path):
    regression = TaskSpec('fixture','regression','synthetic only','fixture approval')
    config['training']['epochs'] = 10
    result = run_experiment(records,manifest,regression,config,0,tmp_path,backend_factory=FakeBackend)
    assert result['selected_epoch'] == 1
    assert len(result['history']) == 3
    assert result['performance']['iid_test']['mae'] == pytest.approx(.1)


@pytest.mark.parametrize('model,encoding', [('gin',{'name':'none'}),('gin',{'name':'degree'}),
    ('gin',{'name':'rwse','walk_length':2}),('gin',{'name':'lappe','k':2,'eigen_tol':1e-8}),
    ('graphgps',{'name':'lappe','k':2,'eigen_tol':1e-8})])
def test_adapter_feature_shapes_without_training(records,config,task,model,encoding,monkeypatch):
    import torch
    from gnn_expressivity import models
    from gnn_expressivity.training.controlled_task_runner import TorchGraphBackend
    calls = []
    class SmallHead(torch.nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.head = torch.nn.Linear(in_dim,out_dim)
        def forward(self,batch):
            pooled = torch.stack([batch.x[batch.batch==i].mean(0) for i in range(batch.num_graphs)])
            return self.head(pooled)
    def factory(config, *, in_dim, out_dim):
        calls.append((config['model']['name'],in_dim,out_dim))
        return SmallHead(in_dim,out_dim)
    monkeypatch.setattr(models,'build_model',factory)
    config['model']['name'], config['encoding'] = model, encoding
    backend = TorchGraphBackend(records[:2],task,config,0)
    predicted = backend.predict([records[0].graph_id,records[1].graph_id])
    assert len(predicted)==2 and all(len(p)==2 for p in predicted)
    assert calls[0][0] == model and calls[0][2] == 2
    assert calls[0][1] == (1 if encoding['name']=='none' else 3 if encoding['name'] in ('rwse','lappe') else 2)


def test_incompatible_split_and_missing_metrics_rejected(records,manifest,task,config,tmp_path):
    from gnn_expressivity.data.controlled_splits import digest
    result = run_experiment(records,manifest,task,config,0,tmp_path,backend_factory=FakeBackend)
    changed = copy.deepcopy(result)
    changed['performance']['iid_test'].pop('accuracy')
    with pytest.raises(ValueError,match='Incomplete'):
        validate_result(changed)
    changed = copy.deepcopy(result)
    changed['identity']['evaluation_ids']['iid_test'][0] = changed['identity']['train_ids'][0]
    changed['identity']['evaluation_hash'] = digest(changed['identity']['evaluation_ids'])
    changed['run_id'] = 'wp4_' + digest(changed['identity'])
    with pytest.raises(ValueError,match='leakage'):
        validate_result(changed)


def test_cli_is_plan_only_by_default(records,manifest,task,config,tmp_path,monkeypatch,capsys):
    import runpy
    import sys
    from dataclasses import asdict
    from pathlib import Path
    from gnn_expressivity.training import controlled_task_runner
    def forbidden(*args, **kwargs):
        pytest.fail('Plan-only CLI attempted training')
    monkeypatch.setattr(controlled_task_runner,'run_experiment',forbidden)
    paths = {}
    for name,value in [('records',[r.to_dict() for r in records]),('manifest',manifest),('task',asdict(task)),('config',config)]:
        paths[name] = tmp_path/(name+'.json')
        paths[name].write_text(json.dumps(value))
    script = Path(__file__).resolve().parents[1]/'scripts/run_wp4_generalization.py'
    output = tmp_path/'results'
    argv = [str(script)]
    for name,path in paths.items():
        argv.extend(['--'+name,str(path)])
    argv.extend(['--seeds','0','1','--output-dir',str(output)])
    monkeypatch.setattr(sys,'argv',argv)
    runpy.run_path(str(script),run_name='__main__')
    assert capsys.readouterr().out.count('PLAN ONLY') == 2
    assert not output.exists()


# Focused review regressions: these fixtures never use an active BREC artifact.
@pytest.mark.parametrize('encoding', [
    *[{'name':'rwse','walk_length':v} for v in (0,-1,True,1.5,float('nan'),float('inf'))],
    *[{'name':'lappe','k':v,'eigen_tol':1e-8} for v in (0,-1,True,1.5,float('nan'),float('inf'))],
    *[{'name':'lappe','k':2,'eigen_tol':v} for v in (0,-100,True,float('nan'),float('inf'),float('-inf'),'invalid')],
    {'name':'none','ignored':1}, {'name':'rwse','walk_length':2,'ignored':1},
])
def test_review_invalid_encoding_config(config,encoding):
    from gnn_expressivity.training.controlled_task_runner import validate_config
    config['encoding'] = encoding
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize('model,field,value', [
    ('gin','hidden_dim',True), ('gin','hidden_dim',0), ('gin','num_layers',1.5),
    ('gin','dropout',True), ('gin','dropout',float('nan')), ('gin','dropout',float('inf')),
    ('gin','activation','unsupported'), ('gin','train_eps',1), ('gin','pooling','unsupported'),
    ('graphgps','num_heads',True), ('graphgps','num_heads',0), ('graphgps','num_heads',3),
    ('gin','ignored',1), ('graphgps','ignored',1), ('gin','out_dim',2), ('gin','in_dim',3),
])
def test_review_invalid_model_config(config,model,field,value):
    from gnn_expressivity.training.controlled_task_runner import validate_config
    config['model']['name'] = model
    config['model'][field] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_review_impossible_classification_metrics(records,manifest,task,config,tmp_path):
    result = run_experiment(records,manifest,task,config,0,tmp_path,backend_factory=FakeBackend)
    for metric in ('accuracy','balanced_accuracy','auroc'):
        for value in (-.01,1.01,float('nan'),float('inf'),True):
            changed = copy.deepcopy(result)
            changed['performance']['iid_test'][metric] = value
            with pytest.raises(ValueError,match='metric'):
                validate_result(changed)
    for value in (-.01,float('nan'),float('inf'),True):
        changed = copy.deepcopy(result)
        changed['performance']['iid_test']['loss'] = value
        with pytest.raises(ValueError,match='metric'):
            validate_result(changed)
    for boundary in (0.0,1.0):
        changed = copy.deepcopy(result)
        for metric in ('accuracy','balanced_accuracy','auroc'):
            changed['performance']['iid_test'][metric] = boundary
        changed['performance']['iid_test']['loss'] = 0.0
        validate_result(changed)
    missing_auc = copy.deepcopy(result)
    missing_auc['performance']['iid_test']['auroc'] = None
    missing_auc['performance']['iid_test']['auroc_unavailable_reason'] = 'single observed class'
    validate_result(missing_auc)


def test_review_impossible_regression_metrics(records,manifest,config,tmp_path):
    task = TaskSpec('fixture','regression','synthetic only','fixture approval')
    result = run_experiment(records,manifest,task,config,0,tmp_path,backend_factory=FakeBackend)
    for metric in ('loss','mae','rmse'):
        for value in (-.01,float('nan'),float('inf'),True,None):
            changed = copy.deepcopy(result)
            changed['performance']['iid_test'][metric] = value
            with pytest.raises(ValueError,match='metric'):
                validate_result(changed)
        changed = copy.deepcopy(result)
        changed['performance']['iid_test'][metric] = 0.0
        validate_result(changed)


def test_review_ood_difference_sign_and_consistency(records,config,task,tmp_path):
    manifest = approve_manifest(create_manifest(records,task_id='fixture',dataset_version='v',
        split_type='ood_size',seed=0,regimes={'id':[3,16],'ood':[16,24]}),
        records,approved_by='fixture',reference='synthetic approval only')
    classification = run_experiment(records,manifest,task,config,0,tmp_path/'classification',backend_factory=FakeBackend)
    regression = run_experiment(records,manifest,TaskSpec('fixture','regression','synthetic','fixture'),
        config,0,tmp_path/'regression',backend_factory=FakeBackend)
    for result, metric, iid, ood in [(classification,'accuracy',.9,.6),(regression,'mae',.2,.5)]:
        result['performance']['iid_test'][metric] = iid
        result['performance']['ood_size_test'][metric] = ood
        result['ood_minus_iid']['ood_size_test'][metric] = ood-iid
        validate_result(result)
        assert result['ood_minus_iid']['ood_size_test'][metric] == pytest.approx(-.3 if metric=='accuracy' else .3)
        result['ood_minus_iid']['ood_size_test'][metric] = iid-ood
        with pytest.raises(ValueError,match='OOD differences'):
            validate_result(result)
    classification['ood_minus_iid']['ood_size_test']['accuracy'] = .6-.9
    classification['performance']['ood_size_test']['auroc'] = None
    classification['performance']['ood_size_test']['auroc_unavailable_reason'] = 'single class'
    classification['ood_minus_iid']['ood_size_test']['auroc'] = None
    validate_result(classification)


@pytest.fixture
def review_one_thread():
    import torch
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_review_native_gin_checkpoint_rng_round_trip(records,task,config,tmp_path,review_one_thread):
    import numpy as np
    import torch
    from gnn_expressivity.models.gin import GIN
    from gnn_expressivity.training.controlled_task_runner import TorchGraphBackend
    config['model']['hidden_dim'] = 4
    config['model']['num_layers'] = 1
    config['model']['dropout'] = .1
    backend = TorchGraphBackend(records[:2],task,config,7)
    assert isinstance(backend.model,GIN)
    before = {k:v.clone() for k,v in backend.model.state_dict().items()}
    assert backend.model(backend._batch([r.graph_id for r in records[:2]])).shape == (2,2)
    loss = backend.epoch([r.graph_id for r in records[:2]])  # Exactly one tiny native Adam step.
    assert math.isfinite(loss) and loss >= 0
    assert any(not torch.equal(v,before[k]) for k,v in backend.model.state_dict().items())
    native = backend.state()
    path = tmp_path/'native.pt'
    torch.save(native,path)
    loaded = torch.load(path,map_location='cpu',weights_only=True)
    expected = ([random.random() for _ in range(3)],np.random.rand(3),torch.rand(3))
    # Fresh native model construction consumes RNG; restoration must undo it.
    restored = TorchGraphBackend(records[:2],task,config,99)
    restored.restore(loaded)
    actual = ([random.random() for _ in range(3)],np.random.rand(3),torch.rand(3))
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1],expected[1])
    torch.testing.assert_close(actual[2],expected[2],rtol=0,atol=0)
    for key,value in backend.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[key],value,rtol=0,atol=0)
    assert restored.optimizer.state_dict()['param_groups'] == backend.optimizer.state_dict()['param_groups']
    for key,state in backend.optimizer.state_dict()['state'].items():
        for field,value in state.items():
            torch.testing.assert_close(restored.optimizer.state_dict()['state'][key][field],value,rtol=0,atol=0)


def test_review_record_order_reproducibility(records,task,config,monkeypatch,review_one_thread):
    import torch
    from gnn_expressivity import encodings
    from gnn_expressivity.data.controlled_splits import dataset_hash
    from gnn_expressivity.training.controlled_task_runner import TorchGraphBackend
    class StochasticEncoding:
        def encode(self,data):
            result = data.clone()
            result.x = torch.rand(result.num_nodes,1)
            return result
        __call__ = encode
    monkeypatch.setattr(encodings,'build_encoding',lambda config: StochasticEncoding())
    graphs = records[:3]
    assert dataset_hash(graphs) == dataset_hash(list(reversed(graphs)))
    first = TorchGraphBackend(graphs,task,config,3)
    second = TorchGraphBackend(list(reversed(graphs)),task,config,3)
    assert list(first.graphs) == list(second.graphs) == sorted(r.graph_id for r in graphs)
    for graph_id in first.graphs:
        torch.testing.assert_close(first.graphs[graph_id].x,second.graphs[graph_id].x,rtol=0,atol=0)
    for key,value in first.model.state_dict().items():
        torch.testing.assert_close(value,second.model.state_dict()[key],rtol=0,atol=0)
    ids = sorted(first.graphs)
    assert first.predict(ids) == second.predict(ids)


@pytest.mark.parametrize('task_type', ['classification','regression'])
def test_review_native_gin_runner_smoke(records,config,tmp_path,review_one_thread,task_type):
    import torch
    from gnn_expressivity.training.controlled_task_runner import TorchGraphBackend
    tiny = records[:6]
    manifest = approve_manifest(create_manifest(tiny,task_id='fixture',dataset_version='native-smoke',
        split_type='iid',seed=2),tiny,approved_by='synthetic fixture',reference='test only')
    task = TaskSpec('fixture',task_type,'synthetic fixture only','test only',2 if task_type=='classification' else None)
    config['model']['hidden_dim'] = 4
    config['training'].update(epochs=1,patience=1)
    result = run_experiment(tiny,manifest,task,config,0,tmp_path)
    assert result['experiment_valid'] and len(result['history']) == result['selected_epoch'] == 1
    assert result['resources']['parameter_count'] > 0
    checkpoint = next(tmp_path.glob('*.partial.pt'))
    payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert payload['epoch'] == 1 and payload['best_epoch'] == 1
    assert payload['identity']['runtime']['backend'].endswith('TorchGraphBackend')
    validate_result(result)
    # Reuse the completed result without optimization, checkpoint rewriting or overwrite.
    before = checkpoint.read_bytes()
    assert run_experiment(tiny,manifest,task,config,0,tmp_path,resume=True) == result
    assert checkpoint.read_bytes() == before


def test_review_valid_encoding_bounds_do_not_consume_rng(config):
    import numpy as np
    import torch
    from gnn_expressivity.training.controlled_task_runner import validate_config
    python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    # Match constructor bounds: no arbitrary k/eigen_tol upper limit is imposed.
    for encoding in ({'name':'rwse','walk_length':1},
                     {'name':'lappe','k':100,'eigen_tol':100.0},
                     {'name':'lappe','k':1,'eigen_tol':1e-12}):
        config['encoding'] = encoding
        validate_config(config)
    assert random.getstate() == python_before
    after = np.random.get_state()
    assert after[0] == numpy_before[0] and after[2:] == numpy_before[2:]
    np.testing.assert_array_equal(after[1],numpy_before[1])
    torch.testing.assert_close(torch.get_rng_state(),torch_before,rtol=0,atol=0)
