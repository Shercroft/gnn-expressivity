"""WP4 supervised protocol, separate from BREC; real runs require approved inputs."""
import copy
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from numbers import Real

from gnn_expressivity.data.controlled_splits import canonical, digest, validate_manifest


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_type: str
    target_description: str
    approval_reference: str
    num_classes: int | None = None

    @property
    def loss_name(self):
        return 'cross_entropy' if self.task_type == 'classification' else 'mean_squared_error'

    def validate(self, records):
        if not self.task_id or not self.target_description or not self.approval_reference:
            raise ValueError("Task definition and approval reference required")
        if self.task_type not in ('classification', 'regression'):
            raise ValueError("Unsupported task type")
        if self.task_type == 'classification' and (type(self.num_classes) is not int or self.num_classes < 2):
            raise ValueError("Classification requires at least two configured classes")
        if self.task_type == 'regression' and self.num_classes is not None:
            raise ValueError("Regression must not declare classes")
        for record in records:
            y = record.target
            if self.task_type == 'classification':
                if type(y) is not int or not 0 <= y < self.num_classes:
                    raise ValueError("Classification targets must be approved integer class IDs")
            elif isinstance(y, bool) or not isinstance(y, (int, float)) or not math.isfinite(y):
                raise ValueError("Finite scalar regression target required")


def metrics(task, targets, predictions):
    if task.task_type not in ('classification', 'regression'):
        raise ValueError('Unsupported task type')
    if any(isinstance(y, bool) or not isinstance(y, (int, float)) or not math.isfinite(y) for y in targets):
        raise ValueError('Finite numeric targets required')
    if task.task_type == 'classification' and (type(task.num_classes) is not int or task.num_classes < 2 or any(type(y) is not int or not 0 <= y < task.num_classes for y in targets)):
        raise ValueError('Invalid classification target schema')
    if not targets or len(targets) != len(predictions):
        raise ValueError("Nonempty aligned targets/predictions required")
    if task.task_type == 'regression':
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) for p in predictions):
            raise ValueError("Finite scalar predictions required")
        errors = [p - y for y, p in zip(targets, predictions)]
        mse = sum(e * e for e in errors) / len(errors)
        return {'loss': mse, 'mae': sum(abs(e) for e in errors)/len(errors), 'rmse': math.sqrt(mse)}
    probabilities, predicted, losses = [], [], []
    for y, scores in zip(targets, predictions):
        if len(scores) != task.num_classes or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in scores):
            raise ValueError("Finite class-logit vectors required")
        maximum = max(scores)
        scaled = [math.exp(x - maximum) for x in scores]
        total = sum(scaled)
        probabilities.append([x/total for x in scaled])
        predicted.append(max(range(len(scores)), key=lambda i: scores[i]))
        losses.append(math.log(total) + maximum - scores[y])
    classes = sorted(set(targets))
    accuracy = sum(a == b for a, b in zip(targets, predicted))/len(targets)
    balanced = sum(sum(y == c and p == c for y, p in zip(targets, predicted))/targets.count(c) for c in classes)/len(classes)
    auroc, reason = None, 'all configured classes must occur in the evaluation pool'
    if set(classes) == set(range(task.num_classes)):
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(targets, [p[1] for p in probabilities])) if task.num_classes == 2 else float(roc_auc_score(targets, probabilities, multi_class='ovr', labels=classes))
        reason = None
    return dict(loss=sum(losses)/len(losses), accuracy=accuracy, balanced_accuracy=balanced,
                auroc=auroc, auroc_unavailable_reason=reason)


def validate_config(config):
    if config['model']['name'] not in ('gin', 'graphgps') or config['encoding']['name'] not in ('none','degree','rwse','lappe'):
        raise ValueError("Use existing approved model/encoding adapters")
    model = config['model']
    model_fields = {'name','hidden_dim','num_layers','dropout','activation','pooling','train_eps'}
    if model['name'] == 'graphgps':
        model_fields.add('num_heads')
    if not set(model) <= model_fields:
        raise ValueError('Unsupported model configuration fields')
    for key, default in (('hidden_dim',128),('num_layers',4)):
        value = model.get(key, default)
        if isinstance(value,bool) or not isinstance(value,int) or value <= 0:
            raise ValueError(f'Positive integer model.{key} required')
    dropout = model.get('dropout',0.1)
    if isinstance(dropout,bool) or not isinstance(dropout,Real) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError('model.dropout must be finite and in [0,1)')
    if model.get('activation','relu') != 'relu' or model.get('pooling','sum') not in ('sum','mean','max'):
        raise ValueError('Unsupported model activation/pooling')
    if type(model.get('train_eps',True)) is not bool:
        raise ValueError('model.train_eps must be boolean')
    if model['name'] == 'graphgps':
        heads = model.get('num_heads',4)
        if isinstance(heads,bool) or not isinstance(heads,int) or heads <= 0 or model.get('hidden_dim',128) % heads:
            raise ValueError('Positive model.num_heads must divide hidden_dim')
    training = config['training']
    if set(training) != {'epochs','batch_size','patience','min_delta','learning_rate','weight_decay','optimizer'}:
        raise ValueError('Unsupported or missing training options; no silently ignored knobs')
    allowed = {'none':{'name'}, 'degree':{'name'}, 'rwse':{'name','walk_length'}, 'lappe':{'name','k','eigen_tol'}}
    if set(config['encoding']) != allowed[config['encoding']['name']]:
        raise ValueError('Explicit supported encoding options required')
    for key in ('epochs', 'batch_size', 'patience'):
        if type(training[key]) is not int or training[key] <= 0:
            raise ValueError(f"Positive integer training.{key} required")
    for key in ('learning_rate', 'weight_decay', 'min_delta'):
        if isinstance(training[key], bool) or not isinstance(training[key], (int, float)) or not math.isfinite(training[key]) or training[key] < 0:
            raise ValueError(f"Invalid training.{key}")
    if training['learning_rate'] <= 0 or training['optimizer'] != 'Adam':
        raise ValueError("This adapter supports explicit Adam with positive learning rate")
    if config['node_features'] not in ('constant', 'record') or config.get('validation_domain', 'id_only') != 'id_only':
        raise ValueError("Explicit node-feature policy and ID-only validation required")
    # Constructor validation is authoritative; this does not compute an encoding or consume RNG.
    from gnn_expressivity.encodings import build_encoding
    build_encoding(config)
    if config.get('device', 'cpu') != 'cpu':
        raise ValueError('Initial WP4 adapter supports CPU only; accelerator integration is deferred')
    canonical(config)


class TorchGraphBackend:
    """CPU adapter using existing factories; no changes to WP3 model/encoding code."""
    def __init__(self, records, task, config, seed):
        import numpy as np
        import torch
        from torch_geometric.data import Data
        from gnn_expressivity.encodings import build_encoding
        from gnn_expressivity.models import build_model
        from gnn_expressivity.training.profiling import count_parameters
        records = sorted(records, key=lambda record: record.graph_id)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.torch, self.task, self.config = torch, task, config
        self.graphs, self.lookup = {}, {r.graph_id: r for r in records}
        encoder = build_encoding(config)
        for record in records:
            edges = list(record.edges)
            if not record.directed:
                edges += [(v, u) for u, v in record.edges]
            edge_index = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous()
            x = None
            if config['node_features'] == 'record':
                if record.node_features is None:
                    raise ValueError("Record node features requested but missing")
                x = torch.tensor(record.node_features, dtype=torch.float32)
            self.graphs[record.graph_id] = encoder(Data(edge_index=edge_index, num_nodes=record.num_nodes, x=x))
        widths = {int(g.x.shape[1]) for g in self.graphs.values()}
        if len(widths) != 1:
            raise ValueError("Encoded feature width must be consistent")
        self.model = build_model(config, in_dim=widths.pop(), out_dim=task.num_classes if task.task_type == 'classification' else 1)
        self.parameter_count = count_parameters(self.model)
        training = config['training']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=training['learning_rate'], weight_decay=training['weight_decay'])

    def _batch(self, ids):
        from torch_geometric.data import Batch
        return Batch.from_data_list([self.graphs[i] for i in ids])

    def epoch(self, ids):
        torch = self.torch
        self.model.train()
        order = list(ids)
        random.shuffle(order)
        total = 0.0
        size = self.config['training']['batch_size']
        for start in range(0, len(order), size):
            batch_ids = order[start:start+size]
            output = self.model(self._batch(batch_ids))
            if self.task.task_type == 'classification':
                targets = torch.tensor([self.lookup[i].target for i in batch_ids], dtype=torch.long)
                loss = torch.nn.functional.cross_entropy(output, targets)
            else:
                targets = torch.tensor([self.lookup[i].target for i in batch_ids], dtype=torch.float32)
                loss = torch.nn.functional.mse_loss(output.flatten(), targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += float(loss.detach()) * len(batch_ids)
        return total/len(order)

    def predict(self, ids):
        self.model.eval()
        result = []
        with self.torch.no_grad():
            size = self.config['training']['batch_size']
            for start in range(0, len(ids), size):
                output = self.model(self._batch(ids[start:start+size]))
                result.extend(output.tolist() if self.task.task_type == 'classification' else output.flatten().tolist())
        return result

    def state(self):
        from gnn_expressivity.training.pair_checkpoint import rng_state
        return copy.deepcopy({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'rng': rng_state()})

    def restore(self, state):
        from gnn_expressivity.training.pair_checkpoint import restore_rng
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'])


def _checkpoint(path, payload):
    import torch
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_result(result):
    identity = result['identity']
    spec = TaskSpec(**identity['task'])
    spec.validate([])
    if identity['loss_function'] != spec.loss_name:
        raise ValueError('Task/loss identity mismatch')
    if type(identity['training_seed']) is not int or not 0 <= identity['training_seed'] < 2**32:
        raise ValueError('Invalid result training seed')
    if result['run_id'] != 'wp4_' + digest(identity) or result.get('experiment_valid') is not True:
        raise ValueError("Invalid result identity/status")
    if identity['task']['task_id'] != identity['task_id'] or not identity['train_ids']:
        raise ValueError("Invalid task/train identity")
    parent = identity['parent_train_ids']
    if len(parent) != len(set(parent)) or len(identity['train_ids']) != len(set(identity['train_ids'])) or not set(identity['train_ids']) <= set(parent):
        raise ValueError('Invalid result training membership')
    seen = set(parent)
    if not {'validation', 'iid_test'} <= set(identity['evaluation_ids']):
        raise ValueError('Required evaluation IDs absent')
    for name, ids in identity['evaluation_ids'].items():
        if not ids or len(ids) != len(set(ids)) or set(ids) & seen:
            raise ValueError('Result train/evaluation leakage')
        seen.update(ids)
    if identity['evaluation_hash'] != digest(identity['evaluation_ids']):
        raise ValueError('Evaluation membership hash mismatch')
    expected = set(identity['evaluation_ids']) - {'validation'}
    if set(result['performance']) != expected:
        raise ValueError("Missing test observations")
    if result['sample_count'] != len(identity['train_ids']) or not result['history']:
        raise ValueError("Invalid sample count or training history")
    validate_config(identity['config'])
    for pool, values in result['performance'].items():
        required = ('loss', 'accuracy', 'balanced_accuracy', 'auroc') if identity['task']['task_type'] == 'classification' else ('loss','mae','rmse')
        if any(metric not in values for metric in required):
            raise ValueError("Incomplete performance metrics")
        for key in required:
            value = values[key]
            if value is None and key == 'auroc':
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Invalid metric value")
            if value < 0 or (key in ('accuracy','balanced_accuracy','auroc') and value > 1):
                raise ValueError(f"Invalid metric range: {pool}/{key}")
    for key in ('training_sec','preprocessing_sec','validation_sec','inference_sec'):
        if not isinstance(result['resources'][key], (int,float)) or not math.isfinite(result['resources'][key]) or result['resources'][key] < 0:
            raise ValueError('Invalid resource measurement')
    if type(result['resources']['parameter_count']) is not int or result['resources']['parameter_count'] < 1:
        raise ValueError('Invalid parameter count')
    if [r['epoch'] for r in result['history']] != list(range(1,len(result['history'])+1)) or result['selected_epoch'] not in range(1,len(result['history'])+1):
        raise ValueError('Invalid selected epoch/history')
    expected_deltas = {pool: {key: value - result['performance']['iid_test'][key] if isinstance(value,(float,int)) and not isinstance(value,bool) and result['performance']['iid_test'].get(key) is not None else None for key,value in measured.items() if key != 'auroc_unavailable_reason'} for pool,measured in result['performance'].items() if pool != 'iid_test'}
    if result['ood_minus_iid'] != expected_deltas:
        raise ValueError('OOD differences disagree with absolute metrics')
    canonical(result)
    return result


def run_experiment(records, manifest, task, config, seed, output_dir, *, train_ids=None,
                   resume=False, backend_factory=TorchGraphBackend, subset_design=None):
    """Only validation selects epochs; IID and OOD test predictions occur at the end."""
    records = sorted(records, key=lambda record: record.graph_id)
    manifest, config = copy.deepcopy(manifest), copy.deepcopy(config)
    validate_manifest(manifest, records, require_approved=True)
    task.validate(records)
    validate_config(config)
    if manifest['task_id'] != task.task_id or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Task identity/seed mismatch")
    selected = sorted(manifest['splits']['train'] if train_ids is None else train_ids)
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(manifest['splits']['train']):
        raise ValueError("Training subset must contain unique approved training IDs only")
    if subset_design is not None:
        from gnn_expressivity.data.controlled_splits import nested_training_subsets
        if subset_design.get('policy') != 'nested' or type(subset_design.get('subset_seed')) is not int or type(subset_design.get('balanced')) is not bool:
            raise ValueError('Invalid nested-subset design')
        expected = nested_training_subsets(manifest, records, [len(selected)], seed=subset_design['subset_seed'], balanced=subset_design['balanced'])[0]
        if set(expected['train_ids']) != set(selected):
            raise ValueError('Training IDs disagree with deterministic nested-subset design')
    evaluation_ids = {k: v for k, v in manifest['splits'].items() if k != 'train'}
    import platform
    import torch
    import numpy as np
    import torch_geometric
    from gnn_expressivity.training.pair_checkpoint import source_digest
    wp4_sources = [Path(__file__), Path(__file__).parents[1] / 'data/controlled_splits.py']
    implementation = digest({'base_sources': source_digest(), 'wp4_sources': [p.read_text() for p in wp4_sources]})
    runtime = dict(python=platform.python_version(), os=platform.platform(), torch=str(torch.__version__), torch_num_threads=torch.get_num_threads(),
        numpy=np.__version__, pyg=str(torch_geometric.__version__), backend=backend_factory.__module__ + '.' + backend_factory.__qualname__, implementation=implementation)
    identity = dict(task_id=task.task_id, task=asdict(task), loss_function=task.loss_name, dataset_version=manifest['dataset_version'],
        dataset_hash=manifest['dataset_hash'], manifest_hash=manifest['manifest_hash'],
        evaluation_hash=digest(evaluation_ids), evaluation_ids=evaluation_ids,
        parent_train_ids=manifest['splits']['train'], train_ids=selected, training_seed=seed, config=config,
        subset_design=subset_design or {'policy': 'explicit_ids' if train_ids is not None else 'full_training_split'}, runtime=runtime)
    run_id = 'wp4_' + digest(identity)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path, checkpoint = directory/(run_id+'.json'), directory/(run_id+'.partial.pt')
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        result = validate_result(json.loads(path.read_text()))
        if result['identity'] != identity:
            raise ValueError("Completed result identity mismatch")
        return result
    if checkpoint.exists() and not resume:
        raise FileExistsError(checkpoint)
    lock = directory/(run_id+'.lock')
    with lock.open('x') as file:
        file.write(str(os.getpid()))
    try:
        preprocessing_start = time.perf_counter()
        backend = backend_factory(records, task, config, seed)
        preprocessing_sec = time.perf_counter() - preprocessing_start
        state = dict(identity=identity, backend=backend.state(), best=None, best_loss=None,
                     best_epoch=None, epoch=0, stale=0, history=[], training_sec=0.0,
                     validation_sec=0.0, preprocessing_sec=preprocessing_sec)
        if checkpoint.exists():
            import torch
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if state['identity'] != identity:
                raise ValueError("Checkpoint identity mismatch")
            backend.restore(state['backend'])
        else:
            _checkpoint(checkpoint, state)
        lookup = {r.graph_id: r for r in records}
        settings = config['training']
        while state['epoch'] < settings['epochs'] and state['stale'] < settings['patience']:
            start = time.perf_counter()
            train_loss = backend.epoch(selected)
            if not math.isfinite(train_loss):
                raise ValueError("Nonfinite training loss")
            state['training_sec'] += time.perf_counter() - start
            start = time.perf_counter()
            ids = evaluation_ids['validation']
            validation = metrics(task, [lookup[i].target for i in ids], backend.predict(ids))
            state['validation_sec'] += time.perf_counter() - start
            state['epoch'] += 1
            improved = state['best_loss'] is None or validation['loss'] < state['best_loss'] - settings['min_delta']
            if improved:
                state.update(best=copy.deepcopy(backend.state()), best_loss=validation['loss'], best_epoch=state['epoch'], stale=0)
            else:
                state['stale'] += 1
            state['history'].append(dict(epoch=state['epoch'], train_loss=train_loss, validation_loss=validation['loss']))
            state['backend'] = backend.state()
            _checkpoint(checkpoint, state)
        backend.restore(state['best'])
        start = time.perf_counter()
        performance = {name: metrics(task, [lookup[i].target for i in ids], backend.predict(ids))
                       for name, ids in evaluation_ids.items() if name != 'validation'}
        inference_sec = time.perf_counter() - start
        iid = performance['iid_test']
        deltas = {name: {key: value - iid[key] if isinstance(value, (float, int)) and not isinstance(value, bool) and iid.get(key) is not None else None
                        for key, value in measured.items() if key != 'auroc_unavailable_reason'}
                  for name, measured in performance.items() if name != 'iid_test'}
        result = dict(run_id=run_id, identity=identity, sample_count=len(selected), performance=performance,
            ood_minus_iid=deltas, selected_epoch=state['best_epoch'], selection='ID validation loss only',
            history=state['history'], experiment_valid=True,
            resources=dict(parameter_count=backend.parameter_count, preprocessing_sec=state['preprocessing_sec'],
                training_sec=state['training_sec'], validation_sec=state['validation_sec'], inference_sec=inference_sec),
            limitations=['Fixed approved split; OOD test never selects checkpoints',
                         'Epoch-boundary resume; single writer per run required; local trusted checkpoints only'])
        validate_result(result)
        with path.open('x', encoding='utf-8') as file:
            file.write(json.dumps(result, indent=2, allow_nan=False))
        return result
    finally:
        lock.unlink(missing_ok=True)
