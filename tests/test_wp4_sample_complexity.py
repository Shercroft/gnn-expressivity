import copy
import json

import pytest

from gnn_expressivity.analysis.wp4_sample_complexity import analyze, save_analysis
from gnn_expressivity.data.controlled_splits import nested_training_subsets, digest
from gnn_expressivity.training.controlled_task_runner import run_experiment
from test_wp4_splits import records, manifest  # noqa: F401
from test_wp4_generalization import config, task, FakeBackend  # noqa: F401


def test_curve_schema_nested_ids_and_fixed_evaluation(records,manifest,config,task,tmp_path):
    subsets = nested_training_subsets(manifest,records,[2,6],seed=11,balanced=True)
    results = [run_experiment(records,manifest,task,config,s,tmp_path,
        train_ids=subset['train_ids'],subset_design={'policy':'nested','subset_seed':11,'balanced':True},
        backend_factory=FakeBackend) for subset in subsets for s in (0,1)]
    rows, summary = analyze(results,expected_seeds=[0,1,2])
    assert {r['sample_count'] for r in rows} == {2,6}
    assert len({r['curve_id'] for r in rows}) == 1
    assert all(r['n_seeds']==2 and json.loads(r['missing_seeds'])==[2] for r in summary)
    assert len({r['identity']['evaluation_hash'] for r in results}) == 1
    assert all(r['training_sec'] >= 0 for r in rows)
    changed = copy.deepcopy(config)
    changed['training']['epochs'] += 1
    results.append(run_experiment(records,manifest,task,changed,0,tmp_path,
        train_ids=subsets[0]['train_ids'],subset_design={'policy':'nested','subset_seed':11,'balanced':True},backend_factory=FakeBackend))
    assert len({r['curve_id'] for r in analyze(results)[0]}) == 2
    paths = save_analysis(results,tmp_path/'analysis')
    assert all(p.exists() for p in paths)


def test_non_nested_or_mismatched_subsets_rejected(records,manifest,config,task,tmp_path):
    with pytest.raises(ValueError,match='deterministic'):
        run_experiment(records,manifest,task,config,0,tmp_path,train_ids=manifest['splits']['train'][:2],
            subset_design={'policy':'nested','subset_seed':11,'balanced':True},backend_factory=FakeBackend)
    subsets = nested_training_subsets(manifest,records,[2,6],seed=11)
    results = [run_experiment(records,manifest,task,config,0,tmp_path,train_ids=s['train_ids'],
        subset_design={'policy':'nested','subset_seed':11,'balanced':False},backend_factory=FakeBackend) for s in subsets]
    changed = copy.deepcopy(results[1])
    pool = set(manifest['splits']['train']) - set(changed['identity']['train_ids'])
    remove = results[0]['identity']['train_ids'][0]
    changed['identity']['train_ids'][changed['identity']['train_ids'].index(remove)] = sorted(pool)[0]
    changed['run_id'] = 'wp4_' + digest(changed['identity'])
    with pytest.raises(ValueError,match='nested'):
        analyze([results[0],changed])


def test_no_fabricated_observations(tmp_path):
    paths = save_analysis([],tmp_path/'empty',expected_seeds=[0,1])
    assert 'DATA-BLOCKED' in paths[-1].read_text()
    assert len(paths[0].read_text().splitlines()) == 1
