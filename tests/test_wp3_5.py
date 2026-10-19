import copy
import json

import pytest

from gnn_expressivity.analysis.brec_analysis import CATEGORIES
from gnn_expressivity.analysis.wp3_5 import aggregate, category_rows, discover, save_tables
from test_brec_analysis import runs  # noqa: F401


@pytest.fixture
def values(runs):
    for run in runs:
        run['reliability_valid'] = True
        run['pairs'] = []
        for category, n in CATEGORIES.items():
            for i in range(n):
                distinguished = category == 'Basic' and i < run['seed']
                run['pairs'].append(dict(pair_id=len(run['pairs']), category=category,
                    distinguished=distinguished, candidate_distinguished=distinguished, reliable=True))
    return runs


def test_counts_statistics_and_outputs(values, tmp_path):
    by_seed, summary, wide, curves = aggregate(values)
    assert len(by_seed) == 30
    basic = summary[0]
    assert basic['category'] == 'Basic'
    assert basic['n_seeds'] == 5 and json.loads(basic['seeds']) == list(range(5))
    assert basic['mean_distinguished'] == 2
    assert basic['std_distinguished'] == pytest.approx(2.5**.5)
    assert basic['mean_fraction'] == pytest.approx(2/60)
    assert basic['std_fraction'] == pytest.approx(2.5**.5/60)
    assert wide[0]['Overall'] == 2 and not curves
    assert category_rows(values[1])[0]['distinguished_fraction'] == 1/60
    assert aggregate(values[:1])[1][0]['std_distinguished'] is None
    assert aggregate(values[:1])[1][0]['std_fraction'] is None
    paths = save_tables(values, tmp_path, {})
    assert all(p.exists() for p in paths)
    with pytest.raises(FileExistsError):
        save_tables(values, tmp_path, {})


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(full_benchmark_completed=False),
    lambda r: r.update(pair_range=[320,330]),
    lambda r: r.update(run_id='brec_gin_none_seed0_range0_400'),
    lambda r: r['pairs'][0].update(category='unknown'),
    lambda r: r['pairs'][0].update(pair_id=1),
    lambda r: r['overall'].update(pairs=399),
    lambda r: r['overall'].update(distinguished=100),
    lambda r: r['by_category']['Basic'].update(distinguished=100),
    lambda r: r['config']['model'].update(name='graphgps'),
    lambda r: r.update(benchmark_valid=False),
])
def test_bad_runs(values, mutation):
    mutation(values[0])
    with pytest.raises(ValueError):
        aggregate(values)


def test_incompatible_and_duplicate(values):
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate([values[0], values[0]])
    values[1]['config']['training']['epochs'] += 1
    with pytest.raises(ValueError, match='Incompatible'):
        aggregate(values)


def test_global_category_membership_across_conditions(values):
    original = values[0]
    changed = copy.deepcopy(original)
    changed['model'] = changed['config']['model']['name'] = 'graphgps'
    changed['run_id'] = 'brec_graphgps_none_seed0'
    # Swap saved labels to preserve category sizes and all summary counts.
    basic = next(p for p in changed['pairs'] if p['category'] == 'Basic')
    regular = next(p for p in changed['pairs'] if p['category'] == 'Regular')
    basic['category'], regular['category'] = regular['category'], basic['category']
    category_rows(original)
    category_rows(changed)
    with pytest.raises(ValueError, match='Category membership differs across runs'):
        aggregate([original, changed])


def test_model_encoding_k_and_curves(values):
    variants = []
    for model, encoding, k in [('gin','none',None), ('graphgps','none',None),
                               ('gin','rwse',None), ('gin','lappe',4),
                               ('gin','lappe',8), ('graphgps','lappe',4)]:
        run = copy.deepcopy(values[0])
        run['model'] = run['config']['model']['name'] = model
        run['encoding'] = run['config']['encoding']['name'] = encoding
        if k:
            run['config']['encoding']['k'] = k
        variants.append(run)
    _, summary, wide, curves = aggregate(variants)
    assert len(summary) == 36 and len(wide) == 6 and len(curves) == 18
    assert {r['k'] for r in curves} == {4,8}
    assert {r['model'] for r in curves} == {'gin','graphgps'}


def test_invalid_reliability_preserved(values):
    run = values[0]
    run['pairs'][0]['reliable'] = False
    run['overall']['reliability_failures'] = 1
    run['by_category']['Basic']['reliability_failures'] = 1
    run['benchmark_valid'] = run['reliability_valid'] = False
    run['primary_metric'] = None
    rows, summary, wide, _ = aggregate([run])
    assert not any(r['benchmark_valid'] for r in rows)
    assert summary[0]['total_reliability_failures'] == 1
    assert wide[0]['all_benchmarks_valid'] is False


def test_discovery_missing_and_diagnostics(values, tmp_path):
    (tmp_path/'brec_gin_none_seed0.json').write_text(json.dumps(values[0]))
    (tmp_path/'brec_gin_none_seed0_range0_20.json').write_text('invalid diagnostic')
    partial = values[1] | {'full_benchmark_completed': False}
    (tmp_path/'brec_gin_none_seed1.json').write_text(json.dumps(partial))
    with pytest.raises(ValueError, match='Missing'):
        discover(tmp_path)
    found, audit = discover(tmp_path, allow_missing=True,
                            conditions=[('gin','none',None), ('gin','lappe',4)])
    assert len(found) == 1 and len(audit['missing']) == 9 and len(audit['skipped']) == 2
    assert audit['available'] == [('gin','none',None,0)]
