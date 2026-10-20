import copy
import csv
import json

import pytest

from gnn_expressivity.analysis.wp3_6 import analyze, load_results, save_tables
from test_wp3_5 import values  # noqa: F401
from test_brec_analysis import runs  # noqa: F401


def test_seed_statistics_and_categories(values):
    tables, audit = analyze(values)
    row = tables['seed_variability'][0]
    assert row['n_seeds'] == 5 and json.loads(row['exact_seed_list']) == [0, 1, 2, 3, 4]
    assert row['mean_distinguished'] == 2
    assert row['std_distinguished'] == pytest.approx(2.5**.5)
    assert row['mean_fraction'] == 2/400
    assert row['std_fraction'] == pytest.approx(2.5**.5/400)
    assert row['min_distinguished'] == 0 and row['max_distinguished'] == 4
    basic = next(r for r in tables['category_variability'] if r['category'] == 'Basic')
    assert basic['mean_distinguished'] == 2
    assert basic['std_fraction'] == pytest.approx(2.5**.5/60)
    assert len(tables['category_variability']) == 6
    assert audit['hyperparameter_status'] == 'DATA-BLOCKED'


def test_pair_rates_and_agreement(values):
    tables, _ = analyze(values[1:3])
    pairs = {r['pair_id']: r for r in tables['pair_stability']}
    assert pairs[0]['classification'] == 'always_distinguished'
    assert pairs[1]['classification'] == 'seed_sensitive'
    assert pairs[1]['distinguished_count'] == 1 and pairs[1]['distinguished_rate'] == .5
    assert pairs[2]['classification'] == 'never_distinguished'
    agreement = tables['seed_agreement'][0]
    assert (agreement['seed_a'], agreement['seed_b']) == (1, 2)
    assert agreement['intersection_size'] == 1 and agreement['union_size'] == 2
    assert agreement['jaccard_similarity'] == .5


def test_single_seed(values):
    tables, _ = analyze(values[:1])
    row = tables['seed_variability'][0]
    assert row['std_distinguished'] is row['std_fraction'] is None
    assert row['n_seeds'] == 1 and 'not established' in row['interpretation_status']
    assert {r['classification'] for r in tables['pair_stability']} == {'insufficient_seeds'}
    assert not tables['seed_agreement']


def test_empty_union(values):
    other = copy.deepcopy(values[0])
    other['seed'] = 1
    tables, _ = analyze([values[0], other])
    assert tables['seed_agreement'][0]['union_size'] == 0
    assert tables['seed_agreement'][0]['jaccard_similarity'] == 1


@pytest.mark.parametrize('change', ['model', 'encoding', 'k', 'architecture', 'encoding_parameter'])
def test_condition_separation(values, change):
    a = copy.deepcopy(values[0])
    b = copy.deepcopy(a)
    if change == 'model':
        b['model'] = b['config']['model']['name'] = 'graphgps'
    elif change == 'encoding':
        b['encoding'] = b['config']['encoding']['name'] = 'rwse'
    elif change == 'k':
        for run, k in [(a, 4), (b, 8)]:
            run['encoding'] = run['config']['encoding']['name'] = 'lappe'
            run['config']['encoding']['k'] = k
    elif change == 'architecture':
        b['config']['model']['hidden_dim'] += 1
    else:
        b['config']['encoding']['walk_length'] = 8
    tables, _ = analyze([a, b])
    assert len(tables['seed_variability']) == 2
    assert len({r['condition_id'] for r in tables['seed_variability']}) == 2


@pytest.mark.parametrize('field', ['learning_rate', 'epochs', 'batch_size', 'optimizer'])
def test_hyperparameter_variants(values, field):
    alternative = copy.deepcopy(values[1])
    if field == 'batch_size':
        alternative['config']['task']['batch_size'] *= 2
    elif field == 'optimizer':
        alternative['config']['training']['optimizer'] = {'name': 'Adam', 'eps': 1e-7}
    else:
        alternative['config']['training'][field] *= 2
    tables, audit = analyze([*values[:2], alternative])
    rows = tables['hyperparameter_summary']
    assert len(rows) == 2 and len({r['condition_id'] for r in rows}) == 1
    assert sorted(r['n_seeds'] for r in rows) == [1, 2]
    assert audit['hyperparameter_status'] == 'RUNNABLE'
    assert audit['hyperparameter_comparisons'][0]['differing_training_fields']
    assert len(tables['hyperparameter_by_seed']) == 3
    assert len(tables['seed_agreement']) == 1


def test_top_level_optimizer_is_training_variant(values):
    original = copy.deepcopy(values[0])
    original['config']['optimizer'] = {'name': 'Adam', 'eps': 1e-8}
    alternative = copy.deepcopy(original)
    alternative['config']['optimizer']['eps'] = 1e-7

    tables, audit = analyze([original, alternative])
    rows = tables['hyperparameter_summary']
    assert len(rows) == 2
    assert len({r['condition_id'] for r in rows}) == 1
    variant_ids = {r['hyperparameter_variant'] for r in rows}
    assert len(variant_ids) == 2
    assert all(json.loads(r['exact_seed_list']) == [0] for r in rows)
    assert {json.dumps(audit['training_variants'][vid]['optimizer'], sort_keys=True)
            for vid in variant_ids} == {
        json.dumps(run['config']['optimizer'], sort_keys=True)
        for run in (original, alternative)}
    comparison = audit['hyperparameter_comparisons'][0]
    assert comparison['differing_training_fields'] == ['optimizer']
    assert set(comparison['variants']) == variant_ids
    assert comparison['pairwise_seed_overlap'][0]['shared_valid_seeds'] == [0]
    assert comparison['status'] == audit['hyperparameter_status'] == 'RUNNABLE'


def test_unmatched_variants_are_confounded(values):
    values[1]['config']['training']['learning_rate'] *= 2
    _, audit = analyze(values[:2])
    assert audit['hyperparameter_status'] == 'DATA-BLOCKED'
    assert 'confounded' in audit['hyperparameter_comparisons'][0]['reason']


def test_invalid_run_diagnostics(values):
    run = values[0]
    run['pairs'][0]['reliable'] = False
    run['overall']['reliability_failures'] = 1
    run['by_category']['Basic']['reliability_failures'] = 1
    run['benchmark_valid'] = run['reliability_valid'] = False
    run['primary_metric'] = None
    tables, _ = analyze(values[:2])
    row = tables['seed_variability'][0]
    assert row['n_seeds'] == 1 and row['mean_distinguished'] == 1
    assert row['all_benchmarks_valid'] is False and row['total_reliability_failures'] == 1
    assert json.loads(row['invalid_seeds']) == [0]
    assert len(tables['hyperparameter_by_seed']) == 2
    assert all(r['n_seeds'] == 1 for r in tables['pair_stability'])
    assert not tables['seed_agreement']
    invalid_only, _ = analyze(values[:1])
    assert invalid_only['seed_variability'][0]['mean_distinguished'] is None
    assert invalid_only['seed_variability'][0]['n_seeds'] == 0
    assert not invalid_only['pair_stability']


@pytest.mark.parametrize('field', ['source_sha256', 'protocol', 'task'])
def test_incompatible_runs(values, field):
    if field == 'source_sha256':
        values[1][field] = 'b'*64
    elif field == 'protocol':
        values[1][field]['threshold'] += 1
    else:
        values[1]['config']['task']['num_workers'] += 1
    with pytest.raises(ValueError, match='Incompatible'):
        analyze(values[:2])


def test_duplicates(values):
    with pytest.raises(ValueError, match='Duplicate'):
        analyze([values[0], values[0]])


def test_global_membership(values):
    other = copy.deepcopy(values[0])
    other['config']['training']['epochs'] *= 2
    other['pairs'][0]['category'], other['pairs'][60]['category'] = 'Regular', 'Basic'
    with pytest.raises(ValueError, match='membership'):
        analyze([values[0], other])


def test_environment_is_not_variant(values):
    values[1]['device'] = 'cuda'
    with pytest.warns(UserWarning, match='environments differ'):
        tables, _ = analyze(values[:2])
    assert len(tables['seed_variability']) == 1
    assert tables['seed_variability'][0]['environment_count'] == 2
    assert tables['seed_variability'][0]['environment_warning']


def test_theory_not_inferred(values):
    tables, _ = analyze(values)
    assert {r['theoretical_status_note'] for r in tables['theory_vs_realized']} == {
        'not established in this project'}


def test_environment_warning_across_variants(values):
    other = copy.deepcopy(values[0])
    other['config']['training']['epochs'] *= 2
    other['device'] = 'cuda'
    with pytest.warns(UserWarning, match='across training variants'):
        tables, audit = analyze([values[0], other])
    assert len(tables['seed_variability']) == 2
    assert audit['hyperparameter_comparisons'][0]['environment_count'] == 2


def test_missing_seeds_within_variant(values, tmp_path):
    other = copy.deepcopy(values[0])
    other['config']['training']['epochs'] *= 2
    for name, run in [('a', values[0]), ('b', values[1]), ('c', other)]:
        (tmp_path/f'brec_{name}.json').write_text(json.dumps(run))
    with pytest.raises(ValueError, match='training-variant seeds'):
        load_results(tmp_path, seeds=[0, 1], conditions=[('gin', 'none', None)])


def test_missing_discovery_and_output(values, tmp_path):
    (tmp_path/'brec_gin_none_seed0.json').write_text(json.dumps(values[0]))
    (tmp_path/'brec_graphgps_lappe_seed0_k4_range320_330.json').write_text('diagnostic')
    (tmp_path/'brec_gin_none_seed1_checkpoint.json').write_text('checkpoint')
    with pytest.raises(ValueError, match='Missing'):
        load_results(tmp_path)
    tables, audit = load_results(tmp_path, allow_missing=True)
    assert audit['discovery']['missing']
    assert len(audit['discovery']['skipped']) == 2
    assert tables['seed_variability'][0]['missing_seeds'] == '[1,2,3,4]'
    paths = save_tables(tables, audit, tmp_path/'analysis')
    assert len(paths) == 8 and all(p.exists() for p in paths)
    with paths[2].open(newline='') as file:
        assert list(csv.DictReader(file)) == []  # single seed: agreement header only
    with pytest.raises(FileExistsError):
        save_tables(tables, audit, tmp_path/'analysis')


def test_reject_range_and_partial(values):
    values[0]['pair_range'] = [0, 20]
    with pytest.raises(ValueError, match='diagnostic'):
        analyze(values[:1])
    values[0].pop('pair_range')
    values[0]['full_benchmark_completed'] = False
    with pytest.raises(ValueError, match='Incomplete'):
        analyze(values[:1])
