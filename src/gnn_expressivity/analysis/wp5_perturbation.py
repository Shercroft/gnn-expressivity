"""WP5.4: small-graph edge-edit diagnostics, not a robustness/generalization score."""
from __future__ import annotations

from collections import Counter
import itertools
import json
from pathlib import Path
import random

import torch
from torch_geometric.data import Batch, Data

from gnn_expressivity.analysis.wp5_permutation import (
    error_record, laplacian_spectrum, native_outputs, permutation, relabel, small_graphs,
)
from gnn_expressivity.encodings import build_encoding
from gnn_expressivity.models import build_model


def simple_edges(data):
    """Validate a single CPU simple undirected graph with explicitly paired arcs."""
    if not isinstance(data, Data) or isinstance(data, Batch) or data.num_nodes is None or data.num_nodes < 1:
        raise ValueError('one nonempty homogeneous graph required; Batch unsupported')
    unsupported = [key for key in data.keys() if key in ('batch', 'ptr', 'adj', 'adj_t', 'adjacency', 'adjacency_matrix')
                   or key.endswith(('_batch', '_ptr')) or key.startswith(('adj_', 'adjacency_'))
                   or ('edge_index' in key and key != 'edge_index')]
    if unsupported:
        raise ValueError(f'Batch metadata or alternate adjacency unsupported: {sorted(unsupported)}')
    edges = data.edge_index
    if not isinstance(edges, torch.Tensor) or edges.device.type != 'cpu' or edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('CPU long edge_index [2,E] required')
    arcs = [tuple(edge) for edge in edges.T.tolist()]
    if len(arcs) != len(set(arcs)):
        raise ValueError('duplicate arcs are unsupported')
    if any(u == v or min(u, v) < 0 or max(u, v) >= data.num_nodes for u, v in arcs):
        raise ValueError('invalid endpoint or self-loop')
    arc_set = set(arcs)
    if any((v, u) not in arc_set for u, v in arcs):
        raise ValueError('symmetric undirected arcs required')
    return {(min(u, v), max(u, v)) for u, v in arcs}


def graph_stats(data):
    import networkx as nx
    edges = simple_edges(data)
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_edges_from(edges)
    n = data.num_nodes
    return dict(num_nodes=n, num_edges=len(edges), density=2 * len(edges) / (n * (n - 1)) if n > 1 else 0.,
                degrees=[graph.degree(i) for i in range(n)], components=nx.number_connected_components(graph))


def perturb(data, kind='add', count=1, seed=0):
    """Choose distinct unordered pairs, toggled once; never silently partially edit.

    add/remove sample absent/present edges. edits toggles any unordered pairs.
    Exact requested count is achieved or the result is infeasible and unchanged.
    Input features/labels remain node-aligned. Edge-associated attributes/index
    schemas are rejected: inventing attributes for new edges would change meaning.
    """
    old = simple_edges(data)
    if kind not in ('add', 'remove', 'edits'):
        raise ValueError('unknown perturbation kind')
    if type(count) is not int or count < 0 or type(seed) is not int or seed < 0:
        raise ValueError('nonnegative integer count and seed required')
    for key, value in data:
        if key != 'edge_index' and (key in ('edge_attr','edge_weight','face') or 'index' in key or data.is_edge_attr(key)):
            raise ValueError(f'unsupported edge/index attribute: {key}')
    pairs = set(itertools.combinations(range(data.num_nodes), 2))
    candidates = sorted(pairs - old if kind == 'add' else old if kind == 'remove' else pairs)
    status = 'noop' if count == 0 else 'infeasible' if count > len(candidates) else 'applied'
    selected = random.Random(seed).sample(candidates, count) if status == 'applied' else []
    added, removed = sorted(set(selected) - old), sorted(set(selected) & old)
    new = (old - set(removed)) | set(added)
    result = data.clone()
    if status == 'applied':
        arcs = sorted([(u, v) for u, v in new] + [(v, u) for u, v in new])
        result.edge_index = torch.tensor(arcs, dtype=torch.long).reshape(-1, 2).T.contiguous()
    before, after = graph_stats(data), graph_stats(result)
    metadata = dict(kind=kind, seed=seed, requested_edits=count, edit_count=len(selected), status=status,
                    added_edges=[list(e) for e in added], removed_edges=[list(e) for e in removed],
                    before=before, after=after, degree_changes=[b-a for a,b in zip(before['degrees'], after['degrees'])],
                    density_change=after['density']-before['density'],
                    warning='Connectivity is not constrained; components and isolates are recorded.' if status == 'applied' else 'No edit requested.' if status == 'noop' else 'Insufficient eligible pairs; no partial edit performed.')
    return result, metadata


def difference(before, after, *, control=False):
    """Absolute max, Frobenius norm and reference-normalized change, no score."""
    record = error_record(before, after, diagnostic=not control)
    valid = before.shape == after.shape and bool(torch.isfinite(before).all() and torch.isfinite(after).all())
    if valid:
        delta = torch.linalg.vector_norm((after-before).double()).item()
        scale = torch.linalg.vector_norm(before.double()).item()
    else:
        delta, scale = None, 0.
    maximum = float(before.abs().max()) if valid and before.numel() else 0.
    # Avoid reference-norm underflow without changing the absolute delta above.
    if valid and scale == 0 and maximum > 0:
        scale = torch.linalg.vector_norm(before.double() / maximum).item() * maximum
    threshold = record['tolerance']['atol']
    warnings = []
    if valid:
        for name, value in (('reference_max_abs', maximum), ('reference_l2', scale)):
            if value == 0:
                warnings.append(f'{name} is zero; its normalized difference is undefined.')
            elif value <= threshold:
                warnings.append(f'{name} is near zero; its ratio may amplify tiny absolute changes.')
    return record | dict(l2_change=delta, normalized_l2_change=delta / scale if delta is not None and scale else None,
                         reference_max_abs=maximum if valid else None, reference_l2=scale if valid else None,
                         near_zero_threshold=threshold, normalization_warnings=warnings,
                         normalization='L2 difference / original L2; null if original norm is zero; near-zero ratios retained with warnings')


def spectral_metadata(data, encoded, k, *, eigen_tol=1e-8, gap_tol=1e-7):
    values = laplacian_spectrum(data)
    positive = values[values > eigen_tol]
    count = min(k, len(positive))
    repeated = [i for i in range(max(0, len(positive)-1)) if float(positive[i+1]-positive[i]) <= gap_tol]
    boundary = len(positive) > k and float(positive[k]-positive[k-1]) <= gap_tol
    near_filter = bool(torch.any((values-eigen_tol).abs() <= min(gap_tol, eigen_tol * .1)))
    q = encoded.x[:, -k:].double()[:, :count]
    ties = [i for i in range(count) if int(torch.isclose(q[:,i].abs(), q[:,i].abs().max(), atol=1e-7, rtol=1e-6).sum()) > 1]
    return dict(eigenvalues=values.tolist(), retained_rank=count, repeated_or_near_positive_pairs=repeated,
                truncated_block=boundary, near_filter=near_filter, sign_anchor_tie_columns=ties,
                eigen_tol=eigen_tol, gap_tol=gap_tol)


def lap_sensitivity(before, after, x_before, x_after, k):
    """Compare raw coordinates descriptively; projector only for complete equal ranks."""
    left, right = spectral_metadata(before, x_before, k), spectral_metadata(after, x_after, k)
    q, r = x_before.x[:, -k:].double(), x_after.x[:, -k:].double()
    raw = difference(q, r) | dict(warning='Unaligned LapPE coordinates: sign/basis changes confound geometric interpretation.')
    supported = left['retained_rank'] == right['retained_rank'] and not any(m['truncated_block'] or m['near_filter'] for m in (left,right))
    if supported:
        projector = difference(q @ q.T, r @ r.T) | dict(warning='Complete selected subspaces; graph changed, so equality is not expected.')
    else:
        projector = dict(status='unsupported', max_absolute_error=None, relative_error=None, l2_change=None,
                         normalized_l2_change=None, tolerance=dict(atol=2e-5, rtol=2e-5),
                         warning='Truncated/near-filter boundary or unequal retained ranks; no comparable complete selected block.')
    # Sign alignment is a separate descriptive diagnostic for simple matched ranks.
    simple = supported and not any(any(i < m['retained_rank'] for i in m['repeated_or_near_positive_pairs']) for m in (left,right))
    if simple:
        signs = (q*r).sum(0).sign()
        signs[signs == 0] = 1
        sign_aligned = difference(q, r*signs) | dict(sign_flipped_columns=(signs < 0).nonzero().flatten().tolist(),
                                                  warning='Column correspondence by ordered simple eigenvalues only; no stability theorem.')
    else:
        sign_aligned = dict(status='unsupported', warning='Repeated, truncated or rank-changing spectrum; sign-only alignment is insufficient.')
    return dict(before=left, after=right, raw_coordinates=raw, projector=projector, sign_aligned=sign_aligned)


def compare(model, original, changed, *, control=False, undo=None):
    graph, nodes = native_outputs(model, original)
    graph_p, nodes_p = native_outputs(model, changed)
    x = changed.x
    if undo is not None:
        x, nodes_p = x[undo], nodes_p[undo]
    metrics = dict(encoding=difference(original.x, x, control=control),
                   node_representation=difference(nodes, nodes_p, control=control),
                   graph_prediction=difference(graph, graph_p, control=control))
    status = 'fail' if any(m['status'] == 'fail' for m in metrics.values()) else 'pass' if control else 'diagnostic'
    return dict(status=status, metrics=metrics)


def run_diagnostics(seeds=(0, 1), model_seed=73):
    if torch.get_num_threads() != 1:
        raise ValueError('one CPU thread required')
    rows = []
    settings_list = [dict(name='none'), dict(name='degree'), dict(name='rwse', walk_length=4),
                     *[dict(name='lappe', k=k, eigen_tol=1e-8) for k in (1,2,4)]]
    for family, graph in small_graphs():
        for settings in settings_list:
            encoder = build_encoding(settings)
            original = encoder(graph)
            for name in ('gin', 'graphgps'):
                config = dict(model=dict(name=name, hidden_dim=8, num_layers=2, dropout=0., pooling='sum'))
                if name == 'graphgps':
                    config['model']['num_heads'] = 2
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(model_seed)
                    model = build_model(config, in_dim=original.x.shape[1], out_dim=2).eval()
                common = dict(graph_family=family, graph=graph_stats(graph), model=name, model_config=config,
                              model_seed=model_seed, encoding=settings, tolerance=dict(atol=2e-5,rtol=2e-5))
                for kind, candidate in [('identical', encoder(graph.clone())), ('repeated_run', original.clone()),
                                         ('permutation', relabel(original, permutation(graph.num_nodes,'random',31)))]:
                    p = permutation(graph.num_nodes,'random',31) if kind == 'permutation' else None
                    rows.append(common | dict(test_type=kind, feature_mode='carried' if p is not None else 'recomputed',
                                              permutation=p.tolist() if p is not None else None) | compare(model, original, candidate, control=True, undo=p))
                for seed in seeds:
                    for kind, count in [('add',1), ('remove',1), ('edits',2)]:
                        changed, edit = perturb(graph, kind, count, seed)
                        for mode in ('frozen','recomputed'):
                            candidate = changed.clone()
                            candidate.x = original.x.clone() if mode == 'frozen' else encoder(changed).x
                            result = compare(model, original, candidate)
                            warning = 'Frozen features are a controlled adjacency-only ablation, not an encoding of the edited graph.' if mode == 'frozen' else 'Recomputed encoding changes with adjacency.'
                            row = common | dict(test_type='edge_perturbation', feature_mode=mode, perturbation=edit,
                                                warning=warning) | result
                            if edit['status'] != 'applied':
                                row['status'] = edit['status']
                            if settings['name'] == 'lappe' and mode == 'recomputed':
                                row['lap_spectral'] = lap_sensitivity(graph, changed, original, candidate, settings['k'])
                                row['warning'] += ' Raw-coordinate node/model changes include arbitrary sign/basis effects; no geometric instability claim.'
                            rows.append(row)
                # Nonconstant tagged features make a deliberately broken alignment detectable.
                tagged = original.clone()
                tagged.x[:,0] = torch.arange(graph.num_nodes, dtype=tagged.x.dtype)
                p = permutation(graph.num_nodes,'reverse')
                wrong = relabel(tagged,p)
                wrong.x = tagged.x.clone()
                measured = compare(model, tagged, wrong, control=True, undo=p)
                detected = measured['metrics']['encoding']['status'] == 'fail'
                rows.append(common | dict(test_type='negative_alignment', feature_mode='deliberately_wrong',
                                          status='pass' if detected else 'fail', expected_failure_detected=detected, node_misalignment_detected=measured['metrics']['node_representation']['status'] == 'fail',
                                          metrics=measured['metrics'], warning='Inner metric FAIL is expected; outer PASS means feature alignment failure detected. A model may erase distinct node features, so a node-level FAIL is not guaranteed.'))
    return rows


def save_diagnostics(rows, directory):
    """Exclusive fresh-directory JSON. Null normalization is explicit, never NaN."""
    import platform
    import torch_geometric
    payload = dict(schema_version=1, interpretation='Descriptive untrained-model sensitivity only; no universal stability score.',
                   environment=dict(python=platform.python_version(),torch=str(torch.__version__),pyg=str(torch_geometric.__version__),threads=torch.get_num_threads()),
                   counts=dict(Counter(row['status'] for row in rows)), rows=rows)
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / 'perturbation_diagnostics.json'
    with path.open('x', encoding='utf-8') as file:
        file.write(serialized)
    return path
