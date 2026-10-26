"""Small CPU permutation diagnostics; never changes training/model implementations.

All permutations map OLD indices to NEW indices. Hooks observe native final node
states; they do not add a node-output API to the production models.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from gnn_expressivity.encodings import build_encoding
from gnn_expressivity.models import build_model


def permutation(n, kind='random', seed=0):
    """Return identity, reverse, or locally seeded random old-to-new indices."""
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError('positive integer node count required')
    if kind == 'identity':
        return torch.arange(n)
    if kind == 'reverse':
        return torch.arange(n - 1, -1, -1)
    if kind != 'random':
        raise ValueError('unknown permutation kind')
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))


def validate_permutation(p, n):
    p = torch.as_tensor(p)
    if p.dtype != torch.long or p.shape != (n,) or not torch.equal(p.sort().values.cpu(), torch.arange(n)):
        raise ValueError('permutation must be a long old-to-new bijection')
    return p


def relabel(data, p, *, node_attrs=(), graph_attrs=('y',)):
    """Clone homogeneous Data/Batch, preserving edge order and graph attributes.

    PyG classifies node/edge attributes; explicit node_attrs resolves ambiguous
    labels (y defaults to graph-level). Non-tensor node attributes and additional
    index-bearing fields are rejected. Batch permutations must remain within each
    graph, retaining sorted membership, ptr, and slice bookkeeping. Encodings must
    be recomputed per graph, not on the union of a batch.
    """
    if not isinstance(data, Data) or data.num_nodes is None or data.num_nodes <= 0:
        raise ValueError('nonempty homogeneous Data/Batch required')
    p = validate_permutation(p, data.num_nodes).to(data.edge_index.device)
    if isinstance(data, Batch) and not torch.equal(data.batch[p], data.batch):
        raise ValueError('Batch permutation must stay within each graph')
    if set(node_attrs) & set(graph_attrs):
        raise ValueError('conflicting attribute roles')
    result = data.clone()
    for key, value in data:
        if key == 'edge_index':
            result[key] = p[value]
        elif key in ('batch', 'ptr', 'num_nodes'):
            continue
        elif key == 'face' or 'index' in key:
            raise ValueError(f'unsupported index-bearing attribute: {key}')
        elif key in graph_attrs:
            continue
        elif key in node_attrs or key in ('x', 'pos') or (not data.is_edge_attr(key) and data.is_node_attr(key)):
            if not isinstance(value, torch.Tensor):
                raise ValueError(f'explicit tensor node attribute required: {key}')
            dim = data.__cat_dim__(key, value)
            if not isinstance(dim, int) or value.shape[dim] != data.num_nodes:
                raise ValueError(f'invalid node attribute: {key}')
            result[key] = value.index_select(dim, p.argsort().to(value.device))
    return result


def error_record(expected, actual, *, atol=2e-5, rtol=2e-5, diagnostic=False, caveat=''):
    """Compare aligned values; nonfinite values always fail, even diagnostics."""
    if expected.shape != actual.shape:
        return dict(max_absolute_error=None, relative_error=None, status='fail',
                    tolerance=dict(atol=atol, rtol=rtol), caveat='shape mismatch')
    finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
    error = float((expected - actual).abs().max()) if finite and expected.numel() else (0.0 if finite else None)
    scale = float(expected.abs().max()) if finite and expected.numel() else 0.0
    close = finite and torch.allclose(expected, actual, atol=atol, rtol=rtol)
    return dict(max_absolute_error=error, relative_error=error / scale if error is not None and scale else None,
                status=('diagnostic' if diagnostic else 'pass') if finite and (close or diagnostic) else 'fail',
                tolerance=dict(atol=atol, rtol=rtol), caveat=caveat)


def native_outputs(model, data):
    """Capture post-dropout GIN or final GPSConv states using a temporary hook."""
    from gnn_expressivity.models import GIN, GraphGPS
    if model.training:
        raise ValueError('model.eval() required')
    if isinstance(model, GIN):
        layer = model.dropout
    elif isinstance(model, GraphGPS):
        layer = model.layers[-1]
    else:
        raise TypeError('native node-state observation unsupported')
    captured = []
    handle = layer.register_forward_hook(lambda module, inputs, output: captured.append(output.detach().clone()))
    try:
        with torch.no_grad():
            graph = model(data).detach().clone()
    finally:
        handle.remove()
    return graph, captured[-1]


def model_checks(model, original, changed, p, *, diagnostic=False, caveat=''):
    p = validate_permutation(p, original.num_nodes)
    graph, nodes = native_outputs(model, original)
    graph_p, nodes_p = native_outputs(model, changed)
    return {
        'graph_invariance': error_record(graph, graph_p, diagnostic=diagnostic, caveat=caveat),
        'node_equivariance': error_record(nodes, nodes_p[p], diagnostic=diagnostic, caveat=caveat),
    }


def laplacian_spectrum(data):
    """Mirror the encoder's normalized Laplacian solely for spectral diagnostics."""
    n = data.num_nodes
    a = torch.zeros(n, n, dtype=torch.float64)
    edges = data.edge_index.cpu()
    a.index_put_((edges[0], edges[1]), torch.ones(edges.shape[1], dtype=torch.float64), accumulate=True)
    if not torch.equal(a, a.T):
        raise ValueError('spectral diagnostics require undirected adjacency')
    degree = a.sum(1)
    inv = degree.clamp_min(1).rsqrt()
    inv[degree == 0] = 0
    return torch.linalg.eigvalsh(torch.diag((degree > 0).double()) - inv[:, None] * a * inv[None, :])


def lappe_checks(data, encoded, recomputed, p, k, *, eigen_tol=1e-8, gap_tol=1e-7):
    """Raw columns are diagnostic. Projectors require a complete selected block.

    Gaps near zero filtering or a truncated repeated/near-repeated block are
    unsupported, not evidence against the encoder. No feature alignment is fed
    back into the native model.
    """
    q, actual = encoded.x[:, -k:].double(), recomputed.x[p, -k:].double()
    spectrum = laplacian_spectrum(data)
    positive = spectrum[spectrum > eigen_tol]
    boundary = len(positive) > k and abs(float(positive[k] - positive[k-1])) <= gap_tol
    near_filter = bool(torch.any((spectrum - eigen_tol).abs() <= min(gap_tol, eigen_tol * 0.1)))
    caveat = 'Sign anchors may tie; repeated eigenvalues permit basis rotation; raw columns are not canonical.'
    raw = error_record(q, actual, diagnostic=True, caveat=caveat)
    if boundary or near_filter:
        projector = dict(max_absolute_error=None, relative_error=None, status='unsupported',
                         tolerance=dict(atol=2e-5, rtol=2e-5),
                         caveat='Truncated repeated/near-repeated eigenvalue block or unstable zero-filter boundary.')
    else:
        projector = error_record(q @ q.T, actual @ actual.T,
                                caveat='Complete retained eigenspace only; does not guarantee raw-feature model invariance.')
    return {'lappe_raw_recomputation': raw, 'lappe_projector': projector}


def small_graphs():
    import networkx as nx
    graphs = [('path', nx.path_graph(5)), ('path_large', nx.path_graph(8)),
              ('cycle', nx.cycle_graph(6)), ('complete', nx.complete_graph(5)),
              ('regular', nx.complete_bipartite_graph(3, 3)), ('star', nx.star_graph(5)),
              ('disconnected', nx.disjoint_union(nx.path_graph(3), nx.empty_graph(2)))]
    for family, graph in graphs:
        edges = list(graph.edges())
        edges += [(v, u) for u, v in edges]
        yield family, Data(num_nodes=graph.number_of_nodes(),
                           edge_index=torch.tensor(edges, dtype=torch.long).T.contiguous())


def run_diagnostics(seeds=(0, 1), model_seed=73):
    """Tiny forward passes only. Caller must set one CPU thread.

    Existing models/encoders are reused; model weights stay fixed across each
    comparison. Forked RNG protects the caller's Torch stream.
    """
    if torch.get_num_threads() != 1:
        raise ValueError('set one CPU thread before running diagnostics')
    rows = []
    encodings = [dict(name='none'), dict(name='degree'), dict(name='rwse', walk_length=4),
                 *[dict(name='lappe', k=k, eigen_tol=1e-8) for k in (1, 2, 4)],
                 dict(name='uid'), dict(name='random', dim=3, seed=29)]
    for family, data in small_graphs():
        for settings in encodings:
            encoder = build_encoding(settings)
            original = encoder(data)
            name = settings['name']
            with torch.random.fork_rng(devices=[]):
                models = {}
                for model_name in ('gin', 'graphgps'):
                    torch.manual_seed(model_seed)
                    config = dict(model=dict(name=model_name, hidden_dim=8, num_layers=2, dropout=0., pooling='sum'))
                    if model_name == 'graphgps':
                        config['model']['num_heads'] = 2
                    models[model_name] = build_model(config, in_dim=original.x.shape[1], out_dim=2).eval()
            for kind, seed in [('identity', None), ('reverse', None), *[('random', seed) for seed in seeds]]:
                p = permutation(data.num_nodes, kind, seed or 0)
                carried = relabel(original, p)
                recomputed = encoder(relabel(data, p))
                metadata = dict(encoding=name, encoding_config=settings, graph_family=family,
                                num_nodes=data.num_nodes, permutation_seed=seed, permutation_kind=kind,
                                permutation=p.tolist(), model_seed=model_seed)
                def add(model, test, outcome):
                    rows.append(metadata | dict(model=model, test_type=test) | outcome)
                diagnostic = name in ('lappe', 'uid', 'random')
                caveat = ('Recomputed index-tied features or ambiguous LapPE basis need not match carried features.'
                          if diagnostic else '')
                add('encoding', 'encoding_recomputation', error_record(original.x, recomputed.x[p], diagnostic=diagnostic, caveat=caveat))
                if name == 'lappe':
                    for test, result in lappe_checks(data, original, recomputed, p, settings['k']).items():
                        add('encoding', test, result)
                for model_name, model in models.items():
                    for test, result in model_checks(model, original, carried, p).items():
                        add(model_name, 'carried_' + test, result)
                    for test, result in model_checks(model, original, recomputed, p, diagnostic=diagnostic, caveat=caveat).items():
                        add(model_name, 'recomputed_' + test, result)
    return rows


def write_diagnostics(rows, output_dir):
    """Exclusive report creation: existing finalized diagnostics are never replaced."""
    import platform
    import torch_geometric
    from collections import Counter
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    payload = dict(schema_version=1, torch=str(torch.__version__), pyg=str(torch_geometric.__version__),
                   python=platform.python_version(), torch_threads=torch.get_num_threads(),
                   counts=dict(Counter(row['status'] for row in rows)), diagnostics=rows)
    path = directory / 'permutation_diagnostics.json'
    with path.open('x', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, allow_nan=False)
    return path
