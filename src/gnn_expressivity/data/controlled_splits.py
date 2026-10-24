"""Task-independent graph records, audited splits and immutable manifests."""
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class GraphRecord:
    graph_id: str
    num_nodes: int
    edges: tuple
    graph_family: str
    target: object
    generation_seed: int
    generator_version: str
    generator_config: dict
    directed: bool = False
    structural_metadata: dict = field(default_factory=dict)
    node_features: object = None

    def __post_init__(self):
        if any(not isinstance(value,str) or not value for value in (self.graph_id,self.graph_family,self.generator_version)):
            raise ValueError("Graph ID, family and generator version are required")
        if type(self.num_nodes) is not int or self.num_nodes < 1 or type(self.directed) is not bool:
            raise ValueError("Positive node count and boolean directed required")
        if type(self.generation_seed) is not int or not isinstance(self.generator_config, dict):
            raise ValueError("Generator seed/config required")
        normalized = []
        for edge in self.edges:
            if len(edge) != 2 or any(type(i) is not int or not 0 <= i < self.num_nodes for i in edge):
                raise ValueError("Invalid edge endpoints")
            u, v = edge
            if u == v:
                raise ValueError("Simple loop-free graphs only; loops need a separately approved schema")
            normalized.append((u, v) if self.directed else tuple(sorted((u, v))))
        if len(set(normalized)) != len(normalized):
            raise ValueError("Duplicate edges: store each undirected edge once")
        object.__setattr__(self, "edges", tuple(sorted(normalized)))
        if self.node_features is not None:
            if len(self.node_features) != self.num_nodes or not self.node_features or not self.node_features[0]:
                raise ValueError("Invalid node features")
            width = len(self.node_features[0])
            if any(len(row) != width or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in row)
                   for row in self.node_features):
                raise ValueError("Node features must be finite and rectangular")
        if not isinstance(self.structural_metadata,dict):
            raise ValueError('Structural metadata must be a mapping')
        canonical(asdict(self))

    @property
    def num_edges(self):
        return len(self.edges)

    @property
    def graph_density(self):
        possible = self.num_nodes * (self.num_nodes - 1)
        if not self.directed:
            possible /= 2
        return self.num_edges / possible if possible else 0.0

    def graph(self):
        graph = nx.DiGraph() if self.directed else nx.Graph()
        graph.add_nodes_from(range(self.num_nodes))
        graph.add_edges_from(self.edges)
        return graph

    def to_dict(self):
        return asdict(self) | {"num_edges": self.num_edges, "graph_density": self.graph_density}

    @classmethod
    def from_dict(cls, data):
        row = dict(data)
        n_edges, density = row.pop("num_edges", None), row.pop("graph_density", None)
        record = cls(**row)
        if n_edges is not None and n_edges != record.num_edges:
            raise ValueError("Recorded edge count disagrees with graph")
        if density is not None and not math.isclose(density, record.graph_density):
            raise ValueError("Recorded density disagrees with graph")
        return record


def load_records(path):
    return [GraphRecord.from_dict(row) for row in json.loads(Path(path).read_text(encoding="utf-8"))]


def dataset_hash(records):
    return digest([r.to_dict() for r in sorted(records, key=lambda r: r.graph_id)])


def validate_records(records, max_isomorphism_checks=100000):
    if not records or len({r.graph_id for r in records}) != len(records):
        raise ValueError("Nonempty records with unique graph IDs required")
    buckets = defaultdict(list)
    checks = 0
    for record in sorted(records, key=lambda r: r.graph_id):
        graph = record.graph()
        degrees = sorted((graph.in_degree(n), graph.out_degree(n)) for n in graph) if record.directed else sorted(dict(graph.degree()).values())
        key = (record.directed, record.num_nodes, record.num_edges, tuple(degrees))
        for prior, prior_graph in buckets[key]:
            checks += 1
            if checks > max_isomorphism_checks:
                raise ValueError("Isomorphism audit budget exceeded; no unchecked manifest is approved")
            if nx.is_isomorphic(graph, prior_graph):
                raise ValueError(f"Isomorphic/underlying duplicate graphs: {prior.graph_id}, {record.graph_id}")
        buckets[key].append((record, graph))
    return {"isomorphism_checks": checks, "policy": "exact topology-only; duplicates rejected globally"}


def distribution(records):
    def stats(values):
        return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)} if values else None
    return {"count": len(records), "label_distribution": dict(Counter(canonical(r.target) for r in records)),
            "num_nodes": stats([r.num_nodes for r in records]),
            "density": stats([r.graph_density for r in records]),
            "families": dict(Counter(r.graph_family for r in records))}


def _partition(records, seed, fractions, stratify):
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or not math.isclose(sum(fractions), 1):
        raise ValueError("Three positive fractions summing to one required")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for record in sorted(records, key=lambda r: r.graph_id):
        if stratify and record.target is None:
            raise ValueError("Stratification requires known classification labels")
        groups[canonical(record.target) if stratify else "all"].append(record.graph_id)
    splits = {name: [] for name in ("train", "validation", "iid_test")}
    for label in sorted(groups):
        ids = groups[label]
        rng.shuffle(ids)
        a, b = int(len(ids) * fractions[0]), int(len(ids) * fractions[1])
        if min(a, b, len(ids) - a - b) < 1:
            raise ValueError("Too few records per stratum for nonempty train/validation/IID test")
        for name, values in zip(splits, (ids[:a], ids[a:a+b], ids[a+b:])):
            splits[name].extend(values)
    return {name: sorted(ids) for name, ids in splits.items()}


def _range(value, bounds):
    return bounds[0] <= value < bounds[1]


def _seal(manifest):
    manifest = dict(manifest)
    manifest.pop("manifest_hash", None)
    return manifest | {"manifest_hash": digest(manifest)}


def create_manifest(records, *, task_id, dataset_version, split_type, seed,
                    fractions=(0.6, 0.2, 0.2), stratify=False, regimes=None,
                    max_isomorphism_checks=100000):
    if split_type not in ("iid", "ood_size", "ood_family", "ood_density"):
        raise ValueError("Unknown split type")
    if not task_id or not dataset_version or type(seed) is not int:
        raise ValueError("Task, dataset version and integer split seed required")
    audit = validate_records(records, max_isomorphism_checks)
    regimes = regimes or {}
    id_records, ood_records, excluded = [], [], []
    if split_type == "iid":
        id_records = list(records)
    else:
        if "id" not in regimes or "ood" not in regimes:
            raise ValueError("Explicit ID and OOD regimes required")
        if split_type == "ood_family":
            if not regimes['id'] or not regimes['ood'] or set(regimes['id']) & set(regimes['ood']):
                raise ValueError("Nonempty disjoint family regimes required")
        else:
            for bounds in (regimes['id'], regimes['ood']):
                if len(bounds) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bounds) or bounds[0] >= bounds[1]:
                    raise ValueError("Finite nonempty half-open ranges required")
            if max(regimes['id'][0], regimes['ood'][0]) < min(regimes['id'][1], regimes['ood'][1]):
                raise ValueError("ID/OOD ranges overlap")
        for record in records:
            value = record.num_nodes if split_type == 'ood_size' else record.graph_density
            is_id = record.graph_family in regimes['id'] if split_type == 'ood_family' else _range(value, regimes['id'])
            is_ood = record.graph_family in regimes['ood'] if split_type == 'ood_family' else _range(value, regimes['ood'])
            (id_records if is_id else ood_records if is_ood else excluded).append(record)
        if not ood_records:
            raise ValueError("No records in held-out OOD regime")
    splits = _partition(id_records, seed, fractions, stratify)
    if ood_records:
        splits[split_type + '_test'] = sorted(r.graph_id for r in ood_records)
    lookup = {r.graph_id: r for r in records}
    config = dict(split_type=split_type, seed=seed, fractions=list(fractions), stratify=stratify, regimes=regimes, max_isomorphism_checks=max_isomorphism_checks)
    observed = {name: distribution([lookup[i] for i in ids]) for name, ids in splits.items()}
    shifts = []
    if ood_records:
        shifts = ["OOD regimes can co-shift size, density, family and labels; inspect observed distributions; causal isolation is not established"]
    generators = {digest({"version": r.generator_version, "configuration": r.generator_config}):
                  {"version": r.generator_version, "configuration": r.generator_config} for r in records}
    shifts_observed = {}
    if ood_records:
        id_distribution = distribution(id_records)
        ood_distribution = distribution(ood_records)
        shifts_observed = {key+'_mean_ood_minus_id': ood_distribution[key]['mean'] - id_distribution[key]['mean'] for key in ('num_nodes','density')}
    return _seal(dict(schema_version=1, task_id=task_id, dataset_version=dataset_version,
        dataset_hash=dataset_hash(records), split_type=split_type, split_seed=seed,
        configuration=config, configuration_hash=digest(config), splits=splits,
        excluded_ids=sorted(r.graph_id for r in excluded), observed=observed,
        generator_metadata=generators, graph_generation_seeds={r.graph_id: r.generation_seed for r in records},
        leakage_audit=audit, shift_warnings=shifts, observed_secondary_shifts=shifts_observed, validation_domain='id_only',
        approval={"status": "draft", "approved_by": None, "reference": None}))


def validate_manifest(manifest, records, *, require_approved=False):
    if manifest.get('manifest_hash') != _seal(manifest)['manifest_hash']:
        raise ValueError("Manifest was modified after sealing")
    if manifest['configuration_hash'] != digest(manifest['configuration']) or manifest['dataset_hash'] != dataset_hash(records):
        raise ValueError("Dataset/configuration hash mismatch")
    validate_records(records, manifest['configuration']['max_isomorphism_checks'])
    all_ids = {r.graph_id for r in records}
    seen = set()
    for name, ids in manifest['splits'].items():
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= all_ids or seen & set(ids):
            raise ValueError("Unknown/duplicate/overlapping split IDs")
        seen.update(ids)
    if not {'train', 'validation', 'iid_test'} <= set(manifest['splits']):
        raise ValueError("Required splits absent")
    if seen & set(manifest['excluded_ids']) or seen | set(manifest['excluded_ids']) != all_ids:
        raise ValueError("Incomplete dataset membership accounting")
    # Recompute the deterministic split to reject rehashed, internally inconsistent manifests.
    config = manifest['configuration']
    rebuilt = create_manifest(records, task_id=manifest['task_id'], dataset_version=manifest['dataset_version'],
        split_type=config['split_type'], seed=config['seed'], fractions=config['fractions'],
        stratify=config['stratify'], regimes=config['regimes'], max_isomorphism_checks=config['max_isomorphism_checks'])
    for key in ('splits', 'excluded_ids', 'observed', 'generator_metadata', 'graph_generation_seeds', 'split_type', 'split_seed', 'validation_domain', 'observed_secondary_shifts'):
        if manifest[key] != rebuilt[key]:
            raise ValueError(f"Manifest content inconsistent: {key}")
    approval = manifest['approval']
    if require_approved and (approval['status'] != 'approved' or not approval['approved_by'] or not approval['reference']):
        raise ValueError("Approved task/split evidence required before training")
    return manifest


def approve_manifest(manifest, records, *, approved_by, reference):
    validate_manifest(manifest, records)
    if manifest['approval']['status'] != 'draft' or not approved_by or not reference:
        raise ValueError("A draft and explicit approval evidence are required")
    return _seal(manifest | {'approval': {'status': 'approved', 'approved_by': approved_by, 'reference': reference}})


def write_manifest(manifest, path):
    if manifest.get('manifest_hash') != _seal(manifest)['manifest_hash']:
        raise ValueError("Unsealed manifest")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as file:
        file.write(json.dumps(manifest, indent=2, allow_nan=False))


def nested_training_subsets(manifest, records, sizes, *, seed, balanced=False):
    validate_manifest(manifest, records, require_approved=True)
    train = manifest['splits']['train']
    if not sizes or sizes != sorted(set(sizes)) or any(type(n) is not int or not 0 < n <= len(train) for n in sizes):
        raise ValueError("Explicit increasing feasible sample counts required")
    rng = random.Random(seed)
    lookup = {r.graph_id: r for r in records}
    pools = defaultdict(list)
    for i in sorted(train):
        pools[canonical(lookup[i].target) if balanced else 'all'].append(i)
    if balanced and any(lookup[i].target is None for i in train):
        raise ValueError("Balanced subsets require labels")
    for ids in pools.values():
        rng.shuffle(ids)
    labels = sorted(pools)
    rng.shuffle(labels)
    ordered = []
    while any(pools.values()):
        for label in labels:
            if pools[label]:
                ordered.append(pools[label].pop())
    return [dict(sample_count=n, actual_sample_count=n, subset_seed=seed, balanced=balanced,
        train_ids=ordered[:n], parent_manifest_hash=manifest['manifest_hash'],
        fixed_evaluation_hash=digest({k: v for k, v in manifest['splits'].items() if k != 'train'}),
        label_distribution=dict(Counter(canonical(lookup[i].target) for i in ordered[:n]))) for n in sizes]
