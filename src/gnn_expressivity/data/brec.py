"""Official BREC v3 ingestion, without training or generated graph pairs.

Layout follows GraphPKU/BREC customize/dataset_v3.py at SOURCE_COMMIT:
400 blocks of 64 interleaved A/B records (32 relabelings), then 400 blocks
of 64 isomorphic reliability records. Default iteration selects relabeling 0.
The raw graph6 array has no node features, class labels, or per-record metadata;
categories/families and record offsets come from the official generation code.
"""

import hashlib
import json
import tempfile
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any
from urllib.request import urlopen

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

SOURCE_COMMIT = "d09e8c349a8bbc0882d2932f7b37b2726f576ce9"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/GraphPKU/BREC/{SOURCE_COMMIT}/BREC_data_all.zip"
)
ARCHIVE_SHA256 = "3a2c8e7ba068f774b1115422d1bc04749432dde9f19631e6625cba142659530f"
DATA_SHA256 = "6975d172d27aedcdf4eb2c747450ede7928c111caeb2689fbc67e6b62fbd5562"
DEFAULT_ROOT = Path("data/raw/brec")
NUM_PAIRS = 400
NUM_RELABELINGS = 32
RECORDS_PER_PAIR = 2 * NUM_RELABELINGS
NUM_RECORDS = 2 * NUM_PAIRS * RECORDS_PER_PAIR
CATEGORY_RANGES = (
    ("Basic", 0, 60),
    ("Regular", 60, 160),
    ("Extension", 160, 260),
    ("CFI", 260, 360),
    ("4-Vertex_Condition", 360, 380),
    ("Distance_Regular", 380, 400),
)


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    # Close before replace: Windows cannot rename an open temporary file.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(payload)
        except BaseException:
            file.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def download_brec(root: str | Path = DEFAULT_ROOT) -> Path:
    """Fetch the pinned official archive and verify both SHA-256 digests.

    Only brec_v3.npy is extracted. Valid local downloads are reused without
    network access. A provenance manifest is saved beside the data.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "brec_v3.npy"
    if not target.exists() or _sha256(target) != DATA_SHA256:
        archive = root / "BREC_data_all.zip"
        if not archive.exists() or _sha256(archive) != ARCHIVE_SHA256:
            with urlopen(SOURCE_URL, timeout=60) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA256:
                raise ValueError("Official BREC archive SHA-256 mismatch")
            _write_atomic(archive, payload)
        with zipfile.ZipFile(archive) as bundle:
            payload = bundle.read("brec_v3.npy")
        if hashlib.sha256(payload).hexdigest() != DATA_SHA256:
            raise ValueError("Official brec_v3.npy SHA-256 mismatch")
        _write_atomic(target, payload)
    manifest = {
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "data_sha256": DATA_SHA256,
        "filename": target.name,
        "num_pairs": NUM_PAIRS,
        "num_records": NUM_RECORDS,
        "num_relabelings": NUM_RELABELINGS,
    }
    _write_atomic(root / "source.json", (json.dumps(manifest, indent=2) + "\n").encode())
    return target


def _category(pair_id: int) -> str:
    return next(name for name, start, stop in CATEGORY_RANGES if start <= pair_id < stop)


def _family(pair_id: int) -> str:
    if 60 <= pair_id < 110:
        return "Simple Regular"
    if 110 <= pair_id < 160:
        return "Strongly Regular"
    return _category(pair_id)


@dataclass(frozen=True)
class BRECPair:
    """One benchmark identity and exactly two CPU PyG graphs.

    pair_id is the official zero-based benchmark index, independent of the
    chosen relabeling. Metadata contains provenance and official file offsets.
    """

    pair_id: int
    graph_a: Data
    graph_b: Data
    category: str
    family: str
    metadata: Mapping[str, Any]

    @property
    def graphs(self) -> tuple[Data, Data]:
        return self.graph_a, self.graph_b


def _decode_graph(record: bytes, index: int) -> Data:
    try:
        graph = nx.from_graph6_bytes(record)
    except (ValueError, IndexError, nx.NetworkXException) as error:
        raise ValueError(f"Malformed graph6 at raw record {index}: {error}") from error
    # Explicit num_nodes preserves isolates. Store both directions for PyG.
    edges = sorted((u, v) for a, b in graph.edges for u, v in ((a, b), (b, a)))
    edge_index = torch.tensor(edges, dtype=torch.long, device="cpu").reshape(-1, 2).t().contiguous()
    data = Data(edge_index=edge_index, num_nodes=graph.number_of_nodes())
    validate_graph(data, context=f"raw record {index}")
    return data


class BRECDataset(Sequence[BRECPair]):
    """Lazy, deterministic CPU loader of the 400 official benchmark pairs.

    No implicit downloads, random relabeling, transforms, or features. Each
    access returns fresh Data objects. verify_checksum=False is for fixtures
    and explicitly modified data only; official shape/dtype remain required.
    """

    def __init__(self, root: str | Path = DEFAULT_ROOT, *, verify_checksum: bool = True):
        self.path = Path(root) / "brec_v3.npy"
        if not self.path.is_file():
            raise FileNotFoundError(
                f"BREC file missing: {self.path}. Run uv run python scripts/download_brec.py"
            )
        digest = _sha256(self.path)
        if verify_checksum and digest != DATA_SHA256:
            raise ValueError("brec_v3.npy SHA-256 mismatch; use the pinned official download")
        # Memory-map the 167 MB fixed-width byte array; never unpickle input.
        self._records = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if self._records.shape != (NUM_RECORDS,) or self._records.dtype.kind != "S":
            raise ValueError(f"Expected {NUM_RECORDS} graph6 byte records in a 1D array")
        self.sha256 = digest

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "BRECDataset":
        """Accept the existing nested task config (or a merged config)."""
        task = config["task"]
        if task["name"] != "brec" or task["dataset"] != "BREC":
            raise ValueError("Expected task.name=brec and task.dataset=BREC")
        return cls(Path(task.get("data_root", DEFAULT_ROOT)))

    def __len__(self) -> int:
        return NUM_PAIRS

    def __getitem__(self, index: int | slice) -> BRECPair | list[BRECPair]:
        if isinstance(index, slice):
            return [self.get_pair(i) for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        return self.get_pair(index)

    def get_pair(self, pair_id: int, relabeling: int = 0) -> BRECPair:
        """Load a specific official A/B relabeling, from 0 through 31."""
        if not 0 <= pair_id < NUM_PAIRS:
            raise IndexError(f"BREC pair ID out of range: {pair_id}")
        if not 0 <= relabeling < NUM_RELABELINGS:
            raise IndexError(f"BREC relabeling out of range: {relabeling}")
        start = pair_id * RECORDS_PER_PAIR + 2 * relabeling
        reliability_start = (NUM_PAIRS + pair_id) * RECORDS_PER_PAIR
        return BRECPair(
            pair_id=pair_id,
            graph_a=_decode_graph(bytes(self._records[start]), start),
            graph_b=_decode_graph(bytes(self._records[start + 1]), start + 1),
            category=_category(pair_id),
            family=_family(pair_id),
            metadata={
                "source_commit": SOURCE_COMMIT,
                "source_file": self.path.name,
                "source_sha256": self.sha256,
                "relabeling": relabeling,
                "raw_indices": (start, start + 1),
                "reliability_range": (reliability_start, reliability_start + RECORDS_PER_PAIR),
            },
        )

    def get_reliability_graph(self, pair_id: int, variant: int = 0) -> Data:
        """Access a preserved official reliability record (0..63).

        These are relabelings of one randomly selected member of the original
        pair. The archive does not record whether that member was A or B.
        """
        if not 0 <= pair_id < NUM_PAIRS or not 0 <= variant < RECORDS_PER_PAIR:
            raise IndexError("BREC reliability pair ID or variant out of range")
        index = (NUM_PAIRS + pair_id) * RECORDS_PER_PAIR + variant
        return _decode_graph(bytes(self._records[index]), index)


def validate_graph(graph: Data, *, context: str = "graph") -> None:
    """Check node/edge bounds and the simple undirected BREC representation."""
    n = graph.num_nodes
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError(f"{context}: num_nodes must be a positive integer")
    edges = graph.edge_index
    if not isinstance(edges, torch.Tensor) or edges.dtype != torch.long:
        raise ValueError(f"{context}: edge_index must be a torch.long tensor")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"{context}: edge_index must have shape [2, E]")
    if edges.device.type != "cpu":
        raise ValueError(f"{context}: ingestion graphs must reside on CPU")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= n):
        raise ValueError(f"{context}: edge index outside node bounds")
    entries = [tuple(edge) for edge in edges.t().tolist()]
    unique = set(entries)
    if len(unique) != len(entries) or any(u == v or (v, u) not in unique for u, v in unique):
        raise ValueError(f"{context}: expected simple undirected edges stored in both directions")


def validate_brec(pairs: Sequence[BRECPair]) -> None:
    """Validate all 400 representative pairs; raise ValueError on first failure.

    This checks ingestion structure, not graph non-isomorphism or RPC results.
    The default loader's checksum verifies all raw records against the source.
    """
    if len(pairs) != NUM_PAIRS:
        raise ValueError(f"Expected {NUM_PAIRS} BREC pairs, got {len(pairs)}")
    seen = set()
    for pair in pairs:
        if pair.pair_id in seen:
            raise ValueError(f"Duplicate BREC pair ID: {pair.pair_id}")
        if not isinstance(pair.pair_id, int) or not 0 <= pair.pair_id < NUM_PAIRS:
            raise ValueError(f"Invalid BREC pair ID: {pair.pair_id}")
        seen.add(pair.pair_id)
        if len(pair.graphs) != 2:
            raise ValueError(f"Pair {pair.pair_id} must contain exactly two graphs")
        if pair.category != _category(pair.pair_id) or pair.family != _family(pair.pair_id):
            raise ValueError(f"Pair {pair.pair_id} has incorrect category/family")
        for graph in pair.graphs:
            validate_graph(graph, context=f"pair {pair.pair_id}")


def summarize_brec(pairs: Sequence[BRECPair]) -> dict[str, Any]:
    """Validate and summarize the 800 representative graphs, counting edges once."""
    pairs = list(pairs)
    validate_brec(pairs)
    nodes = [graph.num_nodes for pair in pairs for graph in pair.graphs]
    edges = [graph.num_edges // 2 for pair in pairs for graph in pair.graphs]

    def stats(values: list[int]) -> dict[str, int | float]:
        return {"min": min(values), "median": median(values), "max": max(values)}

    return {
        "total_pairs": len(pairs),
        "total_graphs": len(nodes),
        "pairs_per_category": dict(Counter(pair.category for pair in pairs)),
        "pairs_per_family": dict(Counter(pair.family for pair in pairs)),
        "nodes": stats(nodes),
        "undirected_edges": stats(edges),
    }
