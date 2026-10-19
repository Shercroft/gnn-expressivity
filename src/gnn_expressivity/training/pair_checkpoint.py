"""Atomic pair-boundary snapshots, including the sequential experiment RNG stream."""

import hashlib
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def source_digest():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ('models', 'encodings', 'data', 'training'):
        for path in sorted((root / folder).glob('*.py')):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def rng_state():
    numpy = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': (numpy[0], numpy[1].tolist(), *numpy[2:]),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'mps': torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    if state['mps'] is not None:
        torch.mps.set_rng_state(state['mps'])


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            writer(file)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            file.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path, identity, records, test, controls, parameters, times):
    state = {'version': 1, 'status': 'partial', 'full_benchmark_completed': False,
             'benchmark_valid': False, 'identity': identity, 'records': records,
             'test': [torch.from_numpy(a) for a in test],
             'controls': [torch.from_numpy(a) for a in controls],
             'parameters': parameters, 'times': times, 'rng': rng_state()}
    atomic_write(path, lambda file: torch.save(state, file))


def load_checkpoint(path, identity):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('version') != 1 or state.get('identity') != identity or state.get('status') != 'partial':
        raise ValueError('Checkpoint configuration/seed/source/protocol/environment/range mismatch')
    records = state['records']
    expected = identity['pair_ids'][:len(records)]
    if len(records) > len(identity['pair_ids']) or [r['pair_id'] for r in records] != expected:
        raise ValueError('Checkpoint must contain a contiguous completed prefix of the selected range')
    for key in ('test', 'controls'):
        arrays = state[key]
        count = len(records) if identity['save_embeddings'] else 0
        if len(arrays) != count or any(a.shape != (64, 16) or not torch.isfinite(a).all() for a in arrays):
            raise ValueError('Checkpoint embeddings missing or corrupt')
    return state
