"""Exact R48 sampling stream and serializable optimizer continuation."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from experiments.round48_common import CONFIG


def epoch_batches(rng, count, stream_hash, batch_size=None, points=None):
    batch_size = CONFIG['batch_size'] if batch_size is None else batch_size
    points = CONFIG['points'] if points is None else points
    order = rng.permutation(count)
    stream_hash.update(order.astype('<i8').tobytes())
    for begin in range(0, count, batch_size):
        index = order[begin:begin+batch_size]
        subset = np.stack([rng.choice(2048, points, replace=False) for _ in index])
        angle = rng.uniform(0, 2*np.pi, len(index)).astype(np.float32)
        jitter = rng.normal(0, CONFIG['jitter_std'], (len(index), points, 3))
        jitter = jitter.clip(-CONFIG['jitter_clip'], CONFIG['jitter_clip']).astype(np.float32)
        for value in (index.astype('<i8'), subset.astype('<i8'), angle, jitter):
            stream_hash.update(value.tobytes())
        yield index, subset, angle, jitter


def replay_sampler(seed, epochs, count):
    rng, digest = np.random.default_rng(seed+48000), hashlib.sha256()
    for _ in range(epochs):
        for _batch in epoch_batches(rng, count, digest):
            pass
    return rng, digest


def atomic_save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def snapshot(model, optimizer):
    return dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [])


def restore(model, optimizer, state):
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
