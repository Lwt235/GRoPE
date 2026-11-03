"""Fixed experiment settings and identity helpers; no arrays on import."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/round65_actual_gs/attempt1'
PREP = ROOT/'results/round65_actual_gs/preparation'
DATA_ROOT = Path('/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data')
RAW = DATA_ROOT/'GSModel60'
CACHE_ROOT = DATA_ROOT/'GRoPE/round65_actual_gs_cache'
SEEDS = (11, 22, 33)
ROLES = ('gaussian_relation', 'direct_mlp', 'distance_mlp', 'center_rope', 'nope')
VARIANTS = ('base', 'sparse256', 'dense1024', 'jitter02', 'tilt', 'permuted')
CONFIG = dict(epochs=120, batch_size=16, tokens=512, cache_tokens=2048,
              evaluation_batch_size=4, learning_rate=.001, final_learning_rate=.00001,
              warmup_epochs=5, weight_decay=.01, label_smoothing=.1, clip_norm=1.,
              validation_every=5, covariance_floor=1e-8,
              split_salt='R65-gs-train-split-v1', sample_salt='R65-native-cache-v1',
              train_rotation='uniform_SO3', jitter_std=.01, jitter_clip=.05,
              minimum_gain_pp=.5, bootstrap_seed=65001, bootstrap_replicates=10000)
CONFIG.update(shard_seconds=28800,memory_ceiling_bytes=32*2**30)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def identity_seed(salt, identity):
    return int.from_bytes(hashlib.sha256((salt+':'+identity).encode()).digest()[:8], 'little')


def source_paths():
    pending = ['ROUND65_ACTUAL_GS_PROTOCOL.md', 'models/round65_actual_gs.py',
               'experiments/round65_acquire.py', 'experiments/round65_common.py',
               'experiments/round65_data.py', 'experiments/round65_run.py',
               'experiments/round65_audit.py', 'experiments/round65_queue.py',
               'tests/test_round65_actual_gs.py', 'tests/test_round65_acquire.py',
               'tests/test_round65_execution.py']
    result = set()
    while pending:
        name = pending.pop()
        if name in result:
            continue
        result.add(name)
        path = ROOT/name
        if path.suffix != '.py':
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(x.name for x in node.names)
            for module in modules:
                if module.split('.')[0] not in ('models', 'experiments'):
                    continue
                local = module.replace('.', '/')+'.py'
                if (ROOT/local).is_file():
                    pending.append(local)
                init = module.split('.')[0]+'/__init__.py'
                if (ROOT/init).is_file():
                    pending.append(init)
    return sorted(result)


def checked(out=OUT):
    lock = read(out/'lock.json')
    if sha(out/'lock.json') != (out/'lock.sha256').read_text().strip() or lock['config'] != CONFIG:
        raise ValueError('frozen experiment identity changed')
    for name, digest in lock['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('frozen source changed '+name)
    if sha(PREP/'train_catalog.json') != lock['catalog_sha256']:
        raise ValueError('data population changed')
    return lock
