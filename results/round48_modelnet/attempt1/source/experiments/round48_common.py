"""Small shared identity/configuration helpers, no data access on import."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/round48_modelnet/attempt1'
SEEDS = (11, 22, 33, 44, 55)
CONFIG = dict(epochs=120, batch_size=32, points=512, evaluation_batch_size=16,
              learning_rate=.001, final_learning_rate=.00001, warmup_epochs=5,
              weight_decay=.01, label_smoothing=.1, clip_norm=1.,
              validation_every=5, split_salt='R48-modelnet-train-only-v1',
              sample_salt='R48-evaluation-point-order-v1',
              train_rotation='uniform_Y_0_2pi', jitter_std=.01, jitter_clip=.05,
              shard_seconds=28800, memory_ceiling_bytes=16*2**30)
VARIANTS = ('base', 'y90', 'so3', 'jitter02', 'points256', 'points1024', 'permuted')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def read(path):
    return json.loads(Path(path).read_text())


def hash_array(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def source_paths():
    pending = ['ROUND48_NATIVE_POINT_PE_PROTOCOL.md',
               'models/round48_point_transformer.py',
               'experiments/round48_acquire.py', 'experiments/round48_common.py',
               'experiments/round48_data.py', 'experiments/round48_train.py',
               'experiments/round48_audit.py', 'experiments/round48_queue.py',
               'tests/test_round48_point_pe.py']
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
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                if module.split('.')[0] not in ('models', 'experiments'):
                    continue
                file = module.replace('.', '/')+'.py'
                if (ROOT/file).is_file():
                    pending.append(file)
                package = module.split('.')[0]+'/__init__.py'
                if (ROOT/package).is_file():
                    pending.append(package)
    return sorted(result)


def checked(out=OUT):
    lock = read(out/'lock.json')
    if sha(out/'lock.json') != (out/'lock.sha256').read_text().strip():
        raise ValueError('changed R48 lock')
    for name, digest in lock['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('changed frozen source '+name)
    if lock['config'] != CONFIG:
        raise ValueError('configuration identity')
    return lock
