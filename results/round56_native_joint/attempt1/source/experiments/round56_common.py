"""R56 identity helpers; keep the R48 scientific training configuration exact."""
import ast
import json
from pathlib import Path
from experiments.round48_common import (ROOT, CONFIG, VARIANTS, sha, read, write,
                                        checked as checked48, OUT as REFERENCE_OUT)
from models.round56_native_joint import NEW_ROLES, REFERENCE_ROLES, ROLES

OUT = ROOT/'results/round56_native_joint/attempt1'
SEEDS = (11, 22, 33)
PRIMARY = 'joint_conditional'


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    write(tmp, value)
    tmp.replace(path)


def source_paths():
    pending = ['ROUND56_NATIVE_JOINT_PROTOCOL.md', 'models/round56_native_joint.py',
               'experiments/round56_common.py', 'experiments/round56_prepare.py',
               'experiments/round56_optimization.py', 'experiments/round56_train.py',
               'experiments/round56_audit.py', 'experiments/round56_queue.py',
               'tests/test_round56_native_joint.py']
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
            modules = [node.module] if isinstance(node, ast.ImportFrom) and node.module else (
                [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
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
    lock = checked48(out)
    if lock['roles'] != list(ROLES) or lock['seeds'] != list(SEEDS):
        raise ValueError('R56 roles/seeds changed')
    for name, digest in lock['input_identities'].items():
        if sha(name) != digest:
            raise ValueError('R56 input identity changed: '+name)
    return lock


def reference(role, seed, lock):
    return next(r for r in lock['references'] if r['role'] == role and r['seed'] == seed)


def model_path(role, seed, out=OUT):
    return (out if role in NEW_ROLES else REFERENCE_OUT)/'runs'/role/str(seed)
