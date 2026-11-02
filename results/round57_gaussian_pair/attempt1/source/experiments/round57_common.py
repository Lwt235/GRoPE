"""R57 identities; scientific optimizer/data configuration stays at R48."""
import ast
from pathlib import Path
from experiments.round48_common import ROOT, CONFIG, VARIANTS, sha, read, write, checked as checked48
from experiments.round56_common import OUT as REFERENCE_OUT, model_path as reference_model_path, atomic_json
from models.round57_gaussian_pair import NEW_ROLES, REFERENCE_ROLES, ROLES, INTERVENTIONS, DIAGNOSTIC_ROLES

OUT = ROOT/'results/round57_gaussian_pair/attempt1'
SEEDS = (11, 22, 33)
PRIMARY = 'pair_full'


def source_paths():
    pending = ['ROUND57_GAUSSIAN_PAIR_PROTOCOL.md', 'models/round57_gaussian_pair.py',
               'experiments/round57_common.py', 'experiments/round57_prepare.py',
               'experiments/round57_train.py', 'experiments/round57_audit.py',
               'experiments/round57_queue.py', 'tests/test_round57_gaussian_pair.py']
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
        raise ValueError('R57 role/seed identity')
    for path, digest in lock['input_identities'].items():
        if sha(path) != digest:
            raise ValueError('R57 frozen input identity: '+path)
    return lock


def reference(role, seed, lock):
    return next(r for r in lock['references'] if r['role'] == role and r['seed'] == seed)


def model_path(role, seed, out=OUT):
    return out/'runs'/role/str(seed) if role in NEW_ROLES else reference_model_path(role, seed)
