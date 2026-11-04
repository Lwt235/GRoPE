"""Fixed scope and identity helpers for inference on unused regional scenes."""
from pathlib import Path
import hashlib
import json

from experiments.round65_common import ROOT, read, write, sha

OUT = ROOT / 'results/region_confirmation/attempt1'
PREP = ROOT / 'results/region_confirmation/preparation'
PLAN = ROOT / 'docs/region_confirmation_plan.md'
DATA = Path('/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/region_confirmation_20261105')
RAW, CACHE = DATA / 'raw', DATA / 'cache'
INVENTORY = ROOT / 'results/research_data_usage/region_confirmation_inventory_20261105.json'
PARENT = ROOT / 'results/round68_fixed_region/attempt1'
CANDIDATE = ROOT / 'results/round69_compact_relation/attempt1'
GEOMETRY = ROOT / 'results/region_geometry_controls/attempt1'
FIXED = ROOT / 'results/region_shape_attribution/attempt1'
SEEDS = (11, 22, 33)
CONDITIONS = ('base', 'sparse256')
ROLES = ('relative5', 'distance_mlp', 'direct_mlp', 'normal_scale', 'gaussian_relation',
         'geotransformer_rpe', 'angle_descriptor_mlp', 'fixed_isotropic5')
UTILITY_CONTROLS = ('distance_mlp', 'direct_mlp', 'normal_scale', 'geotransformer_rpe', 'angle_descriptor_mlp')
SALT = 'GRoPE-region-confirmation-20261105-v1'
PREFIXES = ('brown', 'harvard', 'home', 'hotel', 'mit')


def preserved(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == json.loads(json.dumps(value)), 'preserve existing artifact: ' + str(path)
    else:
        write(path, value)


def select_places(places, excluded):
    remaining = set(places) - set(excluded)
    result = []
    for prefix in PREFIXES:
        candidates = [p for p in remaining if p.startswith(prefix + '_')]
        candidates.sort(key=lambda p: (hashlib.sha256((SALT + ':' + p).encode()).hexdigest(), p))
        assert len(candidates) >= 2, prefix
        result.extend(candidates[:2])
    assert len(result) == len(set(result)) == 10
    return result


def run_root(role):
    assert role in ROLES
    if role == 'relative5':
        return CANDIDATE
    if role in ('geotransformer_rpe', 'angle_descriptor_mlp'):
        return GEOMETRY
    if role == 'fixed_isotropic5':
        return FIXED
    return PARENT


def check_lock(path):
    path = Path(path)
    assert sha(path) == path.with_suffix('.sha256').read_text().strip(), str(path)
    lock = read(path)
    for name, digest in lock['source_sha256'].items():
        assert sha(name) == digest, name
    return lock


def check():
    lock = check_lock(OUT / 'evaluation_lock.json')
    assert lock['roles'] == list(ROLES) and lock['seeds'] == list(SEEDS)
    assert lock['conditions'] == list(CONDITIONS) and lock['pixels_access_permitted']
    for record in lock['checkpoints'].values():
        assert sha(record['path']) == record['sha256'], record['path']
    return lock
