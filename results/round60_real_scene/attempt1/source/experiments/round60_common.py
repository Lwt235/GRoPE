"""R60 fixed real-scene scope and original source-model identities."""
from pathlib import Path
from experiments.round58_common import ROOT, CONFIG as BASE_CONFIG, SEEDS, read, write, sha
from experiments.round59_common import SOURCES as PRIOR_SOURCES, OUT as PRIOR_OUT, run_path
from models.round59_relation_controls import ROLES

OUT = ROOT/'results/round60_real_scene/attempt1'
RAW = ROOT/'data/round5_sun3d'
PAIR_LOCK = ROOT/'results/round46_native3d/attempt1/lock.json'
CONDITIONS = ('base', 'sparse256')
PHASES = ('source', 'adapted')
CONFIG = dict(BASE_CONFIG, lr=1e-4, final_lr=1e-6, data_salt='R60-native-observed-support-v1')
SCENES = ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1',
          'harvard_c8/hv_c8_3', 'harvard_c11/hv_c11_2', 'hotel_umd/maryland_hotel3',
          'mit_76_studyroom/76-1studyroom2', 'mit_dorm_next_sj/dorm_next_sj_oct_30_2012_scan1_erika',
          'mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika')
CACHE_NAMES = ('fit0', 'fit1', 'val', 'eval_base', 'eval_sparse256')
SOURCES = tuple(sorted(set(PRIOR_SOURCES) | {
    'ROUND60_REAL_SCENE_PROTOCOL.md', 'docs/round60_real_scene_plan.md',
    'experiments/round60_common.py', 'experiments/round60_data.py',
    'experiments/round60_run.py', 'experiments/round60_audit.py',
    'experiments/round60_queue.py', 'experiments/round5_rgbd_data.py',
    'tests/test_round60_real_scene.py'}))


def checked():
    lock = read(OUT/'lock.json')
    if sha(OUT/'lock.json') != (OUT/'lock.sha256').read_text().strip():
        raise ValueError('R60 lock identity')
    if lock['config'] != CONFIG or lock['roles'] != list(ROLES) or lock['seeds'] != list(SEEDS):
        raise ValueError('R60 scope')
    for name, digest in lock['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('changed frozen source '+name)
    for name, digest in lock['input_records'].items():
        if sha(name) != digest:
            raise ValueError('changed frozen input record '+name)
    return lock


def source_record(lock, role, seed):
    return next(x for x in lock['source_models'] if (x['role'], x['seed']) == (role, seed))


def capture_inputs():
    prior = read(PAIR_LOCK)
    if sha(PAIR_LOCK) != PAIR_LOCK.with_name('lock.sha256').read_text().strip():
        raise ValueError('old pair lock identity')
    selection = read(RAW/'selection.json')
    scenes = {x['scene']: x for x in selection['scenes']}
    downloads = {x['path']: x for x in read(RAW/'download_manifest.json')}
    if tuple(x['scene'] for x in prior['entries']) != SCENES:
        raise ValueError('old scene scope/order')
    records, entries = {}, []
    for e in prior['entries']:
        old = scenes[e['scene']]
        frame_map = {x['index']: x for x in old['frames']}
        frames = []
        for index in sorted({i for pair in e['pairs'] for i in pair}):
            f = frame_map[index]
            path = RAW/e['scene']/'depth'/f['depth']
            expected = downloads[str(path.relative_to(RAW))]['sha256']
            if sha(path) != expected:
                raise ValueError('native depth identity '+str(path))
            records[str(path)] = expected
            frames.append(dict(index=index, path=str(path), sha256=expected,
                               association_gap_us=f['time_gap_us']))
        kpath, ppath = RAW/e['scene']/'intrinsics.txt', RAW/old['pose_file']
        for path, expected in ((kpath, old['K_sha256']), (ppath, old['pose_sha256'])):
            if sha(path) != expected:
                raise ValueError('native camera identity')
            records[str(path)] = expected
        if not all(pair in old['pairs'] for pair in e['pairs']):
            raise ValueError('pair selection identity')
        if any(frame_map[q]['depth'] == frame_map[k]['depth'] for q, k in e['pairs']):
            raise ValueError('same native measurement in a pair')
        entries.append(dict(scene=e['scene'], split={'train':'fit','val':'val','test':'old_eval'}[e['split']],
            pairs=e['pairs'], frames=frames, intrinsics=str(kpath), poses=str(ppath)))
    return dict(entries=entries, raw_hashes=records)
