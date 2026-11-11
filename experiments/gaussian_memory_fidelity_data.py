"""Fixed data/model identities and later-clips extraction for fidelity controls."""
import argparse
import json
import time

import numpy as np
import torch
from PIL import Image

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_data import CACHE, REPORT, REF, load_images, source_geometry, merge_seeds
from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_view_data import OUT as OLD_OUT, TOKEN_CACHE, load_backbone, cache_tokens, checked as old_checked
from models.gaussian_memory import GaussianWriter, seeds_to_device, detached_memory

OUT = ROOT / 'results/gaussian_memory/fidelity1'
NEW_CACHE = DATA / 'fidelity_confirmation_v1'
SEEDS = (20261110, 20261111, 20261112)
VARIANTS = ('local', 'center')
REGIMES = ('A', 'B', 'C')


def fixed_cohort():
    selected = {}
    for folder, selection in [('round6_sun3d', 'selection.json'), ('round6_sun3d', 'selection_v2.json'),
                               ('round7_sun3d', 'selection.json')]:
        root = ROOT / 'data' / folder
        for row in json.loads((root / selection).read_text())['scenes']:
            selected[row['scene']] = (root, row)
    previous = json.loads((OLD_OUT / 'plan.json').read_text())['transfer_cohort']
    previous += json.loads((ROOT / 'results/gaussian_memory/attention1/confirmation/cohort.json').read_text())['records']
    used = {f['rgb_path'] for row in previous for f in row['frames']}
    rows = []
    for scene, (root, source) in sorted(selected.items()):
        available = sorted([f for f in source['frames'] if (root / scene / 'image' / f['rgb']).is_file()], key=lambda f: f['index'])
        frames = available[80:93]; assert len(frames) == 13
        row = {'id': f'fidelity_{len(rows):02d}', 'scene': scene, 'split': 'transfer', 'root': str(root),
               'intrinsics': str(root / scene / 'intrinsics.txt'), 'poses': str(root / source['pose_file']),
               'reference': REF, 'training_query': [], 'heldout_query': [1,2,3,5,6,7,9,10,11],
               'frames': [{'index': f['index'], 'rgb_path': str(root / scene / 'image' / f['rgb']),
                           'rgb_sha256': sha256(root / scene / 'image' / f['rgb'])} for f in frames]}
        assert not used & {f['rgb_path'] for f in row['frames']}
        for k in ('intrinsics', 'poses'): row[k + '_sha256'] = sha256(row[k])
        rows.append(row)
    assert len(rows) == 12
    return rows


def model_path(regime, variant, seed, initial=False):
    base = OLD_OUT / f'{variant}_{seed}' if regime == 'A' else OUT / f'{regime}_{variant}_{seed}'
    return base / ('initial.pt' if initial else 'step_2000.pt')


def prepare():
    old_checked(); OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'plan.json').exists()
    sources = ['models/gaussian_memory_view_token.py', 'models/gaussian_memory_attention.py',
               'models/gaussian_memory_projection.py', 'models/gaussian_memory.py',
               'experiments/gaussian_memory_data.py', 'experiments/gaussian_memory_view_data.py',
               'experiments/gaussian_memory_view_train.py', 'experiments/gaussian_memory_fidelity_data.py',
               'experiments/gaussian_memory_fidelity_train.py', 'experiments/gaussian_memory_fidelity_evaluate.py']
    old_paths = [model_path('A', v, s, initial) for s in SEEDS for v in VARIANTS for initial in (False, True)]
    identities = [OLD_OUT / 'plan.json', OLD_OUT / 'cache_identity.json', REPORT / 'cache_identity.json',
                  REPORT / 'learned_geometry/step_2000.pt', CACHE / 'pca.pt']
    plan = {'source_sha256': {p: sha256(ROOT / p) for p in sources},
            'plan_sha256': sha256(ROOT / 'docs/gaussian_memory_fidelity_plan.md'),
            'old_model_sha256': {str(p): sha256(p) for p in old_paths},
            'old_identity_sha256': {str(p): sha256(p) for p in identities},
            'seeds': SEEDS, 'variants': VARIANTS, 'steps': 2000, 'batch_masked': 2,
            'losses': {'A': '.8 hidden + .2 visible', 'B': 'mean over all masked-input patches',
                       'C': 'B + mean clean-input identity error; fixed coefficient 1'},
            'C_extra_forwards_per_step': 1, 'new_training_runs': 12, 'reused_models': 6,
            'transfer_cohort': fixed_cohort(), 'new_scene_claim': False, 'all_groups_transfer_without_selection': True}
    write_json(OUT / 'plan.json', plan)
    print(json.dumps({'prepared': True, 'new_runs': 12, 'fixed_transfer_clips': 12}), flush=True)


def checked():
    plan = json.loads((OUT / 'plan.json').read_text())
    for p, h in plan['source_sha256'].items(): assert sha256(ROOT / p) == h, p
    for field in ('old_model_sha256', 'old_identity_sha256'):
        for p, h in plan[field].items(): assert sha256(p) == h, p
    assert sha256(ROOT / 'docs/gaussian_memory_fidelity_plan.md') == plan['plan_sha256']
    old_checked()
    return plan


def lock_models():
    checked()
    paths = [model_path(r, v, s) for r in REGIMES for s in SEEDS for v in VARIANTS]
    for r in ('B', 'C'):
        for s in SEEDS:
            for v in VARIANTS:
                assert json.loads((model_path(r, v, s).parent / 'status.json').read_text())['complete']
    assert (OUT / 'development_evaluation/scores.json').exists()
    assert not (OUT / 'transfer_lock.json').exists()
    write_json(OUT / 'transfer_lock.json', {'models': {str(p): sha256(p) for p in paths},
               'model_selection': False, 'cohort_sha256': sha256(OUT / 'plan.json'),
               'scope': '12 historically used scenes; later non-overlapping clips'})
    print(json.dumps({'locked_models': len(paths), 'cohort_clips': 12}), flush=True)


def assert_lock():
    plan = checked()
    lock = json.loads((OUT / 'transfer_lock.json').read_text())
    assert sha256(OUT / 'plan.json') == lock['cohort_sha256']
    for p, h in lock['models'].items(): assert sha256(p) == h, p
    return plan


def extract():
    from experiments.round5_rgbd_data import read_sun3d_intrinsics, read_sun3d_extrinsics, resize_intrinsics, _listing
    plan = assert_lock(); torch.set_num_threads(4)
    NEW_CACHE.mkdir(parents=True, exist_ok=True); (OUT / 'memories').mkdir(exist_ok=True)
    model = load_backbone()
    writer = GaussianWriter().cuda().eval().requires_grad_(False)
    writer.load_state_dict(torch.load(REPORT / 'learned_geometry/step_2000.pt', weights_only=False, map_location='cuda')['model'])
    pca = torch.load(CACHE / 'pca.pt', weights_only=False, map_location='cuda')
    hashes, rows, started = {}, [], time.monotonic()
    for source in plan['transfer_cohort']:
        row = dict(source)
        for key in ('intrinsics', 'poses'): assert sha256(row[key]) == row[key + '_sha256']
        listing = _listing(ROOT / row['root'] / row['scene'] / 'image_index.html', 'jpg')
        for f in row['frames']: assert listing[f['index']] == f['rgb_path'].rsplit('/', 1)[-1]
        poses = read_sun3d_extrinsics(row['poses'])[[f['index'] for f in row['frames']]]
        poses = np.linalg.inv(poses[0])[None] @ poses
        intrinsic = read_sun3d_intrinsics(row['intrinsics']); Ks = []
        for f in row['frames']:
            with Image.open(f['rgb_path']) as image:
                K = resize_intrinsics(intrinsic, (image.height, image.width), (210, 280))
            K[:2, 2] += .5; Ks.append(K)
        row.update(K=np.stack(Ks).tolist(), c2w_metric=poses.tolist())
        images = load_images(row, 'cuda')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            refs = images[REF][None]; tokens, start = model.aggregator(refs)
            context = tokens[-1][0, :, start:].float().reshape(-1, 2048)
            normalized = (images[REF] - model.aggregator._resnet_mean[0]) / model.aggregator._resnet_std[0]
            teacher = model.aggregator.patch_embed(normalized)['x_norm_patchtokens'].float()
            with torch.autocast('cuda', enabled=False):
                pose = model.camera_head(tokens)[-1]
                depth, conf = model.depth_head(tokens, images=refs, patch_start_idx=start)
                K, c2w = torch.tensor(row['K'], device='cuda').float(), torch.tensor(row['c2w_metric'], device='cuda').float()
                means, cov, valid, c2w, diag = source_geometry(depth, conf, pose, K, c2w)
                mean, chol, near, weights = merge_seeds(means, cov, valid)
                seed_teacher = (teacher.reshape(-1, 1024)[near] * weights[..., None]).sum(1)
                raw = {'seed_mean': mean.cpu(), 'seed_chol': chol.cpu(),
                       'seed_context': (context[near] * weights[..., None]).sum(1).half().cpu(),
                       'seed_teacher': ((seed_teacher - pca['mean']) @ pca['basis'] / pca['scale']).cpu(),
                       'K': (K / K.new_tensor([14, 14, 1])[:, None]).cpu(), 'c2w': c2w.cpu(),
                       'diagnostics': diag, 'episode': row,
                       'rgb': (images.permute(0, 2, 3, 1) * 255).round().byte().cpu()}
                memory = detached_memory(writer(seeds_to_device(raw, 'cuda')))
        rp, mp, tp = NEW_CACHE / (row['id'] + '_raw.pt'), OUT / 'memories' / (row['id'] + '.pt'), NEW_CACHE / (row['id'] + '.pt')
        assert not any(p.exists() for p in (rp, mp, tp))
        torch.save(raw, rp)
        torch.save({'memory': memory, 'episode': row['id'], 'writer_sha256': sha256(REPORT / 'learned_geometry/step_2000.pt')}, mp)
        del tokens, context, refs, teacher
        cache_tokens(model, raw, tp)
        for p in (rp, mp, tp): hashes[str(p)] = sha256(p)
        rows.append(row)
        print(json.dumps({'cached': row['id'], 'seconds': time.monotonic() - started,
                          'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
    write_json(OUT / 'transfer_cache_identity.json', {'sha256': hashes, 'episodes': rows,
               'seconds': time.monotonic() - started, 'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare', 'lock', 'extract'])
    a = p.parse_args(); {'prepare': prepare, 'lock': lock_models, 'extract': extract}[a.command]()
