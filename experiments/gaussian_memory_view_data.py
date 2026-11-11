"""Leakage-separated frozen VGGT caches for actual view-token experiments."""
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import torch

from experiments.gaussian_memory_setup import DATA, ROOT, COMMIT, sha256
from experiments.gaussian_memory_data import CACHE, REPORT, REF, TRAIN_QUERY, load_images
from experiments.gaussian_memory_attention_train import write_json, load_inputs
from models.gaussian_memory_view_token import apply_rgb_mask

OUT = ROOT / 'results/gaussian_memory/view_token1'
TOKEN_CACHE = DATA / 'view_token_cache_v1'
CONF_CACHE = DATA / 'view_token_confirmation_v1'
SEEDS = (20261110, 20261111, 20261112)
VARIANTS = ('local', 'center', 'gaussian')


def rectangle(episode, query, sample):
    digest = hashlib.sha256(f'view-token-v1:{episode}:{query}:{sample}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    y, x = int(rng.integers(9)), int(rng.integers(11))
    mask = torch.zeros(15, 20, dtype=torch.bool)
    mask[y:y + 7, x:x + 10] = True
    return mask.flatten()


def load_backbone():
    from safetensors.torch import load_file
    sys.path.insert(0, str(ROOT / 'third_party/vggt'))
    from vggt.models.vggt import VGGT
    assert (ROOT / 'third_party/vggt/PINNED_COMMIT').read_text().strip() == COMMIT
    model = VGGT(enable_point=False, enable_track=False)
    state = load_file(str(DATA / 'model.safetensors'))
    mismatch = model.load_state_dict(state, strict=False)
    assert not mismatch.missing_keys
    assert all(k.startswith(('point_head.', 'track_head.')) for k in mismatch.unexpected_keys)
    del state
    return model.eval().requires_grad_(False).cuda()


def make_confirmation_cohort():
    selected = {}
    for folder, selection in [('round6_sun3d', 'selection.json'), ('round6_sun3d', 'selection_v2.json'),
                               ('round7_sun3d', 'selection.json')]:
        root = ROOT / 'data' / folder
        for row in json.loads((root / selection).read_text())['scenes']:
            selected[row['scene']] = (root, row)
    records = []
    old = json.loads((ROOT / 'results/gaussian_memory/attention1/confirmation/cohort.json').read_text())
    old_paths = {f['rgb_path'] for r in old['records'] for f in r['frames']}
    for scene, (root, row) in sorted(selected.items()):
        available = sorted([f for f in row['frames'] if (root / scene / 'image' / f['rgb']).is_file()], key=lambda f: f['index'])
        frames = available[40:53]
        assert len(frames) == 13
        rec = {'id': f'transfer_{len(records):02d}', 'scene': scene, 'split': 'transfer',
               'root': str(root), 'intrinsics': str(root / scene / 'intrinsics.txt'),
               'poses': str(root / row['pose_file']), 'reference': REF, 'training_query': [],
               'heldout_query': [1, 2, 3, 5, 6, 7, 9, 10, 11],
               'frames': [{'index': f['index'], 'rgb_path': str(root / scene / 'image' / f['rgb']),
                           'rgb_sha256': sha256(root / scene / 'image' / f['rgb'])} for f in frames]}
        assert not old_paths & {f['rgb_path'] for f in rec['frames']}
        for k in ('intrinsics', 'poses'):
            rec[k + '_sha256'] = sha256(rec[k])
        records.append(rec)
    assert len(records) == 12
    return records


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'plan.json').exists()
    memories = {str(i['path']): sha256(i['path']) for i in load_inputs()}
    sources = ['models/gaussian_memory_view_token.py', 'experiments/gaussian_memory_view_data.py',
               'experiments/gaussian_memory_view_train.py', 'experiments/gaussian_memory_view_evaluate.py',
               'models/gaussian_memory_attention.py', 'models/gaussian_memory_projection.py']
    plan = {'hypothesis': 'Reference-only Gaussian memory supplements masked actual VGGT view tokens',
            'plan_sha256': sha256(ROOT / 'docs/gaussian_memory_view_token_plan.md'),
            'source_sha256': {p: sha256(ROOT / p) for p in sources},
            'old_cache_identity_sha256': sha256(REPORT / 'cache_identity.json'),
            'memory_sha256': memories, 'backbone_identity': json.loads((DATA / 'source_identity.json').read_text()),
            'center_variance': .7146816253662109, 'seeds': SEEDS, 'variants': VARIANTS,
            'steps': 2000, 'loss_hidden_visible': [.8, .2], 'masked_patches': 70,
            'transfer_cohort': make_confirmation_cohort(),
            'transfer_history': 'New later clips for this adapter, historically used scenes; not untouched external test scenes'}
    write_json(OUT / 'plan.json', plan)
    print(json.dumps({'prepared': True, 'old_memories': len(memories), 'prospective_transfer_clips': 12}), flush=True)


def checked():
    plan = json.loads((OUT / 'plan.json').read_text())
    for p, value in plan['source_sha256'].items():
        assert sha256(ROOT / p) == value, p
    for p, value in plan['memory_sha256'].items():
        assert sha256(p) == value, p
    assert sha256(ROOT / 'docs/gaussian_memory_view_token_plan.md') == plan['plan_sha256']
    return plan


@torch.inference_mode()
def cache_tokens(model, raw, destination):
    row = raw['episode']
    images = raw['rgb'].permute(0, 3, 1, 2).float().cuda() / 255.
    teacher = []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for batch in images.split(2):
            tokens, start = model.aggregator(batch[:, None])  # B independent, S=1.
            teacher.append(tokens[-1][:, 0, start:].half().cpu())
    teacher = torch.cat(teacher)
    ids, masks, roles = [], [], []
    for q in sorted(set(row['training_query'] + row['heldout_query'])):
        train = q in row['training_query']
        for s in range(4 if train else 2):
            ids.append(q); masks.append(rectangle(row['id'], q, s if train else s + 100))
            roles.append('train' if train else 'evaluation')
    masks = torch.stack(masks)
    masked_tokens = []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for begin in range(0, len(ids), 2):
            x = apply_rgb_mask(images[ids[begin:begin + 2]], masks[begin:begin + 2].cuda())
            tokens, start = model.aggregator(x[:, None])
            masked_tokens.append(tokens[-1][:, 0, start:].half().cpu())
    result = {'episode': row, 'query': torch.tensor(ids), 'mask': masks, 'roles': roles,
              'input_tokens': torch.cat(masked_tokens), 'teacher_tokens': teacher,
              'mask_before_backbone': True, 'independent_single_view_axis': True}
    assert torch.isfinite(result['input_tokens']).all() and torch.isfinite(teacher).all()
    torch.save(result, destination)
    return result


def extract():
    checked(); torch.set_num_threads(4)
    TOKEN_CACHE.mkdir(parents=True, exist_ok=True)
    model = load_backbone()
    started, hashes = time.monotonic(), {}
    for item in load_inputs():
        rid = item['row']['id']; dest = TOKEN_CACHE / (rid + '.pt')
        assert not dest.exists(), 'preserve existing extraction; diagnose explicitly before resuming'
        record = cache_tokens(model, item['raw'], dest)
        hashes[str(dest)] = sha256(dest)
        print(json.dumps({'cached': rid, 'samples': len(record['query']), 'seconds': time.monotonic() - started,
                          'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
    write_json(OUT / 'cache_identity.json', {'sha256': hashes, 'seconds': time.monotonic() - started,
               'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3, 'plan_sha256': sha256(OUT / 'plan.json')})


def transfer_extract():
    from PIL import Image
    from experiments.round5_rgbd_data import read_sun3d_intrinsics, read_sun3d_extrinsics, resize_intrinsics, _listing
    from experiments.gaussian_memory_data import source_geometry, merge_seeds
    from models.gaussian_memory import GaussianWriter, seeds_to_device, detached_memory
    plan = checked()
    activation = json.loads((OUT / 'transfer_activation.json').read_text())
    assert activation['activate']
    for p, h in activation['models'].items():
        assert sha256(p) == h
    CONF_CACHE.mkdir(parents=True, exist_ok=True)
    (OUT / 'transfer_memories').mkdir(exist_ok=True)
    model = load_backbone()
    writer = GaussianWriter().cuda().eval().requires_grad_(False)
    writer.load_state_dict(torch.load(REPORT / 'learned_geometry/step_2000.pt', weights_only=False)['model'])
    pca = torch.load(CACHE / 'pca.pt', weights_only=False, map_location='cuda')
    hashes, rows = {}, []
    for source in plan['transfer_cohort']:
        row = dict(source)
        for key in ('intrinsics', 'poses'):
            assert sha256(row[key]) == row[key + '_sha256']
        listing = _listing(ROOT / row['root'] / row['scene'] / 'image_index.html', 'jpg')
        for frame in row['frames']:
            assert listing[frame['index']] == frame['rgb_path'].rsplit('/', 1)[-1]
        poses = read_sun3d_extrinsics(row['poses'])[[f['index'] for f in row['frames']]]
        poses = np.linalg.inv(poses[0])[None] @ poses
        k = read_sun3d_intrinsics(row['intrinsics']); Ks = []
        for f in row['frames']:
            with Image.open(f['rgb_path']) as im:
                ki = resize_intrinsics(k, (im.height, im.width), (210, 280))
            ki[:2, 2] += .5; Ks.append(ki)
        row.update(K=np.stack(Ks).tolist(), c2w_metric=poses.tolist())
        images = load_images(row, 'cuda')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            refs = images[REF][None]
            tokens, start = model.aggregator(refs)
            context = tokens[-1][0, :, start:].float().reshape(-1, 2048)
            normalized = (images[REF] - model.aggregator._resnet_mean[0]) / model.aggregator._resnet_std[0]
            teacher = model.aggregator.patch_embed(normalized)['x_norm_patchtokens'].float()
            with torch.autocast('cuda', enabled=False):
                pose = model.camera_head(tokens)[-1]
                depth, conf = model.depth_head(tokens, images=refs, patch_start_idx=start)
                K = torch.tensor(row['K'], device='cuda').float()
                c2w = torch.tensor(row['c2w_metric'], device='cuda').float()
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
        raw_path = CONF_CACHE / (row['id'] + '_raw.pt')
        mem_path = OUT / 'transfer_memories' / (row['id'] + '.pt')
        token_path = CONF_CACHE / (row['id'] + '.pt')
        assert not raw_path.exists() and not mem_path.exists() and not token_path.exists()
        torch.save(raw, raw_path); torch.save({'memory': memory, 'episode': row['id']}, mem_path)
        del tokens, context, refs, teacher
        cache_tokens(model, raw, token_path)
        for p in (raw_path, mem_path, token_path): hashes[str(p)] = sha256(p)
        rows.append(row)
        print(json.dumps({'transfer_cached': row['id'], 'scene': row['scene'], 'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
    write_json(OUT / 'transfer_cache_identity.json', {'sha256': hashes, 'episodes': rows,
               'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare', 'extract', 'transfer'])
    a = p.parse_args(); {'prepare': prepare, 'extract': extract, 'transfer': transfer_extract}[a.command]()
