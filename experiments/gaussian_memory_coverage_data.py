"""Fixed scene coverage with equal clip/query/mask/optimization budgets."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_data import CACHE, REPORT, REF, TRAIN_QUERY, HOLD_QUERY, load_images, source_geometry, merge_seeds
from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_view_data import load_backbone, rectangle
from experiments.gaussian_memory_direct_data import OUT as OLD_OUT, REF_CACHE, SEEDS, VARIANTS, model_path as old_model_path, reference_memory
from models.gaussian_memory import GaussianWriter, seeds_to_device, detached_memory
from models.gaussian_memory_view_token import apply_rgb_mask

OUT = ROOT / 'results/gaussian_memory/coverage1'
NEW_CACHE = DATA / 'coverage_v1'
SALT = 'GRoPE-memory-reader-coverage-20261110-v1:'
PROJECTION = DATA.parent / 'projection_confirmation_20261104/raw/selection.json'


def model_path(regime, variant, seed, initial=False):
    if regime == 'narrow':
        return old_model_path(variant, seed, initial)
    return OUT / f'{variant}_{seed}' / ('initial.pt' if initial else 'step_2000.pt')


def projection_row(source, ordinal):
    from experiments.round5_rgbd_data import read_sun3d_intrinsics, read_sun3d_extrinsics, resize_intrinsics, _listing
    base = PROJECTION.parent
    frames = sorted(source['frames'], key=lambda f: f['index'])[:13]
    assert len(frames) == 13
    poses_path = base / source['pose_file']
    intrinsic_path = base / source['scene'] / 'intrinsics.txt'
    poses = read_sun3d_extrinsics(poses_path)[[f['index'] for f in frames]]
    poses = np.linalg.inv(poses[0])[None] @ poses
    intrinsic = read_sun3d_intrinsics(intrinsic_path)
    listing = _listing(base / source['scene'] / 'image_index.html', 'jpg')
    Ks, selected = [], []
    for frame in frames:
        assert listing[frame['index']] == frame['rgb']
        path = base / source['scene'] / 'image' / frame['rgb']
        with Image.open(path) as im:
            k = resize_intrinsics(intrinsic, (im.height, im.width), (210, 280))
        k[:2, 2] += .5
        Ks.append(k)
        selected.append({'index': frame['index'], 'rgb_path': str(path), 'rgb_sha256': sha256(path)})
    row = {'id': f'coverage_{ordinal:02d}', 'scene': source['scene'], 'split': 'fit', 'start': 0,
           'reference': REF, 'training_query': TRAIN_QUERY, 'heldout_query': HOLD_QUERY,
           'frames': selected, 'K': np.stack(Ks).tolist(), 'c2w_metric': poses.tolist(),
           'intrinsics': str(intrinsic_path), 'poses': str(poses_path),
           'intrinsics_sha256': sha256(intrinsic_path), 'poses_sha256': sha256(poses_path)}
    return {'id': row['id'], 'row': row, 'cohort': 'added_training', 'new_geometry': True,
            'raw': str(NEW_CACHE / (row['id'] + '_raw.pt')),
            'tokens': str(NEW_CACHE / (row['id'] + '.pt')),
            'memory': str(OUT / 'memories' / (row['id'] + '.pt')),
            'direct': str(NEW_CACHE / (row['id'] + '_direct.pt'))}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'plan.json').exists()
    old = json.loads((OLD_OUT / 'plan.json').read_text())
    catalog = copy.deepcopy(old['catalog'])
    for meta in catalog:
        meta.update(direct=str(REF_CACHE / (meta['id'] + '.pt')), new_geometry=False, new_tokens=False)
    old_fit = [r for r in catalog if r['cohort'] == 'development' and r['row']['split'] == 'fit']
    pool = [{'scene': r['row']['scene'], 'kind': 'old_development', 'id': r['id']} for r in catalog
            if r['cohort'] == 'development' and r['row']['split'] != 'fit']
    projection = json.loads(PROJECTION.read_text())['scenes']
    pool += [{'scene': r['scene'], 'kind': 'projection', 'source': r} for r in projection]
    for r in pool:
        r['rank_sha256'] = hashlib.sha256((SALT + r['scene']).encode()).hexdigest()
    pool.sort(key=lambda r: r['rank_sha256'])
    assert len(pool) == 11
    primary_places = {r['row']['scene'].split('/')[0] for r in catalog if r['cohort'].startswith('used_clip')}
    assert len(primary_places) == 12
    assert not primary_places & {r['scene'].split('/')[0] for r in pool}
    slots = [i for i in range(12) if i not in (0, 4, 8)]
    training = {i: old_fit[i] for i in (0, 4, 8)}
    for rank, (candidate, slot) in enumerate(zip(pool[:9], slots)):
        if candidate['kind'] == 'old_development':
            meta = next(r for r in catalog if r['id'] == candidate['id'])
            meta['original_tokens'] = meta['tokens']
            meta['tokens'] = str(NEW_CACHE / (meta['id'] + '.pt'))
            meta['row'] = dict(meta['row'], split='fit', training_query=TRAIN_QUERY, heldout_query=HOLD_QUERY)
        else:
            meta = projection_row(candidate['source'], rank)
            catalog.append(meta)
        meta.update(new_tokens=True, mask_template=old_fit[slot]['id'])
        training[slot] = meta
    train_rows = [training[i] for i in range(12)]
    train_ids = [r['id'] for r in train_rows]
    assert len({r['row']['scene'] for r in train_rows}) == 12
    assert not primary_places & {r['row']['scene'].split('/')[0] for r in train_rows}
    for meta in catalog:
        rid = meta['id']
        if meta['cohort'].startswith('used_clip'):
            meta['eval_split'] = meta['cohort']
        elif rid in ('fit_00', 'fit_04', 'fit_08'):
            meta['eval_split'] = 'common_train_heldout'
        elif rid in train_ids:
            meta['eval_split'] = 'added_train_heldout'
        elif meta['row']['split'] == 'fit':
            meta['eval_split'] = 'old_fit_other_clip'
        else:
            meta['eval_split'] = 'remaining_development_scene'
    sources = ['experiments/gaussian_memory_coverage_' + n + '.py' for n in ('data', 'train', 'evaluate')]
    sources += list(old['source_sha256']) + ['models/gaussian_memory.py', 'experiments/gaussian_memory_view_train.py']
    identities = [PROJECTION, OLD_OUT / 'plan.json', OLD_OUT / 'reference_cache_identity.json',
                  OLD_OUT / 'evaluation/scores.json', CACHE / 'pca.pt', REPORT / 'learned_geometry/step_2000.pt']
    identities += [old_model_path(v, s, ini) for v in VARIANTS for s in SEEDS for ini in (False, True)]
    identities += [Path(r[k]) for r in old['catalog'] for k in ('raw', 'tokens', 'memory')]
    identities += [REF_CACHE / (r['id'] + '.pt') for r in old['catalog']]
    identities += [Path(f['rgb_path']) for r in catalog if r.get('new_geometry') for f in r['row']['frames']]
    plan = {'source_sha256': {p: sha256(ROOT / p) for p in sorted(set(sources))},
            'input_sha256': {str(p): sha256(p) for p in identities},
            'document_sha256': sha256(ROOT / 'docs/gaussian_memory_coverage_plan.md'),
            'candidate_ranking': pool, 'selection_salt': SALT, 'catalog': catalog,
            'training_ids': train_ids, 'narrow_training_ids': [r['id'] for r in old_fit],
            'primary_places': sorted(primary_places), 'seeds': SEEDS, 'variants': VARIANTS,
            'new_runs': 12, 'reused_runs': 12, 'steps': 2000, 'batch': 2,
            'center_variance': .7146816253662109, 'training_scenes': {'narrow': 3, 'broad': 12},
            'clips_per_regime': 12, 'unique_training_queries': 72, 'masked_training_examples': 288,
            'frozen_writer_training_scenes': 3, 'confirmation_claim': False, 'no_new_downloads': True}
    write_json(OUT / 'plan.json', plan)
    print(json.dumps({'prepared': True, 'training_ids': train_ids, 'evaluation_episodes': len(catalog),
                      'new_geometry': sum(bool(r.get('new_geometry')) for r in catalog),
                      'primary_places': 12}), flush=True)


def checked(inputs=False):
    plan = json.loads((OUT / 'plan.json').read_text())
    for p, h in plan['source_sha256'].items():
        assert sha256(ROOT / p) == h, p
    assert sha256(ROOT / 'docs/gaussian_memory_coverage_plan.md') == plan['document_sha256']
    if inputs:
        for p, h in plan['input_sha256'].items():
            assert sha256(p) == h, p
    return plan


def load_record(meta, device='cpu'):
    return {'meta': meta, 'row': meta['row'],
            'raw': torch.load(meta['raw'], map_location='cpu', weights_only=False),
            'rec': torch.load(meta['tokens'], map_location='cpu', weights_only=False),
            'memory': torch.load(meta['memory'], map_location=device, weights_only=False)['memory'],
            'direct_memory': {k: v.float() for k, v in torch.load(meta['direct'], map_location=device, weights_only=False)['memory'].items()}}


@torch.inference_mode()
def new_geometry(model, writer, pca, meta):
    row = meta['row']
    images = load_images(row, 'cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        refs = images[REF][None]
        tokens, start = model.aggregator(refs)
        context = tokens[-1][0, :, start:].float().reshape(-1, 2048)
        normalized = (images[REF] - model.aggregator._resnet_mean[0]) / model.aggregator._resnet_std[0]
        teacher = model.aggregator.patch_embed(normalized)['x_norm_patchtokens'].float()
    pose = model.camera_head(tokens)[-1]
    depth, conf = model.depth_head(tokens, images=refs, patch_start_idx=start)
    K = torch.tensor(row['K'], device='cuda').float()
    c2w_metric = torch.tensor(row['c2w_metric'], device='cuda').float()
    means, cov, valid, c2w, diag = source_geometry(depth, conf, pose, K, c2w_metric)
    mean, chol, near, weights = merge_seeds(means, cov, valid)
    seed_teacher = (teacher.reshape(-1, 1024)[near] * weights[..., None]).sum(1)
    raw = {'seed_mean': mean.cpu(), 'seed_chol': chol.cpu(),
           'seed_context': (context[near] * weights[..., None]).sum(1).half().cpu(),
           'seed_teacher': ((seed_teacher - pca['mean']) @ pca['basis'] / pca['scale']).cpu(),
           'K': (K / K.new_tensor([14, 14, 1])[:, None]).cpu(), 'c2w': c2w.cpu(),
           'diagnostics': diag, 'episode': row,
           'rgb': (images.permute(0, 2, 3, 1) * 255).round().byte().cpu()}
    memory = detached_memory(writer(seeds_to_device(raw, 'cuda')))
    direct, direct_diag = reference_memory(model, images[REF], K[REF], c2w_metric[REF], diag['unit_meters'])
    for key in ('raw', 'memory', 'direct'):
        assert not Path(meta[key]).exists(), meta[key]
    torch.save(raw, meta['raw'])
    torch.save({'memory': memory, 'episode': row['id'], 'writer_sha256': sha256(REPORT / 'learned_geometry/step_2000.pt'),
                'query_images_used': False}, meta['memory'])
    torch.save({'memory': direct, 'diagnostics': direct_diag, 'episode': row['id'],
                'reference_frames': REF, 'query_images_used': False, 'teacher_used': False}, meta['direct'])
    return raw


@torch.inference_mode()
def training_tokens(model, raw, meta):
    row = meta['row']
    images = raw['rgb'].permute(0, 3, 1, 2).float().cuda() / 255.
    if 'original_tokens' in meta:
        previous = torch.load(meta['original_tokens'], map_location='cpu', weights_only=False)
        teacher = previous['teacher_tokens']
    else:
        teacher = []
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for batch in images.split(2):
                tokens, start = model.aggregator(batch[:, None])
                teacher.append(tokens[-1][:, 0, start:].half().cpu())
        teacher = torch.cat(teacher)
        previous = None
    queries, masks, roles, values = [], [], [], []
    todo = []
    for q in sorted(TRAIN_QUERY + HOLD_QUERY):
        train = q in TRAIN_QUERY
        for s in range(4 if train else 2):
            mask = rectangle(meta['mask_template'] if train else row['id'], q, s if train else s + 100)
            queries.append(q); masks.append(mask); roles.append('train' if train else 'evaluation')
            if not train and previous is not None:
                ids = ((previous['query'] == q) & (previous['mask'] == mask).all(-1)).nonzero().flatten()
                assert len(ids) >= 1
                assert all(torch.equal(previous['input_tokens'][int(ids[0])], previous['input_tokens'][int(i)]) for i in ids)
                values.append(previous['input_tokens'][int(ids[0])])
            else:
                values.append(None); todo.append(len(values) - 1)
    masks = torch.stack(masks)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for begin in range(0, len(todo), 2):
            ids = todo[begin:begin + 2]
            x = apply_rgb_mask(images[[queries[i] for i in ids]], masks[ids].cuda())
            tokens, start = model.aggregator(x[:, None])
            output = tokens[-1][:, 0, start:].half().cpu()
            for i, v in zip(ids, output):
                values[i] = v
    result = {'episode': row, 'query': torch.tensor(queries), 'mask': masks, 'roles': roles,
              'input_tokens': torch.stack(values), 'teacher_tokens': teacher,
              'mask_template': meta['mask_template'], 'mask_before_backbone': True,
              'independent_single_view_axis': True}
    assert torch.isfinite(result['input_tokens']).all() and torch.isfinite(teacher).all()
    assert not Path(meta['tokens']).exists()
    torch.save(result, meta['tokens'])


def extract():
    plan = checked(True); torch.set_num_threads(4)
    NEW_CACHE.mkdir(parents=True, exist_ok=True); (OUT / 'memories').mkdir(exist_ok=True)
    model = load_backbone()
    writer = GaussianWriter().cuda().eval().requires_grad_(False)
    writer.load_state_dict(torch.load(REPORT / 'learned_geometry/step_2000.pt', map_location='cuda', weights_only=False)['model'])
    pca = torch.load(CACHE / 'pca.pt', map_location='cuda', weights_only=False)
    started = time.monotonic(); hashes = {}; diagnostics = []
    for meta in plan['catalog']:
        if not meta.get('new_tokens'):
            continue
        raw = new_geometry(model, writer, pca, meta) if meta.get('new_geometry') else torch.load(meta['raw'], map_location='cpu', weights_only=False)
        training_tokens(model, raw, meta)
        for key in ('raw', 'tokens', 'memory', 'direct'):
            hashes[meta[key]] = sha256(meta[key])
        diagnostics.append({'id': meta['id'], 'scene': meta['row']['scene'], 'geometry': raw['diagnostics'],
                            'new_geometry': meta.get('new_geometry', False)})
        print(json.dumps({'cached': meta['id'], 'seconds': time.monotonic() - started,
                          'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
    write_json(OUT / 'cache_identity.json', {'sha256': hashes, 'diagnostics': diagnostics,
               'seconds': time.monotonic() - started, 'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3})


def checked_cache():
    plan = checked()
    for p, h in json.loads((OUT / 'cache_identity.json').read_text())['sha256'].items():
        assert sha256(p) == h, p
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'extract'])
    args = parser.parse_args(); {'prepare': prepare, 'extract': extract}[args.command]()
