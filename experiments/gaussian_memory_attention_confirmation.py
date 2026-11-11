"""Fixed reader confirmation on twelve existing but method-unseen SUN3D scenes."""
import argparse
import json
import sys

import numpy as np
import torch
from PIL import Image

from experiments.gaussian_memory_attention_train import OUT, SEEDS, VARIANTS, write_json, assert_identity
from experiments.gaussian_memory_data import (CACHE, REPORT, REF, HEIGHT, WIDTH, PATCH, load_images,
                                               source_geometry, merge_seeds)
from experiments.gaussian_memory_setup import DATA, ROOT, COMMIT, sha256
from experiments.round5_rgbd_data import read_sun3d_intrinsics, read_sun3d_extrinsics, resize_intrinsics, _listing
from models.gaussian_memory import GaussianWriter, seeds_to_device, detached_memory, read_memory
from models.gaussian_memory_attention import GaussianAttentionReader, geometry_inputs
from models.gaussian_memory_projection import projected_inputs
from experiments.gaussian_memory_attention_evaluate import aggregate

DEST = OUT / 'confirmation'
NEW_CACHE = DATA / 'attention_confirmation_cache'


def cohort():
    DEST.mkdir(exist_ok=True)
    assert not (DEST / 'cohort.json').exists()
    selected = {}
    for folder, selection in [('round6_sun3d', 'selection.json'), ('round6_sun3d', 'selection_v2.json'),
                               ('round7_sun3d', 'selection.json')]:
        root = ROOT / 'data' / folder
        source = root / selection
        for row in json.loads(source.read_text())['scenes']:
            selected[row['scene']] = (root, source, row)
    records = []
    for scene, (root, source, s) in sorted(selected.items()):
        frames = sorted([f for f in s['frames'] if (root / scene / 'image' / f['rgb']).is_file()],
                        key=lambda f: f['index'])[:13]
        assert len(frames) == 13
        records.append({'id': f'confirm_{len(records):02d}', 'root': str(root), 'scene': scene,
                        'selection': str(source), 'selection_sha256': sha256(source),
                        'intrinsics': str(root / scene / 'intrinsics.txt'), 'intrinsics_sha256': s['K_sha256'],
                        'poses': str(root / s['pose_file']), 'poses_sha256': s['pose_sha256'],
                        'frames': [{'index': f['index'], 'rgb_path': str(root / scene / 'image' / f['rgb']),
                                    'rgb_sha256': sha256(root / scene / 'image' / f['rgb'])} for f in frames],
                        'reference': REF, 'training_query': [], 'heldout_query': [1, 2, 3, 5, 6, 7, 9, 10, 11]})
    assert len(records) == 12
    old = {r['scene'] for r in json.loads((REPORT / 'data_manifest.json').read_text())['episodes']}
    assert not old & set(selected)
    plan = {'selection': 'All 12 locally available R6/R7 scenes; first 13 downloaded original-ordinal frames; no outcome selection',
            'conditional_on': 'Development Gaussian better than MLP and Center in both heldout roles, all paired seeds, majority of six scenes',
            'history': 'Previously used in older research; none used by the current writer/reader experiment; not historically untouched external data',
            'pixel_decode_or_prediction_for_confirmation_started': False, 'new_training': False, 'new_pca_fit': False,
            'query_rgb_used_as_teacher_only': True, 'records': records,
            'checkpoints': {str(p): sha256(p) for p in OUT.glob('*/step_2000.pt')},
            'writer_sha256': sha256(REPORT / 'learned_geometry/step_2000.pt'),
            'pca_sha256': sha256(CACHE / 'pca.pt')}
    assert len(plan['checkpoints']) == 9
    write_json(DEST / 'cohort.json', plan)
    print(json.dumps({'scenes': len(records), 'queries': 108, 'fixed_models': 9}), flush=True)


def checked():
    assert_identity()
    plan = json.loads((DEST / 'cohort.json').read_text())
    for path, expected in plan['checkpoints'].items():
        assert sha256(path) == expected
    assert sha256(REPORT / 'learned_geometry/step_2000.pt') == plan['writer_sha256']
    assert sha256(CACHE / 'pca.pt') == plan['pca_sha256']
    adaptation = json.loads((DEST / 'adaptation_lock.json').read_text())
    assert sha256(DEST / 'cohort.json') == adaptation['cohort_sha256']
    for item in adaptation['models'].values():
        assert sha256(item['path']) == item['sha256']
    for path, expected in adaptation['source_sha256'].items():
        assert sha256(path) == expected
    return plan


def extract():
    from safetensors.torch import load_file
    sys.path.insert(0, str(ROOT / 'third_party/vggt'))
    from vggt.models.vggt import VGGT
    torch.set_num_threads(4)
    torch.manual_seed(20261110)
    plan = checked()
    assert json.loads((DEST / 'adaptation_activation.json').read_text())['development_signal']
    NEW_CACHE.mkdir(parents=True, exist_ok=True)
    assert (ROOT / 'third_party/vggt/PINNED_COMMIT').read_text().strip() == COMMIT
    model = VGGT(enable_point=False, enable_track=False)
    state = load_file(str(DATA / 'model.safetensors'))
    mismatch = model.load_state_dict(state, strict=False)
    assert not mismatch.missing_keys
    assert all(k.startswith(('point_head.', 'track_head.')) for k in mismatch.unexpected_keys)
    del state
    model.eval().requires_grad_(False).cuda()
    writer = GaussianWriter().cuda().eval().requires_grad_(False)
    writer.load_state_dict(torch.load(REPORT / 'learned_geometry/step_2000.pt', weights_only=False)['model'])
    pca = torch.load(CACHE / 'pca.pt', weights_only=False, map_location='cuda')
    (DEST / 'memories').mkdir(exist_ok=True)
    manifest, hashes, diagnostics = [], {}, []
    for row in plan['records']:
        dest = NEW_CACHE / (row['id'] + '.pt')
        assert not dest.exists(), 'Existing confirmation extraction must be preserved'
        for key in ('intrinsics', 'poses'):
            assert sha256(row[key]) == row[key + '_sha256']
        rgb_names = _listing(ROOT / row['root'] / row['scene'] / 'image_index.html', 'jpg')
        for frame in row['frames']:
            assert rgb_names[frame['index']] == frame['rgb_path'].rsplit('/', 1)[-1]
        poses = read_sun3d_extrinsics(row['poses'])[[f['index'] for f in row['frames']]]
        poses = np.linalg.inv(poses[0])[None] @ poses
        intrinsic = read_sun3d_intrinsics(row['intrinsics'])
        Ks = []
        for frame in row['frames']:
            with Image.open(frame['rgb_path']) as im:
                native_hw = (im.height, im.width)
            k = resize_intrinsics(intrinsic, native_hw, (HEIGHT, WIDTH))
            k[:2, 2] += .5  # corner-origin pixel centers, exactly as the original cache
            Ks.append(k)
        row = dict(row, K=np.stack(Ks).tolist(), c2w_metric=poses.tolist(), split='confirmation')
        images = load_images(row, 'cuda')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            teacher = []
            for chunk in images.split(4):
                normalized = (chunk - model.aggregator._resnet_mean[0]) / model.aggregator._resnet_std[0]
                teacher.append(model.aggregator.patch_embed(normalized)['x_norm_patchtokens'])
            teacher = torch.cat(teacher).float()
            ref_images = images[REF][None]
            tokens, start = model.aggregator(ref_images)
            context = tokens[-1][0, :, start:].float().reshape(-1, 2048)
            with torch.autocast('cuda', enabled=False):
                pose = model.camera_head(tokens)[-1]
                depth, conf = model.depth_head(tokens, images=ref_images, patch_start_idx=start)
                K = torch.tensor(row['K'], device='cuda').float()
                c2w = torch.tensor(row['c2w_metric'], device='cuda').float()
                means, cov, valid, c2w, diag = source_geometry(depth, conf, pose, K, c2w)
                seed_mean, seed_chol, near_ids, weights = merge_seeds(means, cov, valid)
                seed_context = (context[near_ids] * weights[..., None]).sum(1)
                seed_teacher = (teacher[REF].reshape(-1, 1024)[near_ids] * weights[..., None]).sum(1)
                # Match original half-precision teacher cache, followed by the fixed training PCA.
                teacher = ((teacher.half().float() - pca['mean']) @ pca['basis']) / pca['scale']
                seed_teacher = ((seed_teacher - pca['mean']) @ pca['basis']) / pca['scale']
                raw = {'seed_mean': seed_mean.cpu(), 'seed_chol': seed_chol.cpu(),
                       'seed_context': seed_context.half().cpu(), 'seed_teacher': seed_teacher.cpu(),
                       'teacher': teacher.reshape(13, 15, 20, 32).cpu(),
                       'K': (K / K.new_tensor([PATCH, PATCH, 1])[:, None]).cpu(), 'c2w': c2w.cpu(),
                       'diagnostics': diag, 'episode': row, 'rgb': (images.permute(0, 2, 3, 1) * 255).round().byte().cpu()}
                memory = detached_memory(writer(seeds_to_device(raw, 'cuda')))
            torch.save(raw, dest)
            memory_path = DEST / 'memories' / (row['id'] + '.pt')
            torch.save({'memory': memory, 'episode': row['id'], 'scene': row['scene'],
                        'checkpoint_sha256': plan['writer_sha256'], 'coordinate_unit_meters': diag['unit_meters']}, memory_path)
        hashes[str(dest)] = sha256(dest)
        hashes[str(memory_path)] = sha256(memory_path)
        manifest.append(row); diagnostics.append(dict(episode=row['id'], **diag))
        print(json.dumps({'extracted': row['id'], 'scene': row['scene'], 'scale_ratio': diag['reference_scale_ratio_max_over_min']}), flush=True)
        del images, tokens, context, teacher, ref_images, raw
    write_json(DEST / 'data_manifest.json', {'episodes': manifest, 'diagnostics': diagnostics,
               'cache_sha256': hashes, 'cohort_sha256': sha256(DEST / 'cohort.json'),
               'extractor_sha256': sha256(__file__), 'original_extractor_sha256': sha256(ROOT / 'experiments/gaussian_memory_data.py'),
               'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3})
    checked()


def evaluate():
    torch.set_num_threads(4)
    checked()
    manifest = json.loads((DEST / 'data_manifest.json').read_text())
    for path, expected in manifest['cache_sha256'].items():
        assert sha256(path) == expected
    assert not (DEST / 'scores.json').exists()
    models = {}
    locked = json.loads((DEST / 'adaptation_lock.json').read_text())['models']
    for seed in SEEDS:
        for variant in VARIANTS:
            ckpt = torch.load(locked[f'{variant}_{seed}']['path'], weights_only=False, map_location='cuda')
            model = GaussianAttentionReader(variant, ckpt['config']['center_variance']).cuda().eval()
            model.load_state_dict(ckpt['model']); models[(variant, seed)] = model
    scores, figures = [], {}
    with torch.no_grad():
        for row in manifest['episodes']:
            rid, qids = row['id'], row['heldout_query']
            raw = torch.load(NEW_CACHE / (rid + '.pt'), weights_only=False, map_location='cpu')
            memory = torch.load(DEST / 'memories' / (rid + '.pt'), weights_only=False, map_location='cuda')['memory']
            c2w, K = raw['c2w'][qids].cuda(), raw['K'][qids].cuda()
            target = raw['teacher'][qids].cuda().flatten(1, 2)
            inputs = projected_inputs(memory, c2w, K)
            rendered, alpha = read_memory(memory, c2w, K)
            figures[rid] = {'qids': qids, 'scene': row['scene'], 'rgb': raw['rgb'][qids],
                            'target': target.cpu().reshape(-1, 15, 20, 32), 'renderer': rendered.cpu()}
            predictions = [('renderer', 0, 'normal', rendered.flatten(1, 2), 1 - alpha.mean((1, 2, 3)))]
            for (variant, seed), model in models.items():
                for condition in (['normal', 'isotropic'] if variant == 'gaussian' else ['normal']):
                    prediction, attn = model(inputs, condition, return_attention=True)
                    predictions.append((variant, seed, condition, prediction, attn[..., -1].mean((1, 2))))
                    if seed == SEEDS[0]:
                        figures[rid][variant + '/' + condition] = prediction.cpu().reshape(-1, 15, 20, 32)
            for variant, seed, condition, prediction, background in predictions:
                mse = (prediction - target).square().mean((1, 2))
                cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=-1).mean(1)
                ratio = prediction.norm(dim=-1).mean(1) / target.norm(dim=-1).mean(1).clamp_min(1e-8)
                for i, q in enumerate(qids):
                    scores.append({'episode': rid, 'scene': row['scene'], 'frame': row['frames'][q]['index'],
                                   'query_ordinal': q, 'role': 'confirmation_scene', 'variant': variant,
                                   'seed': seed, 'condition': condition, 'mse': mse[i].item(), 'cosine': cosine[i].item(),
                                   'background_mass': background[i].item(), 'norm_ratio': ratio[i].item()})
            print(json.dumps({'confirmation_evaluated': rid}), flush=True)
    fields = ['role', 'variant', 'condition', 'seed']
    write_json(DEST / 'scores.json', {'scores': scores, 'summary': aggregate(scores, fields),
               'per_scene': aggregate(scores, ['scene', *fields]), 'cohort_sha256': sha256(DEST / 'cohort.json'),
               'data_manifest_sha256': sha256(DEST / 'data_manifest.json')})
    torch.save(figures, DEST / 'visual_data.pt')
    checked()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['cohort', 'extract', 'evaluate'])
    args = p.parse_args()
    {'cohort': cohort, 'extract': extract, 'evaluate': evaluate}[args.command]()
