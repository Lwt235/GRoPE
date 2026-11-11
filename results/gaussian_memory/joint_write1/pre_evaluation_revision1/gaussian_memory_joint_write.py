"""One prospectively fixed compact writer experiment; no new data or search."""
import argparse
import hashlib
import json
import time
import traceback

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments import gaussian_memory_summary_data as previous
from experiments.gaussian_memory_surface_train import loss_function
from experiments.gaussian_memory_surface_evaluate import metrics
from experiments.gaussian_memory_surface_diagnostic import decompose
from models.gaussian_memory_compact_writer import CompactGaussianWriter
from models.gaussian_memory_transport import frozen_memory, read_inputs

OUT = ROOT / 'results/gaussian_memory/joint_write1'
SEEDS = previous.SEEDS
NEW = ('full_gaussian', 'full_center', 'content_gaussian')
KINDS = NEW + previous.KINDS
DOC = ROOT / 'docs/gaussian_memory_joint_write_plan.md'


def model_path(kind, seed, initial=False):
    if kind not in NEW:
        return previous.path(kind, seed, initial)
    return OUT / f'{kind}_{seed}' / ('initial.pt' if initial else 'step_2000.pt')


def make_models(kind, plan):
    if kind not in NEW:
        return previous.model(kind, plan), None
    kernel = kind.split('_')[-1]
    return previous.model(kernel, plan), CompactGaussianWriter(kind.startswith('full'))


def seeds_on(raw, device):
    return {k: raw[k].float().to(device) for k in ('seed_mean', 'seed_chol', 'seed_teacher')}


def checked(locked=False):
    plan = json.loads((OUT / 'plan.json').read_text())
    assert sha256(DOC) == plan['document_sha256']
    for name, digest in plan['source_sha256'].items():
        assert sha256(ROOT / name) == digest, name
    for name, digest in plan['old_models_sha256'].items():
        assert sha256(name) == digest, name
    if locked:
        lock = json.loads((OUT / 'evaluation_lock.json').read_text())
        assert sha256(OUT / 'plan.json') == lock['plan_sha256']
        for name, digest in lock['models'].items():
            assert sha256(name) == digest, name
    return plan


def prepare():
    old = previous.locked()
    cache = json.loads((previous.OUT / 'cache_identity.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'plan.json').exists()
    catalog = old['catalog'] + cache['catalog']
    identities = dict(cache['sha256'])
    for m in old['catalog']:
        for name in (m['raw'], m['tokens'], DATA / 'surface_v1' / (m['id'] + '.pt'),
                     previous.OLD / 'memories' / (m['id'] + '_predicted.pt')):
            identities[str(name)] = sha256(name)
    sources = dict(old['source_sha256'])
    for name in ('models/gaussian_memory_compact_writer.py', 'experiments/gaussian_memory_joint_write.py'):
        sources[name] = sha256(ROOT / name)
    plan = {**{k: old[k] for k in ('training_ids', 'primary_places', 'steps', 'batch', 'lr',
                                   'weight_decay', 'center_variance', 'gradient_clip')},
            'catalog': catalog, 'seeds': SEEDS, 'new_kinds': NEW, 'all_kinds': KINDS,
            'input_sha256': identities, 'source_sha256': sources,
            'old_models_sha256': {str(previous.path(k, s)): sha256(previous.path(k, s))
                                  for k in previous.KINDS for s in SEEDS},
            'document_sha256': sha256(DOC), 'confirmation_claim': False,
            'previous_scores_sha256': sha256(previous.OUT / 'evaluation/scores.json'),
            'old_plan_sha256': sha256(previous.OUT / 'plan.json'),
            'primary': '32 used non-training places, clips averaged within place',
            'writer_input': 'same reference state, no 2048 context, query RGB or sensor depth'}
    for seed in SEEDS:
        for kind in NEW:
            torch.manual_seed(seed)
            reader, writer = make_models(kind, plan)
            init = torch.load(previous.path(kind.split('_')[-1], seed, True),
                              map_location='cpu', weights_only=False)['model']
            reader.load_state_dict(init)
            config = {'kind': kind, 'seed': seed, 'reader_parameters': sum(p.numel() for p in reader.parameters()),
                      'writer_parameters': sum(p.numel() for p in writer.parameters()),
                      'inactive_parameters': 650 if kind.startswith('content') else 0}
            dest = model_path(kind, seed, True); dest.parent.mkdir()
            torch.save({'model': reader.state_dict(), 'writer': writer.state_dict(), 'config': config}, dest)
            identities[str(dest)] = sha256(dest)
    write_json(OUT / 'plan.json', plan)
    write_json(OUT / 'status.json', {'complete': False, 'pending': ['train', 'evaluate', 'checks', 'analysis', 'visual_review', 'report']})
    print(json.dumps({'prepared': True, 'new_fits': 9, 'old_models': 15, 'clips': len(catalog),
                      'writer_parameters': config['writer_parameters'], 'fresh_confirmation': False}), flush=True)


def train():
    plan = checked(); torch.set_num_threads(4); data = []
    for filename, digest in plan['input_sha256'].items():
        assert sha256(filename) == digest, filename
    for rid in plan['training_ids']:
        meta = next(m for m in plan['catalog'] if m['id'] == rid)
        d = previous.load_record(meta); raw, rec, target = d['raw'], d['rec'], d['target']
        keep = torch.tensor([r == 'train' for r in rec['roles']]); q = rec['query'][keep]
        assert len(q) == 24 and not set(q.tolist()) & {0, 2, 4, 6, 8, 10, 12}
        seeds = seeds_on(raw, 'cuda')
        baseline = frozen_memory(seeds)
        assert all(torch.allclose(baseline[k].cpu(), d['memory'][k], atol=1e-7, rtol=1e-6) for k in baseline)
        data.append({'id': rid, 'x': rec['input_tokens'][keep].cuda(), 'q': q.cuda(),
                     'y': target['log_depth_m'][q].cuda(), 'valid': target['valid'][q].cuda(),
                     'c2w': raw['c2w'].cuda(), 'K': raw['K'].cuda(),
                     'unit': raw['diagnostics']['unit_meters'], 'seeds': seeds})
    for seed in SEEDS:
        for kind in NEW:
            dest = model_path(kind, seed); assert not dest.exists()
            initial = torch.load(model_path(kind, seed, True), map_location='cuda', weights_only=False)
            reader, writer = make_models(kind, plan); reader.cuda(); writer.cuda()
            reader.load_state_dict(initial['model']); writer.load_state_dict(initial['writer'])
            with torch.no_grad():
                memory = writer(data[0]['seeds']); base = frozen_memory(data[0]['seeds'])
                identity = max(float((memory[k] - base[k]).abs().max()) for k in memory)
                assert identity < 1e-6
            parameters = list(reader.parameters()) + list(writer.parameters())
            opt = torch.optim.AdamW(parameters, lr=plan['lr'], weight_decay=plan['weight_decay'])
            rng = np.random.default_rng(seed + 73); losses = []; gradients = []
            torch.cuda.synchronize(); start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
            with (dest.parent / 'training.jsonl').open('w') as log:
                for step in range(1, 2001):
                    slot = int(rng.integers(len(data))); d = data[slot]
                    ids = rng.choice(len(d['x']), 2, replace=False).tolist(); q = d['q'][ids]
                    opt.zero_grad(set_to_none=True)
                    pred = reader.read(d['x'][ids], writer(d['seeds']), d['c2w'][q], d['K'][q], d['unit'])
                    loss = loss_function(pred, d['y'][ids], d['valid'][ids]); assert torch.isfinite(loss)
                    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(parameters, plan['gradient_clip'])
                    assert torch.isfinite(norm)
                    if step in (2, 100, 2000):
                        g = writer.net[-1].weight.grad
                        gradients.append({'step': step, 'center': float(g[:3].norm()), 'shape': float(g[3:9].norm()),
                                          'opacity': float(g[9].norm()), 'content': float(g[10:].norm())})
                    opt.step(); losses.append(float(loss))
                    log.write(json.dumps({'step': step, 'slot': slot, 'episode': d['id'], 'sample_ids': ids,
                                          'loss': float(loss), 'gradient_norm': float(norm)}) + '\n')
                    if step % 500 == 0:
                        log.flush(); print(json.dumps({'training': kind, 'seed': seed, 'step': step,
                                                       'last100': float(np.mean(losses[-100:]))}), flush=True)
            torch.cuda.synchronize()
            torch.save({'model': reader.state_dict(), 'writer': writer.state_dict(), 'config': initial['config'],
                        'plan_sha256': sha256(OUT / 'plan.json')}, dest)
            status = {'complete': True, 'kind': kind, 'seed': seed, 'steps': 2000, 'seconds': time.monotonic() - start,
                      'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3, 'initial_memory_max_difference': identity,
                      'first100': float(np.mean(losses[:100])), 'last100': float(np.mean(losses[-100:])),
                      'writer_gradients': gradients, 'model_sha256': sha256(dest)}
            write_json(dest.parent / 'status.json', status); print(json.dumps({'fit_complete': status}), flush=True)
            del reader, writer, opt, parameters; torch.cuda.empty_cache()
    write_json(OUT / 'evaluation_lock.json', {'plan_sha256': sha256(OUT / 'plan.json'),
        'models': {str(model_path(k, s)): sha256(model_path(k, s)) for k in KINDS for s in SEEDS}})


def conditions(kind):
    if kind in NEW:
        return ('normal', 'removed', 'shuffled', 'wrong_camera', 'initial_geometry', 'initial_memory')
    return ('normal',)


@torch.inference_mode()
def evaluate():
    plan = checked(True); torch.set_num_threads(4); nets = {}
    for kind in KINDS:
        for seed in SEEDS:
            reader, writer = make_models(kind, plan)
            state = torch.load(model_path(kind, seed), map_location='cuda', weights_only=False)
            reader.cuda().eval(); reader.load_state_dict(state['model'])
            if writer is not None:
                writer.cuda().eval(); writer.load_state_dict(state['writer'])
            nets[(kind, seed)] = (reader, writer)
    dest = OUT / 'evaluation'; assert not dest.exists(); dest.mkdir()
    memory_dir = OUT / 'memories'; memory_dir.mkdir()
    rows = []; full_rows = []; updates = []; identities = {}; memory_ids = {}; counts = []
    for meta in plan['catalog']:
        d = previous.load_record(meta); raw, rec, target = d['raw'], d['rec'], d['target']; rid = meta['id']
        keep = torch.tensor([r == 'evaluation' for r in rec['roles']]); q = rec['query'][keep]; uq = sorted(set(q.tolist()))
        x = rec['input_tokens'][keep].cuda(); y = target['log_depth_m'][q].cuda(); valid = target['valid'][q].cuda()
        mask = rec['mask'][keep].cuda(); full_x = rec['teacher_tokens'][uq].cuda()
        full_y, full_v = target['log_depth_m'][uq].cuda(), target['valid'][uq].cuda()
        camera, K, unit = raw['c2w'].cuda(), raw['K'].cuda(), raw['diagnostics']['unit_meters']
        seeds = seeds_on(raw, 'cuda'); base = {k: v.cuda() for k, v in d['memory'].items()}
        generator = torch.Generator().manual_seed(int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16))
        perm = torch.randperm(256, generator=generator).cuda()
        predictions = {}; full_predictions = {}
        tag = {'episode': rid, 'scene': meta['row']['scene'],
               'split': 'used_summary20' if meta['eval_split'] == 'confirmation' else meta['eval_split']}
        counts.append(dict(tag, queries=len(uq), masked_inputs=len(q)))
        for (kind, seed), (net, writer) in nets.items():
            memory = base
            if writer is not None:
                written = {k: v.cpu().contiguous() for k, v in writer(seeds).items()}
                filename = memory_dir / f'{rid}_{kind}_{seed}.pt'
                torch.save({'memory': written, 'unit': unit, 'reference_frames': [0, 4, 8, 12],
                            'writer_checkpoint': str(model_path(kind, seed)), 'query_in_writer': False}, filename)
                memory_ids[str(filename)] = sha256(filename)
                memory = torch.load(filename, map_location='cuda', weights_only=False)['memory']
                assert all(torch.equal(memory[k].cpu(), written[k]) for k in memory)
                relative = torch.linalg.norm(memory['covars'] - base['covars'], dim=(-1, -2)) / torch.linalg.norm(base['covars'], dim=(-1, -2))
                updates.append(dict(tag, kind=kind, seed=seed,
                    center_displacement=float((memory['means'] - base['means']).norm(dim=-1).mean()),
                    center_displacement_m=float((memory['means'] - base['means']).norm(dim=-1).mean()) * unit,
                    covariance_relative_change=float(relative.mean()),
                    opacity_abs_change=float((memory['opacities'] - base['opacities']).abs().mean()),
                    feature_delta_rms=float((memory['features'] - base['features']).square().mean().sqrt()),
                    minimum_eigenvalue=float(torch.linalg.eigvalsh(memory['covars']).min()),
                    saved_roundtrip_max_difference=0.))
            g = read_inputs(memory, camera, K, unit)
            geometry = {'normal': g}
            if writer is not None:
                geometry['shuffled'] = dict(g, key=torch.cat([g['key'][:, perm, :32], g['key'][..., 32:]], -1), value=g['value'][:, perm])
                geometry['wrong_camera'] = read_inputs(memory, camera, K, unit, camera[(torch.arange(13, device='cuda') + 4) % 13])
                reset_geometry = dict(base, features=memory['features'])
                geometry['initial_geometry'] = read_inputs(reset_geometry, camera, K, unit)
                geometry['initial_memory'] = read_inputs(base, camera, K, unit)
            for condition in conditions(kind):
                inputs = {k: v[q] for k, v in geometry.get(condition, g).items()}
                p = net.read(x, None, camera[q], K[q], unit) if kind == 'large_local' else net(x, inputs, condition)
                assert torch.isfinite(p).all()
                result = dict(tag, kind=kind, seed=seed, condition=condition, **metrics(p, y, valid, mask))
                if condition == 'normal':
                    result.update({k: v for k, v in decompose(p, y, valid).items() if k != 'log_mse'})
                    full = (net.read(full_x, None, camera[uq], K[uq], unit) if kind == 'large_local'
                            else net(full_x, {k: v[uq] for k, v in g.items()}))
                    full_rows.append(dict(tag, kind=kind, seed=seed, condition='normal',
                        **metrics(full, full_y, full_v, torch.zeros_like(full_v)),
                        **{k: v for k, v in decompose(full, full_y, full_v).items() if k != 'log_mse'}))
                    full_predictions[f'{kind}_{seed}'] = full.cpu()
                rows.append(result); predictions[f'{kind}_{seed}_{condition}'] = p.cpu()
            if kind == 'content_gaussian':
                assert torch.equal(predictions[f'{kind}_{seed}_normal'], predictions[f'{kind}_{seed}_initial_geometry'])
        filename = dest / (rid + '_maps.pt')
        torch.save({'episode': meta['row'], 'split': tag['split'], 'query': q, 'full_query': uq,
                    'target': y.cpu(), 'valid': valid.cpu(), 'mask': mask.cpu(),
                    'full_target': full_y.cpu(), 'full_valid': full_v.cpu(), 'predictions': predictions,
                    'full_predictions': full_predictions, 'rgb': raw['rgb'], 'permutation': perm.cpu()}, filename)
        identities[str(filename)] = sha256(filename)
        print(json.dumps({'evaluated': rid, 'split': tag['split'], 'predictions': len(predictions)}), flush=True)
    write_json(dest / 'scores.json', {'rows': rows, 'full_query': full_rows, 'updates': updates,
        'maps_sha256': identities, 'memory_sha256': memory_ids, 'counts': counts, 'confirmation_claim': False})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('prepare', 'train', 'evaluate'))
    args = parser.parse_args(); start = time.monotonic()
    try:
        globals()[args.stage]()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / f'failure_{args.stage}_{int(time.time())}.json',
                   {'stage': args.stage, 'traceback': traceback.format_exc(), 'scientific_result': False})
        raise
    write_json(OUT / f'{args.stage}_complete.json', {'complete': True, 'seconds': time.monotonic() - start})


if __name__ == '__main__':
    main()
