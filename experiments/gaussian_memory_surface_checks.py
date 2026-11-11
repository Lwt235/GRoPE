"""Proportional checks of label identity, matched training and saved-state reads."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_surface_data import OUT, CACHE, SEEDS, VARIANTS, checked, model_path, load_record
from models.gaussian_memory_surface import SurfaceMemory


def main():
    plan = checked(True); torch.set_num_threads(4)
    training = [r for r in plan['catalog'] if r['id'] in plan['training_ids']]
    places = {r['row']['scene'].split('/')[0] for r in training}
    assert len(places) == 12 and not places & set(plan['primary_places'])
    lock = json.loads((OUT / 'evaluation_lock.json').read_text()); runs = []; sampling = {}
    for path, digest in lock['models'].items(): assert sha256(path) == digest
    for seed in SEEDS:
        common = torch.load(model_path('gaussian', seed, True), map_location='cpu', weights_only=False)['model']
        expected = None
        for variant in VARIANTS:
            directory = model_path(variant, seed).parent
            logs = [json.loads(s) for s in (directory / 'training.jsonl').read_text().splitlines()]
            assert len(logs) == 2000 and [r['step'] for r in logs] == list(range(1, 2001))
            assert np.isfinite([[r['loss'], r['gradient_norm']] for r in logs]).all()
            order = [(r['episode'], r['slot'], r['sample_ids']) for r in logs]
            if expected is None: expected = order
            else: assert order == expected
            initial = torch.load(model_path(variant, seed, True), map_location='cpu', weights_only=False)['model']
            assert torch.count_nonzero(initial['output.weight']) == 0 and torch.count_nonzero(initial['output.bias']) == 0
            if variant != 'local': assert initial.keys() == common.keys() and all(torch.equal(initial[k], common[k]) for k in common)
            status = json.loads((directory / 'status.json').read_text()); assert status['complete']
            if variant == 'fixed': assert all(r['center'] == 0 and r['shape'] == 0 for r in status['writer_gradients'])
            elif variant != 'local': assert all(status['writer_gradients'][-1][k] > 0 for k in ('center', 'shape', 'opacity', 'content'))
            runs.append(dict(variant=variant, seed=seed, **status))
        sampling[str(seed)] = {'identical_across_four_variants': True, 'steps': len(expected)}
    coordinates = []; target_count = 0; independent_labels = []
    for meta in plan['catalog']:
        d = load_record(meta); raw, rec, target = d['raw'], d['rec'], d['target']
        assert target['supervision_only'] and not target['writer_sensor_used'] and target['reference_rgb_only']
        assert not set(target['query_ids']) & {0, 4, 8, 12}
        keep = torch.tensor([r == 'train' for r in rec['roles']])
        if meta['id'] in plan['training_ids']:
            assert int(keep.sum()) == 24
            assert set(rec['query'][keep].tolist()).isdisjoint(rec['query'][~keep].tolist())
        unit = raw['diagnostics']['unit_meters']
        metric = torch.tensor(meta['row']['c2w_metric'], dtype=torch.float32)
        c2w = raw['c2w'].clone(); c2w[:, :3, 3] *= unit
        error = float((c2w - metric).abs().max()); assert error < 2e-5
        K = torch.tensor(meta['row']['K'], dtype=torch.float32) / torch.tensor([14., 14., 1.])[:, None]
        intrinsic_error = float((K - raw['K']).abs().max()); assert intrinsic_error < 2e-5
        coordinates.append({'episode': meta['id'], 'camera_metric_max_diff': error, 'K_patch_max_diff': intrinsic_error})
        target_count += len(target['query_ids'])
        if meta['id'] in ('fit_00', 'coverage_04', 'fidelity_07'):
            source = target['sources'][0]; q = source['query']
            assert sha256(source['path']) == source['sha256']
            with Image.open(source['path']) as im: encoded = np.asarray(im).astype('uint32')
            metres = ((encoded >> 3) + ((encoded & 7) << 13)).astype('float32') / 1000
            # Use the specified PIL tie breaking, not a real-arithmetic floor
            # formula that selects an adjacent source pixel at exact boundaries.
            # Decode and per-patch aggregation remain independent implementations.
            iy = np.asarray(Image.fromarray(np.arange(metres.shape[0], dtype='int32')[:, None]).resize((1, 210), Image.Resampling.NEAREST))[:, 0]
            ix = np.asarray(Image.fromarray(np.arange(metres.shape[1], dtype='int32')[None]).resize((280, 1), Image.Resampling.NEAREST))[0]
            small = metres[iy[:, None], ix[None]]
            values = []; masks = []
            for row in range(15):
                for col in range(20):
                    a = small[row*14:(row+1)*14, col*14:(col+1)*14].flatten()
                    a = a[np.isfinite(a) & (a > .1) & (a < 10.)]
                    values.append(float(np.median(a)) if len(a) else 1.); masks.append(len(a) >= 98)
            label_error = float(np.max(np.abs(np.array(values) - target['depth_m'][q].numpy())))
            assert label_error < 1e-6 and np.array_equal(masks, target['valid'][q].numpy())
            independent_labels.append({'episode': meta['id'], 'query': q, 'max_depth_diff_m': label_error})
    assert target_count == 360
    scores = json.loads((OUT / 'evaluation/scores.json').read_text())
    assert len(scores['rows']) == 46 * 60 and len(scores['full_query']) == 46 * 12
    state_diff = max(r['state_max_diff'] for r in scores['memory_replay']); assert state_diff == 0
    memory_identity = json.loads((OUT / 'memory_identity.json').read_text())
    for path, digest in memory_identity['sha256'].items(): assert sha256(path) == digest
    for row in memory_identity['changes']:
        if row['variant'] == 'fixed': assert row['center_mean_shift'] == 0 and abs(row['covariance_trace_ratio'] - 1) < 1e-6
    independent_reads = []; independent_scores = []
    for rid in ('fit_00', 'transfer_08', 'fidelity_07'):
        meta = next(r for r in plan['catalog'] if r['id'] == rid); d = load_record(meta)
        raw, rec, target = d['raw'], d['rec'], d['target']; sample = rec['roles'].index('evaluation'); q = int(rec['query'][sample])
        maps = torch.load(OUT / 'evaluation' / (rid + '_maps.pt'), map_location='cpu', weights_only=False)
        for variant in VARIANTS:
            seed = SEEDS[0]; net = SurfaceMemory(variant, plan['local_width'], plan['center_variance']).eval()
            net.load_state_dict(torch.load(model_path(variant, seed), map_location='cpu', weights_only=False)['model'])
            memory = None
            if variant != 'local':
                path = OUT / 'memories' / f'{rid}_{variant}_{seed}.pt'
                memory = torch.load(path, map_location='cpu', weights_only=False)['memory']
                del net.writer  # A separate CPU reader cannot access or reconstruct references.
            with torch.inference_mode():
                pred = net.read(rec['input_tokens'][sample:sample+1], memory, raw['c2w'][q:q+1], raw['K'][q:q+1], raw['diagnostics']['unit_meters'])[0]
            saved = maps['predictions'][f'{variant}_{seed}_normal'][0]
            diff = float((pred - saved).abs().max()); assert diff < 2e-4, (rid, variant, diff)
            independent_reads.append({'episode': rid, 'variant': variant, 'max_log_depth_diff_CPU_GPU': diff,
                                      'writer_removed': variant != 'local'})
            prediction = maps['predictions'][f'{variant}_{seed}_normal'].numpy().astype('float64')
            truth = maps['target'].numpy().astype('float64'); valid = maps['valid'].numpy()
            mse = float((((prediction - truth)**2 * valid).sum(-1) / valid.sum(-1)).mean())
            reported = next(r for r in scores['rows'] if r['episode'] == rid and r['variant'] == variant and r['seed'] == seed and r['condition'] == 'normal')['log_mse']
            error = abs(mse - reported); assert error < 1e-6
            independent_scores.append({'episode': rid, 'variant': variant, 'numpy_log_mse_difference': error})
        # Check the diagnostic is derived from fixed sensor associations, not each model's predictions.
        qids = maps['full_query']; camera = raw['c2w'][qids].numpy(); K = raw['K'][qids].numpy(); unit = raw['diagnostics']['unit_meters']
        yy, xx = np.meshgrid(np.arange(15)+.5, np.arange(20)+.5, indexing='ij'); uv = np.stack([xx, yy, np.ones_like(xx)], -1).reshape(300, 3)
        rays = np.einsum('bij,pj->bpi', np.linalg.inv(K.astype('float64')), uv)
        gtdepth = target['depth_m'][qids].numpy(); gt = np.einsum('bij,bpj->bpi', camera[:, :3, :3], rays * gtdepth[..., None]) + camera[:, None, :3, 3] * unit
        pairs = np.array(maps['correspondences']); a,i,b,j = pairs.T
        for variant in VARIANTS:
            pred = maps['full_predictions'][f'{variant}_{SEEDS[0]}'].numpy().astype('float64')
            world = np.einsum('bij,bpj->bpi', camera[:, :3, :3], rays * np.exp(pred[..., None])) + camera[:, None, :3, 3] * unit
            residual = world - gt; score = float(((residual[a,i] - residual[b,j])**2).sum(-1).mean())
            reported = next(r for r in scores['consistency'] if r['episode'] == rid and r['variant'] == variant and r['seed'] == SEEDS[0])['residual_xyz_mse']
            assert abs(score-reported) < max(1e-6, abs(score)*1e-5)
    write_json(OUT / 'checks.json', {'complete': True, 'new_runs': runs, 'sampling': sampling,
               'targets': target_count, 'coordinates': coordinates, 'independent_labels': independent_labels,
               'saved_memory_exact_replay_max_diff': state_diff, 'independent_CPU_reads': independent_reads,
               'independent_numpy_scores': independent_scores, 'correspondence_score_recomputed': True,
               'confirmation_claim': False, 'primary_train_place_disjoint': True,
               'source_sha256': sha256(Path(__file__)), 'plan_sha256': sha256(OUT / 'plan.json')})
    print(json.dumps({'checks_complete': True, 'runs': len(runs), 'targets': target_count,
                      'max_CPU_read_difference': max(r['max_log_depth_diff_CPU_GPU'] for r in independent_reads),
                      'max_numpy_score_difference': max(r['numpy_log_mse_difference'] for r in independent_scores)}))


if __name__ == '__main__': main()
