"""Independent CPU label/routing checks and paired descriptive analysis."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_addressability import (
    OUT, PARENT, TRAIN, ROOT, SOURCE_FRAMES, CONDITIONS, POLICIES, SEEDS, KINDS, read, load, sha256, write_json)
from experiments.gaussian_memory_incremental_data import sensor_target
from experiments.gaussian_memory_visibility_checks import cloud_numpy, index_numpy


def independent_pairs(meta, masks, persistent):
    row = meta['row']; cameras = np.asarray(row['c2w_metric']); K = np.asarray(row['K'])
    target, valid = sensor_target(row['frames'][11]); target, valid = target.numpy().ravel(), valid.numpy().ravel()
    yy, xx = np.mgrid[:210, :280]; patch = ((yy // 14) * 20 + xx // 14).ravel()
    codes = []
    for fi, f in enumerate(SOURCE_FRAMES):
        depth, ok = sensor_target(row['frames'][f]); ok = ok.numpy()
        if persistent and f >= 4: ok = ok & ~masks[f][patch].reshape(210, 280)
        cloud = cloud_numpy(depth.numpy().astype(float), ok, cameras[f], K[f])
        inside, uv, z = index_numpy(cloud, cameras[11], K[11])
        qp = uv[:, 1] * 280 + uv[:, 0]; sp = patch[ok.ravel()][inside] + fi * 300
        same = valid[qp] & (np.abs(z - target[qp]) <= .1 * target[qp])
        codes.append(qp[same] * 2400 + sp[same])
    codes = np.unique(np.concatenate(codes)); qp, sp = codes // 2400, codes % 2400
    early, late = np.zeros(58800, bool), np.zeros(58800, bool)
    early[qp[sp < 1200]] = True; late[qp[sp >= 1200]] = True
    exclusive = early & ~late; counts = np.bincount(patch[valid], minlength=300).astype(float)
    matrix = {}
    for name, mask in [('all', np.ones(len(qp), bool)), ('early_exclusive', exclusive[qp])]:
        m = np.bincount(patch[qp[mask]] * 2400 + sp[mask], minlength=300 * 2400).reshape(300, 2400)
        matrix[name] = m / np.maximum(counts, 1)[:, None]
    return codes, matrix, counts, exclusive, patch


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True); e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def check():
    start = time.monotonic(); plan = read(OUT / 'plan.json'); discrepancies = []; clips = []
    for name, digest in plan['source_sha256'].items():
        assert sha256(ROOT / name) == digest == sha256(OUT / 'source_snapshot' / name)
    for path, digest in plan['inputs'].items(): assert sha256(path) == digest
    for spec in plan['models']: assert sha256(spec['path']) == spec['sha256']
    maxima = {'label_matrix': 0., 'slot_utility': 0., 'score': 0., 'hit_mass': 0., 'best': 0., 'uniform': 0.}
    row_count = 0; qk_count = 0; source_difference = 0; provenance_mismatch = 0
    for meta in plan['catalog']:
        rid = meta['id']; rec = read(OUT / 'clips' / f'{rid}.json')
        assert rec['complete'] and rec['plan_sha256'] == sha256(OUT / 'plan.json')
        assert sha256(rec['artifact_path']) == rec['artifact_sha256']
        artifact = load(rec['artifact_path']); roi = artifact['roi'].numpy()
        from experiments.gaussian_memory_visibility_data import CACHE
        masks = load(CACHE / rid / 'masks.pt')['patch_masks'].numpy()
        independent = {}; utilities = {}
        for condition in ('full', 'persistent'):
            codes, matrices, counts, exclusive, patch = independent_pairs(meta, masks, condition == 'persistent')
            saved = artifact['labels'][condition]; qp, sp = saved['pairs'].numpy().astype(np.int64)
            difference = np.setxor1d(codes, qp * 2400 + sp).size
            source_difference += int(difference)
            discrepancies.append({'id': rid, 'condition': condition, 'unique_pairs': len(codes), 'symmetric_difference': int(difference)})
            for name, matrix in matrices.items():
                err = float(np.abs(matrix - saved['matrix'][name].numpy()).max()); maxima['label_matrix'] = max(maxima['label_matrix'], err)
            assert np.array_equal(counts, saved['valid_counts'].numpy())
            independent[condition] = (matrices, counts, exclusive, patch, codes)
        for state, binding in artifact['bindings'].items():
            condition, policy = state.split('/'); matrices, counts, exclusive, patch, codes = independent[condition]
            near, w = binding['near'].numpy(), binding['weights'].numpy().astype(float)
            for name, matrix in matrices.items():
                u = (matrix[:, near] * w[None]).sum(-1)
                utilities[state, name] = u
                maxima['slot_utility'] = max(maxima['slot_utility'], float(np.abs(u - binding['utility'][name].numpy()).max()))
            qp, sp = codes // 2400, codes % 2400
            raw = np.ones(2400, bool); raw[:1200] = policy == 'history'
            stage_mask = {'raw': raw, 'admitted': binding['admitted'].numpy(), 'retained': binding['retained'].numpy()}
            qvalid = artifact['labels'][condition]['query_valid'].numpy()
            for pr in [r for r in rec['provenance'] if r['state'] == state]:
                region = qvalid & (roi[patch] if pr['region'] == 'roi' else np.ones(58800, bool))
                subset = region & (exclusive if pr['label'] == 'early_exclusive' else np.ones(58800, bool))
                assert pr['valid_pixels'] == int(region.sum()) and pr['subset_pixels'] == int(subset.sum())
                for name, keep in stage_mask.items():
                    hit = np.zeros(58800, bool); hit[qp[keep[sp]]] = True
                    provenance_mismatch = max(provenance_mismatch, abs(int((hit & subset).sum()) - pr[name + '_pixels']))
        qk = {(r['condition'], r['policy'], r['family'], r['seed']): r for r in artifact['qk']}
        qk_count += len(qk); attentions = {}
        for row in rec['rows']:
            condition = 'persistent' if row['condition'] == 'persistent' else 'full'
            state = condition + '/' + row['policy']; label = row['label']
            att_key = (row['condition'], row['policy'], row['family'], row['seed'], row['prior'])
            if att_key not in attentions:
                bias = artifact['priors'][state][row['prior']].numpy().astype(float)
                if row['family'] == 'prior_only': logits = bias[None]
                else:
                    item = qk[att_key[:4]]
                    q = item['q'].float().numpy().astype(float); k = item['k'].float().numpy().astype(float)
                    logits = np.einsum('hpd,hsd->hps', q, k) * .25 + bias[None]
                attentions[att_key] = softmax(logits)
            attention = attentions[att_key]; u = utilities[state, label]; count = independent[condition][1]
            weight = count * (roi if row['region'] == 'roi' else np.ones(300, bool))
            score = ((attention * u[None]).sum(-1).mean(0) * weight).sum() / weight.sum()
            best = (u.max(-1) * weight).sum() / weight.sum(); uniform = (u.mean(-1) * weight).sum() / weight.sum()
            maxima['score'] = max(maxima['score'], abs(float(score) - row['score']))
            maxima['best'] = max(maxima['best'], abs(float(best) - row['best_slot_score']))
            maxima['uniform'] = max(maxima['uniform'], abs(float(uniform) - row['uniform_score']))
            if row['hit_mass_on_supported'] is not None:
                supported = weight * (u.max(-1) > 0)
                mass = ((attention * (u > 0)[None]).sum(-1).mean(0) * supported).sum() / supported.sum()
                maxima['hit_mass'] = max(maxima['hit_mass'], abs(float(mass) - row['hit_mass_on_supported']))
            row_count += 1
        clips.append({'id': rid, 'rows': len(rec['rows']), 'qk_records': len(qk)})
        print(json.dumps({'checked': rid, 'scores': len(rec['rows'])}), flush=True)
    assert qk_count == 1512 and row_count == 19008
    assert source_difference == 0 and provenance_mismatch == 0
    assert max(maxima.values()) < 2e-6, maxima
    records = [read(OUT / 'clips' / f'{m["id"]}.json') for m in plan['catalog']]
    sdpa = [c for r in records for c in r['attention_checks']]
    assert len(sdpa) == 126 and records[0]['sentinel'] is not None
    result = {'complete': True, 'state_count': 48, 'qk_records': qk_count, 'independent_score_rows': row_count,
              'source_pair_symmetric_difference': source_difference, 'provenance_count_difference': provenance_mismatch,
              'max_abs': maxima, 'clips': clips, 'source_checks': discrepancies,
              'reconstruction_max_abs': {key: max(e[key] for r in records for e in r['reconstruction'].values())
                                         for key in ('means', 'covars', 'features')},
              'sdpa_reference_cases': len(sdpa), 'sdpa_max_relative_rms': max(r['sdpa_relative_rms'] for r in sdpa),
              'native_sentinel': records[0]['sentinel'], 'seconds': time.monotonic() - start,
              'checker_sha256': sha256(ROOT / 'experiments/gaussian_memory_addressability_analysis.py')}
    write_json(OUT / 'independent_checks.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_checks', 'clips')}), flush=True)


def summary(values, places):
    a = np.asarray(values, dtype=float)
    if a.ndim == 1: a = a[:, None]
    assert a.shape[0] == len(places) and np.isfinite(a).all()
    rng = np.random.default_rng(20261111)
    pi = rng.integers(len(places), size=(5000, len(places))); si = rng.integers(a.shape[1], size=(5000, a.shape[1]))
    boot = a[pi[:, :, None], si[:, None, :]].mean((1, 2))
    return {'mean': float(a.mean()), 'paired_95': np.quantile(boot, [.025, .975]).tolist(),
            'by_place': dict(zip(places, a.mean(1).tolist())), 'by_seed': a.mean(0).tolist(),
            'positive_places': int((a.mean(1) > 0).sum()), 'positive_seeds': int((a.mean(0) > 0).sum()), 'places': len(places)}


def comparison(a, b, places):
    # Positive means higher correspondence-weighted routing score, NOT lower task loss.
    d = summary(b - a, places)
    d.update(baseline_mean=float(a.mean()), candidate_mean=float(b.mean()),
             ratio=float(b.mean() / a.mean()) if a.mean() else None)
    return d


def analyze():
    plan = read(OUT / 'plan.json'); catalog = sorted(plan['catalog'], key=lambda m: m['place'])
    places = [m['place'] for m in catalog]; records = {m['id']: read(OUT / 'clips' / f'{m["id"]}.json') for m in catalog}
    indexed = {}
    for rid, rec in records.items():
        for row in rec['rows']:
            key = (rid, row['condition'], row['policy'], row['region'], row['label'], row['family'], row['prior'], row['seed'])
            assert key not in indexed; indexed[key] = row
    families = list(dict.fromkeys(m['family'] for m in plan['models'])) + ['prior_only']
    def matrix(c, p, r, l, family, prior, metric='score'):
        seeds = [None] if family == 'prior_only' else SEEDS
        return np.asarray([[indexed[m['id'], c, p, r, l, family, prior, s][metric] for s in seeds] for m in catalog], float)
    def paired_shape(a, b):
        n = max(a.shape[1], b.shape[1]); return np.broadcast_to(a, (12, n)), np.broadcast_to(b, (12, n))
    results = {}; within = {}; actual = {}; interactions = {}
    actual_kind = {m['family']: m['kind'] for m in plan['models']}
    for c in CONDITIONS:
        for p in POLICIES:
            for r in ('roi', 'all'):
                for l in ('all', 'early_exclusive'):
                    context = '/'.join((c, p, r, l)); results[context] = {}; within[context] = {}; actual[context] = {}; interactions[context] = {}
                    for fam in families:
                        for kind in KINDS:
                            a = matrix(c, p, r, l, fam, kind); key = fam + '/' + kind
                            result = summary(a, places)
                            for metric in ('best_slot_score', 'uniform_score', 'hit_mass_on_supported', 'uniform_hit_mass_on_supported'):
                                x = matrix(c, p, r, l, fam, kind, metric); mask = np.isfinite(x).all(1)
                                result[metric] = summary(x[mask], [places[i] for i in np.flatnonzero(mask)]) if mask.any() else None
                            best = matrix(c, p, r, l, fam, kind, 'best_slot_score').mean()
                            result['fraction_of_best_slot'] = float(a.mean() / best) if best else None
                            results[context][key] = result
                        for control in ('content', 'soft_center'):
                            a = matrix(c, p, r, l, fam, control); b = matrix(c, p, r, l, fam, 'soft')
                            within[context][fam + ':soft_vs_' + control] = comparison(a, b, places)
                    for fam, control in [('soft_dynamic', 'content_static'), ('soft_dynamic', 'content_dynamic'),
                                         ('soft_dynamic', 'center_dynamic'), ('new_soft', 'new_content'),
                                         ('new_soft', 'new_center'), ('new_soft', 'soft_dynamic')]:
                        a = matrix(c, p, r, l, control, actual_kind[control]); b = matrix(c, p, r, l, fam, actual_kind[fam])
                        actual[context][fam + '_vs_' + control] = comparison(a, b, places)
                    for fam in ('soft_dynamic', 'new_soft'):
                        qk_gain = matrix(c, p, r, l, fam, 'soft') - matrix(c, p, r, l, fam, 'content')
                        prior_gain = matrix(c, p, r, l, 'prior_only', 'soft') - matrix(c, p, r, l, 'prior_only', 'content')
                        a, b = paired_shape(prior_gain, qk_gain)
                        interactions[context][fam + ':qk_minus_prior_only_gain'] = summary(b - a, places)
    provenance = {}
    pindex = {(rid, row['state'], row['region'], row['label']): row for rid, rec in records.items() for row in rec['provenance']}
    for state in ('full/recent', 'full/history', 'persistent/recent', 'persistent/history'):
        for region in ('roi', 'all'):
            for label in ('all', 'early_exclusive'):
                values = [pindex[m['id'], state, region, label] for m in catalog]; output = {}
                for metric in ('raw_coverage', 'admitted_coverage', 'retained_coverage', 'admitted_over_raw',
                               'retained_over_admitted', 'retained_over_raw', 'admitted_patch_count', 'retained_patch_count',
                               'early_slots', 'early_mixture_weight'):
                    available = [i for i, v in enumerate(values) if v[metric] is not None]
                    output[metric] = summary([values[i][metric] for i in available], [places[i] for i in available]) if available else None
                output['zero_raw_support_places'] = [places[i] for i, v in enumerate(values) if not v['raw_pixels']]
                output['pooled_retained_over_raw'] = sum(v['retained_pixels'] for v in values) / sum(v['raw_pixels'] for v in values) if sum(v['raw_pixels'] for v in values) else None
                output['place_rows'] = values; provenance[state + '/' + region + '/' + label] = output
    workers = [read(OUT / f'worker_{i}.json') for i in range(2)]; assert all(w['complete'] for w in workers)
    result = {'complete': True, 'places': places, 'route': results, 'same_qk_comparisons': within,
              'actual_reader_comparisons': actual, 'qk_gain_interactions': interactions, 'provenance': provenance,
              'cost': {'workers': workers, 'new_training_steps': 0, 'new_download_bytes': 0, 'new_task_predictions': 0,
                       'native_sentinel_replays': 1, 'qk_records': 1512, 'scored_rows': len(indexed)},
              'scope': 'Previously used data; correspondence provenance proxy and direct query attention, not feature information or causal task utility.',
              'analysis_source_sha256': sha256(ROOT / 'experiments/gaussian_memory_addressability_analysis.py')}
    write_json(OUT / 'analysis.json', result)
    context = 'persistent/history/roi/all'
    print(json.dumps({'provenance': {k: provenance['persistent/history/roi/all'][k] for k in ('raw_coverage', 'retained_coverage', 'retained_over_raw')},
                      'same_qk': within[context], 'actual': actual[context]}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('check', 'analyze')); args = p.parse_args()
    torch.set_num_threads(4)
    (check if args.stage == 'check' else analyze)()
