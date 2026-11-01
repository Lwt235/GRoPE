"""Independent arithmetic audit of fresh-camera predictions; no model import."""
import hashlib
import json
from pathlib import Path
import numpy as np


ROOT = Path('results/round3_normalized_confirmation')
SEEDS = [11, 22, 33, 44, 55]
SCENES = ['teatime', 'waldo_kitchen', 'figurines', 'temple']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary = json.loads((ROOT/'fresh_summary.json').read_text())
    verification = json.loads((ROOT/'verification.json').read_text())
    protocol = json.loads((ROOT/'frozen_protocol.json').read_text())
    if sha(ROOT/'frozen_protocol.json') != summary['protocol_sha256']:
        raise ValueError('protocol digest mismatch')
    for path, expected in verification['prediction_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'prediction changed: {path}')
    for path, expected in verification['checkpoint_first_replay_hashes'].items():
        if sha(path) != expected:
            raise ValueError(f'checkpoint changed: {path}')
    labels, valid, cameras = {}, {}, {}
    for scene in SCENES:
        folder = Path('results/round3_fresh_real_data')/scene
        manifest = json.loads((folder/'manifest.json').read_text())
        for alt in [False, True]:
            suffix = '_alt' if alt else ''
            ys, masks, qids = [], [], []
            for record in manifest['pairs']:
                path = folder/record['file']
                if sha(path) != record['npz_sha256']:
                    raise ValueError('source NPZ mismatch')
                with np.load(path) as d:
                    y = d['soft_labels'].copy()
                    retained = (y*d['k_valid'+suffix][None]).sum(-1)
                    mask = d['label_valid'] & d['q_valid'+suffix] & (retained >= .95)
                    y = (y*d['k_valid'+suffix][None]/np.maximum(retained[:,None], 1e-10)).astype(np.float32)
                    ys.append(y)
                    masks.append(mask)
                    qids.extend([int(record['q_id'])]*int(mask.sum()))
            labels[scene,alt] = np.stack(ys).astype(np.float64)
            valid[scene,alt] = np.stack(masks)
            cameras[scene,alt] = np.array(qids)
    values, max_error, mass_error = {}, 0., 0.
    for record in summary['records']:
        kind, scene, alt = record['kind'], record['scene'], record['alternate']
        seed, method = record['seed'], record['method']
        stem = f'{kind}-{scene}-{int(alt)}-'
        stem += (f'{seed}-' if seed is not None else '')+method+'.npz'
        with np.load(ROOT/'predictions'/stem) as f:
            p = f['probability'].astype(np.float64)
            np.testing.assert_array_equal(f['valid'], valid[scene,alt])
        if not np.isfinite(p).all() or (p < 0).any():
            raise ValueError('invalid probability')
        mass_error = max(mass_error, float(abs(p.sum(-1)-1).max()))
        y, mask = labels[scene,alt], valid[scene,alt]
        ce = -np.einsum('ijk,ijk->ij', y, np.log(p.clip(1e-30)))
        entropy = -np.einsum('ijk,ijk->ij', y, np.log(y.clip(1e-30)))
        midpoint = (p+y)/2
        js = .5*np.sum(y*np.log(y.clip(1e-30)/midpoint.clip(1e-30))+
                       p*np.log(p.clip(1e-30)/midpoint.clip(1e-30)), -1)
        top = p.argmax(-1)
        mass = np.take_along_axis(y, top[...,None], -1)[...,0]
        metrics = dict(cross_entropy=ce, kl=ce-entropy, js=js,
            top1_target_mass=mass, dominant_region_accuracy=(top == y.argmax(-1)),
            top1_max_bin_accuracy=(mass >= y.max(-1)-1e-8))
        for metric, array in metrics.items():
            max_error = max(max_error, abs(float(array[mask].mean())-record['summary'][metric]))
        key = kind,scene,alt,method
        if key not in values:
            values[key] = {}
        values[key][seed] = metrics['cross_entropy'][mask]
    if mass_error > 2e-6 or max_error > 2e-12:
        raise ValueError(f'probability/summary arithmetic mismatch {mass_error}, {max_error}')
    # Independent bootstrap implementation: multinomial camera multiplicities,
    # rather than indexing flattened observation draws in the original helper.
    intervals = {}
    for kind in ['raw','frozen']:
        for alt in [False,True]:
            for control in ['normalized_fixed','normalized_trace','normalized_area','projected_center_rbf']:
                sums, counts, strata = [], [], []
                for scene in ['figurines','temple']:
                    camera = cameras[scene,alt]
                    ids = np.unique(camera)
                    delta = []
                    for seed in SEEDS:
                        full = values[kind,scene,alt,'normalized_shape'][seed]
                        group = values[kind,scene,alt,control]
                        other = group[None] if None in group else group[seed]
                        delta.append(full-other)
                    delta = np.stack(delta)
                    sums.append(np.stack([delta[:,camera==c].sum(-1) for c in ids],axis=1))
                    counts.append(np.array([(camera==c).sum() for c in ids]))
                    strata.append(len(ids))
                numerator, denominator = np.concatenate(sums,axis=1), np.concatenate(counts)
                rng = np.random.default_rng(811751)
                boot = []
                for _ in range(4000):
                    weights = np.concatenate([rng.multinomial(n, np.full(n,1/n)) for n in strata])
                    selected_seeds = rng.choice(5,5,replace=True)
                    boot.append(float((numerator[selected_seeds]@weights).mean()/(denominator@weights)))
                result = summary['comparisons'][f'{kind}/{int(alt)}/external/{control}']['cross_entropy']
                mean = float(numerator.sum(-1).mean()/denominator.sum())
                np.testing.assert_allclose(mean, result['mean'], atol=2e-12, rtol=0)
                ci = np.quantile(boot,[.025,.975]).tolist()
                intervals[f'{kind}/{int(alt)}/{control}'] = dict(
                    mean=mean, independent_ci95=ci, original_ci95=result['ci95_seed_camera'])
                if ci[1] >= 0:
                    raise ValueError('independent CI does not support the primary support/shape claims')
    block_sensitivity = {}
    for block_size in [5, 10]:
        for kind in ['raw','frozen']:
            for alt in [False,True]:
                for control in ['normalized_fixed','normalized_trace','normalized_area','projected_center_rbf']:
                    sums, counts, strata = [], [], []
                    for scene in ['figurines','temple']:
                        camera = cameras[scene,alt]
                        # Export order follows image filenames. Preserve encounter
                        # order rather than sorting SfM's arbitrary integer IDs.
                        ordered = list(dict.fromkeys(camera.tolist()))
                        group_of = {c: i//block_size for i,c in enumerate(ordered)}
                        blocks = np.array([group_of[c] for c in camera])
                        ids = np.unique(blocks)
                        delta = []
                        for seed in SEEDS:
                            group = values[kind,scene,alt,control]
                            delta.append(values[kind,scene,alt,'normalized_shape'][seed] -
                                         (group[None] if None in group else group[seed]))
                        delta = np.stack(delta)
                        sums.append(np.stack([delta[:,blocks==c].sum(-1) for c in ids],axis=1))
                        counts.append(np.array([(blocks==c).sum() for c in ids]))
                        strata.append(len(ids))
                    numerator, denominator = np.concatenate(sums,axis=1), np.concatenate(counts)
                    rng = np.random.default_rng(811751)
                    boot = []
                    for _ in range(4000):
                        weights = np.concatenate([rng.multinomial(n,np.full(n,1/n)) for n in strata])
                        seeds = rng.choice(5,5,replace=True)
                        boot.append(float((numerator[seeds]@weights).mean()/(denominator@weights)))
                    block_sensitivity[f'{block_size}/{kind}/{int(alt)}/{control}'] = dict(
                        query_cameras_per_block=block_size, blocks=sum(strata),
                        ci95_seed_camera_block=np.quantile(boot,[.025,.975]).tolist(),
                        scope='post-hoc correlation sensitivity, not an independent primary test')
    output = dict(no_model_code_imported=True, audited_records=len(summary['records']),
        probability_archives=len(verification['prediction_sha256']),
        checkpoints=len(verification['checkpoint_first_replay_hashes']),
        max_probability_mass_error=mass_error, max_metric_error=max_error,
        independent_external_ce_bootstrap=intervals,
        adjacent_camera_block_sensitivity=block_sensitivity,
        audit_source_sha256=sha(__file__),
        summary_sha256=sha(ROOT/'fresh_summary.json'))
    (ROOT/'independent_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='independent_external_ce_bootstrap'}))


if __name__ == '__main__':
    main()
