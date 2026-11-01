"""CPU-only exact-lift audit on the two already inspected validation places.

This reads no held-out scene and chooses no model/hyperparameter. The actual
GaussianQuadraticMatcher implementation is compared with the actual pair
kernel AND an independent float64 solve/slogdet reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from models.gaussian_quadratic_pe import GaussianQuadraticMatcher
from models.gaussian_measure_pe import gaussian_measure_log_bias


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('q', 'k', 'qmean', 'qcov', 'kmean', 'kcov',
          'q_to_k', 'q_to_kcov', 'k_to_q', 'k_to_qcov')
SCENES = {'harvard_c8/hv_c8_3', 'brown_bm_4/brown_bm_4'}
CONFIGS = [('point', .5, 5.), ('point', 1., 10.), ('fixed', .5, 5.),
           ('isotropic', .5, 5.), ('full', .5, 5.)]
SOURCE_PATHS = ('models/gaussian_quadratic_pe.py', 'models/gaussian_measure_pe.py',
                'experiments/round6_quadratic_audit.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def controls(d, method, reverse=False):
    """Independent control dispatch; does not import the model dispatcher."""
    source, target = ('k', 'q') if reverse else ('q', 'k')
    qm, km = d[source+'_to_'+target], d[target+'mean']
    qc, kc = d[source+'_to_'+target+'cov'], d[target+'cov']
    native_kc = kc
    if method == 'point':
        qc, kc = qc * 0, kc * 0
    elif method == 'fixed':
        qc = d[source+'cov']
    elif method == 'isotropic':
        eye = torch.eye(2, dtype=qc.dtype)
        qc = qc.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye
        kc = kc.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye
    elif method != 'full':
        raise ValueError(method)
    mass = 12 * torch.linalg.det(native_kc).sqrt()
    q, k = d[source], d[target]
    return qm, qc, km, kc, mass, q, k


def pair_score(d, method, width, temperature, reverse=False, reference=False):
    qm, qc, km, kc, mass, q, k = controls(d, method, reverse)
    if reference:
        covariance = qc[:, :, None] + kc[:, None] + width**2 * torch.eye(2, dtype=qc.dtype)
        delta = qm[:, :, None] - km[:, None]
        mahal = (delta * torch.linalg.solve(covariance, delta.unsqueeze(-1)).squeeze(-1)).sum(-1)
        logdet = torch.linalg.slogdet(covariance).logabsdet
        bias = mass.log()[:, None] - .5 * (mahal+logdet) - np.log(2*np.pi)
    else:
        bias = gaussian_measure_log_bias(qm, qc, km, kc, key_mass=mass, bandwidth=width)
    return temperature * (q @ k.transpose(-1, -2)) + bias


def metric_accumulator():
    return dict(max_probability_difference=0., max_valid_row_ce_difference=0.,
        absolute_weighted_ce_difference_sum=0., signed_weighted_ce_difference_sum=0.,
        visible_mass=0., argmax_mismatches=0, valid_argmax_mismatches=0,
        mismatches_with_reference_top_gap_gt_1e_4=0, max_mismatched_reference_top_gap=0.,
        max_centered_logit_difference=0., max_reference_regret_at_mismatched_key=0.,
        mismatches_beyond_numeric_score_bound=0)


def compare(acc, actual, reference, labels, mass, valid):
    actual, reference = actual.double(), reference.double()
    a, r = actual.softmax(-1), reference.softmax(-1)
    lp, lr = actual.log_softmax(-1), reference.log_softmax(-1)
    ace, rce = -(labels*lp).sum(-1), -(labels*lr).sum(-1)
    ce_delta = ace-rce
    acc['max_probability_difference'] = max(acc['max_probability_difference'], float((a-r).abs().max()))
    if valid.any():
        acc['max_valid_row_ce_difference'] = max(acc['max_valid_row_ce_difference'], float(ce_delta[valid].abs().max()))
    acc['absolute_weighted_ce_difference_sum'] += float((ce_delta.abs()*mass).sum())
    acc['signed_weighted_ce_difference_sum'] += float((ce_delta*mass).sum())
    acc['visible_mass'] += float(mass.sum())
    # Centering removes analytically irrelevant row constants before error bounds.
    centered_error = ((actual-actual.amax(-1, keepdim=True)) -
                      (reference-reference.amax(-1, keepdim=True))).abs().amax(-1)
    acc['max_centered_logit_difference'] = max(acc['max_centered_logit_difference'], float(centered_error.max()))
    ai, ri = actual.argmax(-1), reference.argmax(-1)
    mismatch = ai != ri
    acc['argmax_mismatches'] += int(mismatch.sum())
    acc['valid_argmax_mismatches'] += int((mismatch & valid).sum())
    top = reference.topk(2, dim=-1).values
    gap = top[..., 0]-top[..., 1]
    acc['mismatches_with_reference_top_gap_gt_1e_4'] += int((mismatch & (gap>1e-4)).sum())
    if mismatch.any():
        acc['max_mismatched_reference_top_gap'] = max(acc['max_mismatched_reference_top_gap'], float(gap[mismatch].max()))
        regret = reference.amax(-1)-reference.gather(-1, ai[..., None]).squeeze(-1)
        acc['max_reference_regret_at_mismatched_key'] = max(acc['max_reference_regret_at_mismatched_key'], float(regret[mismatch].max()))
        acc['mismatches_beyond_numeric_score_bound'] += int((mismatch & (regret>2*centered_error+1e-10)).sum())


def gradient_and_unit_audit(d, method, width, temperature):
    """Declared first pair, small fixed token prefix; no optimization step."""
    d = {k: v[:1, :min(v.shape[1], 13)].double() for k, v in d.items()}
    # Inputs represent symmetric support; use its symmetric extension when
    # comparing unconstrained matrix derivatives outside the SPD parameterization.
    leaves = {k: v.clone().requires_grad_() for k,v in d.items()}
    effective = {k: (v+v.transpose(-1,-2))*.5 if k.endswith('cov') else v
                 for k,v in leaves.items()}
    model = GaussianQuadraticMatcher(method, width, temperature).double()
    model.bandwidth.requires_grad_()
    values = torch.linspace(-.4, .8, d['k'].shape[1]*7, dtype=torch.float64).reshape(1,-1,7).requires_grad_()
    out = model.attend(effective, values)
    ref = pair_score(effective, method, model.bandwidth, model.temperature, reference=True).softmax(-1) @ values
    weight = torch.linspace(.2, .9, out.numel(), dtype=torch.float64).reshape_as(out)
    args = [values, leaves['q'], leaves['k'], leaves['q_to_k'], leaves['q_to_kcov'], model.bandwidth]
    ag = torch.autograd.grad((out*weight).sum(), args, allow_unused=True, retain_graph=True)
    rg = torch.autograd.grad((ref*weight).sum(), args, allow_unused=True)
    grad = {}
    for name, a, r in zip(('values','query_content','key_content','query_mean','query_covariance','bandwidth'), ag, rg):
        if a is None: a = torch.zeros_like(args[len(grad)])
        if r is None: r = torch.zeros_like(args[len(grad)])
        grad[name] = dict(max_abs_difference=float((a-r).abs().max()),
                          actual_norm=float(a.norm()), reference_norm=float(r.norm()),
                          finite=bool(torch.isfinite(a).all() & torch.isfinite(r).all()))
    original = GaussianQuadraticMatcher(method,width,temperature).double()(d).softmax(-1)
    scaled = dict(d)
    scale, shift = 8., torch.tensor([200., -137.], dtype=torch.float64)
    for key in ('qmean','kmean','q_to_k','k_to_q'):
        scaled[key] = d[key]*scale+shift
    for key in ('qcov','kcov','q_to_kcov','k_to_qcov'):
        scaled[key] = d[key]*scale**2
    transformed = GaussianQuadraticMatcher(method,width*scale,temperature).double()(scaled).softmax(-1)
    return dict(query_tokens=d['q'].shape[1], key_tokens=d['k'].shape[1],
        output_max_abs_difference=float((out-ref).abs().max()), gradients=grad,
        unit_rescale_translate_max_probability_difference=float((original-transformed).abs().max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT/'results/round6_quadratic_audit')
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = args.out/'source_snapshot'; snapshot.mkdir()
    hashes = {f: sha(ROOT/f) for f in SOURCE_PATHS}
    for f in SOURCE_PATHS:
        target = snapshot/f; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/f, target)
    records, source_records, scenes = [], [], set()
    began = time.time()
    for root in ('round6_region_inputs', 'round6_new_region_inputs'):
        for directory in sorted((ROOT/'results'/root/'val').glob('*/*')):
            manifest_path = directory/'manifest.json'
            if not manifest_path.exists(): continue
            manifest = json.loads(manifest_path.read_text())
            if manifest['scene'] not in SCENES or manifest['split'] != 'val':
                raise ValueError('audit is restricted to the two already inspected validation scenes')
            scenes.add(manifest['scene'])
            for file,key in (('inputs.npz','input_sha256'),('labels.npz','label_sha256')):
                if sha(directory/file) != manifest[key]: raise ValueError('archive hash mismatch')
            with np.load(directory/'inputs.npz', allow_pickle=False) as z:
                d = {k: torch.from_numpy(z[k]) for k in FIELDS}
            with np.load(directory/'labels.npz', allow_pickle=False) as z:
                labels = torch.from_numpy(z['soft_labels']).double()
                valid = torch.from_numpy(z['gt_valid']).bool()
                mass = torch.from_numpy(z['sample_count']).double()*torch.from_numpy(z['visible_fraction']).double()
            n = len(manifest['pairs'])
            if any(len(v) != n for v in list(d.values())+[labels,valid,mass]):
                raise ValueError('manifest/array row count mismatch')
            if not torch.equal(mass>0, valid): raise ValueError('mass/validity disagreement')
            source_records.append(dict(path=str(directory.relative_to(ROOT)), scene=manifest['scene'],
                grid=manifest['grid'], pairs=n, input_sha256=manifest['input_sha256'],
                label_sha256=manifest['label_sha256'], manifest_sha256=sha(manifest_path)))
            for method,width,temperature in CONFIGS:
                comparisons = {name: metric_accumulator() for name in (
                    'lift32_vs_pair32', 'lift64_vs_pair64', 'lift64_vs_reference64',
                    'pair64_vs_reference64', 'lift32_vs_reference64', 'pair32_vs_reference64')}
                model32 = GaussianQuadraticMatcher(method,width,temperature)
                model64 = GaussianQuadraticMatcher(method,width,temperature).double()
                reverse_max = dict(lift32_vs_pair32=0., lift64_vs_reference64=0.)
                for index in range(n):
                    one = {k:v[index:index+1] for k,v in d.items()}
                    double = {k:v.double() for k,v in one.items()}
                    scores = dict(lift32=model32(one), lift64=model64(double),
                        pair32=pair_score(one,method,width,temperature),
                        pair64=pair_score(double,method,width,temperature),
                        reference64=pair_score(double,method,width,temperature,reference=True))
                    for name, acc in comparisons.items():
                        a,r = name.split('_vs_')
                        compare(acc,scores[a],scores[r],labels[index:index+1],mass[index:index+1],valid[index:index+1])
                    # Reverse geometry has no reverse supervision in this archive;
                    # inspect probabilities for the fixed first pair of each file.
                    if index == 0:
                        q,k = model32.encode(one,reverse=True)
                        a = (q@k.transpose(-1,-2)).softmax(-1)
                        b = pair_score(one,method,width,temperature,reverse=True).softmax(-1)
                        reverse_max['lift32_vs_pair32'] = float((a-b).abs().max())
                        q,k = model64.encode(double,reverse=True)
                        a = (q@k.transpose(-1,-2)).softmax(-1)
                        b = pair_score(double,method,width,temperature,reverse=True,reference=True).softmax(-1)
                        reverse_max['lift64_vs_reference64'] = float((a-b).abs().max())
                for acc in comparisons.values():
                    acc['mean_absolute_visible_mass_weighted_ce_difference'] = acc['absolute_weighted_ce_difference_sum']/acc['visible_mass']
                    acc['signed_visible_mass_weighted_ce_difference'] = acc['signed_weighted_ce_difference_sum']/acc['visible_mass']
                    acc['kl_difference_equals_ce_difference'] = True
                with torch.enable_grad():
                    grad = gradient_and_unit_audit(d,method,width,temperature)
                record = dict(scene=manifest['scene'],grid=manifest['grid'],method=method,
                    bandwidth=width,temperature=temperature,pairs=n,query_rows=int(valid.numel()),
                    valid_rows=int(valid.sum()), comparisons=comparisons,
                    reverse_first_pair_probability_audit=reverse_max,first_pair_gradient_unit_audit=grad)
                records.append(record)
                print(manifest['scene'],manifest['grid'],method,width,'pairs',n,
                    'max_prob32_pair',comparisons['lift32_vs_pair32']['max_probability_difference'],flush=True)
    if scenes != SCENES or len(source_records)!=8: raise ValueError('missing scene/grid conditions')
    if hashes != {f:sha(ROOT/f) for f in SOURCE_PATHS}: raise ValueError('source changed during audit')
    summary = dict(scope='two previously inspected validation places only; no held-out test',
        all_pairs_per_scene_grid=True,forward_pair_instances=sum(r['pairs'] for r in source_records),
        distinct_physical_pairs=len({(r['scene'],q,k) for r in source_records
            for q,k in json.loads((ROOT/r['path']/'manifest.json').read_text())['pairs']}),
        configurations=[dict(method=m,bandwidth=w,temperature=t) for m,w,t in CONFIGS],
        source_sha256=hashes,data_sources=source_records,records=records,seconds=time.time()-began,
        denominators='CE/KL differences use identical visible-source-pixel mass; maximum row CE uses GT-valid rows',
        forward_scope='all 64 pairs in each of 8 scene/grid archives; five declared configurations',
        reverse_scope='fixed first pair per scene/grid/configuration; probabilities only',
        gradient_scope='fixed first pair; first 13 tokens per side; float64; no optimizer step',
        argmax_tie_definition='near-tie report: reference top-two logit gap <=1e-4; also explicit per-row numeric score bound',
        novel_operator_claim=False)
    (args.out/'audit.json').write_text(json.dumps(summary,indent=2))
    (args.out/'README.md').write_text(
        '# Round6 quadratic implementation audit\n\n'
        'CPU only. All pairs from the two already inspected validation places; no fresh held-out scene was read. '
        'The actual five-channel model is compared with the actual pair kernel and an independent float64 solve/slogdet reference. '
        'Model/source/data digests and exact denominators are in audit.json. '
        'This verifies implementation equivalence, not novelty, confirmation performance, or independent seed replication.\n')
    print('DONE',len(records),'records',round(time.time()-began,2),'seconds',flush=True)


if __name__ == '__main__': main()
