"""Read-only attention decomposition on mechanically selected old fit objects."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from experiments.round65_common import ROOT, read, write, sha
from experiments.round65_data import load
from experiments.round65_run import configure
from experiments.geometry_scale_diagnostic import rng
from experiments.round73_native_relative5 import local_dependencies
from experiments.native_shape_attribution import check as check_parent
from models.round65_actual_gs import NativeGSClassifier
from models.round73_native_relative5 import CompactNativeClassifier
from models.native_shape_attribution import NativeFixedIsotropic
from models.round58_support_pe import HEADS, HD

OUT = ROOT / 'results/native_attention_balance_diagnostic/attempt1'
ROOTS = dict(relative5=ROOT / 'results/round73_native_relative5/attempt1',
             distance_mlp=ROOT / 'results/round65_actual_gs/attempt1',
             fixed_isotropic5=ROOT / 'results/native_shape_attribution/attempt1')
SEEDS = (11, 22, 33)
METRICS = ('content_rms', 'position_rms', 'position_to_content_rms', 'total_rms',
           'content_position_correlation', 'total_entropy_fraction', 'content_entropy_fraction',
           'attention_tv_change', 'attention_top1_change', 'total_max_probability',
           'total_probability_over_half_fraction', 'total_self_probability')


def prepare():
    check_parent()
    assert not (OUT / 'plan.json').exists()
    sources = {str(p): sha(p) for p in local_dependencies([Path(__file__)])}
    manifest_path = ROOTS['distance_mlp'] / 'data/manifest.json'
    manifest = read(manifest_path)
    fit = next(r for r in manifest['partitions'] if r['split'] == 'fit')
    labels_path = ROOTS['distance_mlp'] / 'data/fit/labels.npy'
    assert sha(labels_path) == fit['hashes']['labels.npy']
    labels = np.load(labels_path, allow_pickle=False)
    selected_objects = []
    for category in range(60):
        indices = sorted(np.flatnonzero(labels == category).tolist(), key=lambda i: fit['ids'][i])[:2]
        assert len(indices) == 2
        selected_objects.extend(dict(index=i, identity=fit['ids'][i], category=category) for i in indices)
    previous = read(ROOT / 'results/geometry_scale_diagnostic/attempt1/result.json')
    assert [r['identity'] for r in selected_objects] == [r['identity'] for r in previous['contexts'] if r['kind'] == 'native']
    sources[str(manifest_path)] = sha(manifest_path)
    checkpoints = []
    for role, root in ROOTS.items():
        selected = read(root / 'selected.json')
        sources[str(root / 'selected.json')] = sha(root / 'selected.json')
        for seed in SEEDS:
            row = next(r for r in selected['records'] if (r['role'], r['seed']) == (role, seed))
            path = root / 'runs' / role / str(seed) / 'best.pt'
            assert sha(path) == row['checkpoint_sha256']
            checkpoints.append(dict(role=role, seed=seed, path=str(path),
                sha256=row['checkpoint_sha256'], selected_epoch=row['best_epoch']))
    write(OUT / 'plan.json', dict(created_unix=time.time(),
        hypothesis='Gross dominance or saturation of the learned position term may distinguish the previously weaker relative5 seed 22 and help explain training-seed sensitivity.',
        competing_explanations=[
            'The position MLP adapts large raw descriptor ranges without gross attention imbalance.',
            'Attention specialization or complementary geometry, rather than harmful dominance, explains large position effects.',
            'Content representations, finite data and optimization differences remain possible sources of seed variation.'],
        fixed_comparison='Existing relative5, Distance and equal-capacity fixed-isotropic models, old seeds 11/22/33 only, same 120 old fit objects and 512 primitive indices from the previous geometry-scale diagnostic; all four layers and both heads.',
        input='Two lexically first old fit objects/class, fixed geometry-scale-fit-only-v1 token sampling, original cached coordinates/covariance/appearance, no new augmentation or labels for scoring.',
        selected_objects=selected_objects, checkpoints=checkpoints, source_sha256=sources,
        metrics=list(METRICS),
        decomposition='Recompute original Q/K and PE score in observation-only pre-hooks. Remove each query row mean before score RMS/correlation, since constant row shifts do not affect softmax. Compare total attention to content-only softmax at the SAME frozen layer input; this is a local counterfactual, not a model ablation score.',
        decisions=dict(distinctive_imbalance='Consider a separately planned old-data position/content balance intervention only after the fixed replication; no automatic normalization or new gate.',
                       no_distinctive_imbalance='Do not prioritize gross PE domination as a unique explanation for the weak seed; retain optimization/task-semantic alternatives.'),
        scope=['No training, optimizer update, temperature scan, model selection, val/dev_eval or official TEST inputs.',
               'Fit-object internal measurements are descriptive; no generalization or causal failure claim.',
               'All seeds and layers are retained; large attention changes are not inherently harmful.',
               'No change to the running new-five replication or its original decision.'],
        resource='One whole-idle GPU outside active training cards; original precision/determinism, inference only, batch4, allocator cap4 GiB.',
        prior_goal_turn_classification='progress: authorized two-card continuation launched and both fits verified advancing.',
        model_changed=False, official_test_access=False, fresh_confirmation=False, goal_complete=False))
    (OUT / 'plan.sha256').write_text(sha(OUT / 'plan.json') + '\n')
    print('ATTENTION_BALANCE_PLAN_PREPARED', sha(OUT / 'plan.json'), flush=True)


def checked():
    plan = read(OUT / 'plan.json')
    assert sha(OUT / 'plan.json') == (OUT / 'plan.sha256').read_text().strip()
    for name, digest in plan['source_sha256'].items():
        assert sha(name) == digest, name
    for row in plan['checkpoints']:
        assert sha(row['path']) == row['sha256']
    return plan


def digest_state(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def observed_statistics(block, inputs):
    x, relative = inputs[:2]
    b, n, _ = x.shape
    assert relative is not None and n == 512
    q, k, _ = block.qkv(block.norm(x)).reshape(b, n, 3, HEADS, HD).permute(2, 0, 3, 1, 4).unbind(0)
    content_raw = q @ k.transpose(-1, -2)
    position_raw = torch.einsum('bhid,bijhd->bhij', q, relative.reshape(b, n, n, HEADS, HD))
    total = (content_raw + position_raw) / math.sqrt(HD)
    content, position = content_raw / math.sqrt(HD), position_raw / math.sqrt(HD)
    content = (content - content.mean(-1, keepdim=True)).double()
    position = (position - position.mean(-1, keepdim=True)).double()
    total_centered = (total - total.mean(-1, keepdim=True)).double()
    cr = content.square().mean((-1, -2)).sqrt()
    pr = position.square().mean((-1, -2)).sqrt()
    assert bool((cr > 0).all()) and bool((pr > 0).all())
    probability = total.softmax(-1).double()
    content_probability = (content_raw / math.sqrt(HD)).softmax(-1).double()
    entropy = lambda p: -(p * p.clamp_min(1e-30).log()).sum(-1).mean(-1) / math.log(n)
    maximum = probability.max(-1).values
    values = dict(content_rms=cr, position_rms=pr, position_to_content_rms=pr / cr,
        total_rms=total_centered.square().mean((-1, -2)).sqrt(),
        content_position_correlation=(content * position).mean((-1, -2)) / (cr * pr),
        total_entropy_fraction=entropy(probability), content_entropy_fraction=entropy(content_probability),
        attention_tv_change=.5 * (probability - content_probability).abs().sum(-1).mean(-1),
        attention_top1_change=(probability.argmax(-1) != content_probability.argmax(-1)).double().mean(-1),
        total_max_probability=maximum.mean(-1),
        total_probability_over_half_fraction=(maximum > .5).double().mean(-1),
        total_self_probability=probability.diagonal(dim1=-2, dim2=-1).mean(-1))
    assert set(values) == set(METRICS)
    return {name: value.cpu().numpy() for name, value in values.items()}


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    assert np.isfinite(values).all()
    return dict(count=values.size, mean=float(values.mean()),
        p05=float(np.quantile(values, .05)), median=float(np.median(values)), p95=float(np.quantile(values, .95)),
        minimum=float(values.min()), maximum=float(values.max()))


@torch.no_grad()
def run():
    plan = checked()
    assert not (OUT / 'result.json').exists() and not (OUT / 'execution.json').exists()
    visible = os.environ['CUDA_VISIBLE_DEVICES']
    before = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    row = next([v.strip() for v in line.split(',')] for line in before.splitlines() if visible in line)
    assert int(row[2]) <= 512 and int(row[3]) >= 16 * 1024 and int(row[4]) <= 10
    configure(); assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(4 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    probe = torch.ones(4, device='cuda'); assert float(probe.sum()) == 4; del probe
    write(OUT / 'execution.json', dict(pid=os.getpid(), started_unix=time.time(),
        gpu_uuid=visible, gpu_observation=before, plan_sha256=sha(OUT / 'plan.json')))
    bundle = load('fit', ROOTS['distance_mlp'])
    inputs = {name: [] for name in ('centers', 'covariance', 'appearance')}
    identities = []
    for obj in plan['selected_objects']:
        i = obj['index']; assert str(bundle['ids'][i]) == obj['identity']
        token_ids = rng(obj['identity'] + ':tokens').choice(bundle['centers'].shape[1], 512, replace=False)
        digest = hashlib.sha256(token_ids.tobytes())
        for name in inputs:
            value = np.asarray(bundle[name][i, token_ids]); inputs[name].append(value); digest.update(value.tobytes())
        identities.append(dict(**obj, sampled_input_sha256=digest.hexdigest()))
    inputs = {name: np.stack(values) for name, values in inputs.items()}
    summaries, checks, raw = [], [], {}
    for checkpoint in plan['checkpoints']:
        role, seed = checkpoint['role'], checkpoint['seed']
        factory = dict(relative5=CompactNativeClassifier, distance_mlp=NativeGSClassifier, fixed_isotropic5=NativeFixedIsotropic)[role]
        model = factory(role).cuda().eval()
        model.load_state_dict(torch.load(checkpoint['path'], map_location='cpu', weights_only=True)['state_dict'])
        before_state = digest_state(model)
        collected = {layer: {name: [] for name in METRICS} for layer in range(4)}
        def observe(layer):
            def hook(block, args):
                values = observed_statistics(block, args)
                for name, value in values.items(): collected[layer][name].append(value)
            return hook
        first = [torch.tensor(inputs[name][:4], device='cuda') for name in inputs]
        original_logits = model(*first)
        handles = [layer.register_forward_pre_hook(observe(i)) for i, layer in enumerate(model.layers)]
        observed_logits = model(*first)
        error = float((original_logits - observed_logits).abs().max())
        assert torch.allclose(original_logits, observed_logits, atol=2e-5, rtol=2e-5)
        for start in range(4, len(identities), 4):
            values = [torch.tensor(inputs[name][start:start+4], device='cuda') for name in inputs]
            logits = model(*values); assert bool(torch.isfinite(logits).all())
        for handle in handles: handle.remove()
        assert digest_state(model) == before_state
        layers = []
        for layer in range(4):
            arrays = {name: np.concatenate(values) for name, values in collected[layer].items()}
            assert all(v.shape == (120, HEADS) for v in arrays.values())
            for name, value in arrays.items(): raw[f'{role}__{seed}__{layer}__{name}'] = value
            for head in range(HEADS):
                layers.append(dict(layer=layer, head=head, metrics={name: summary(v[:, head]) for name, v in arrays.items()}))
        pooled = {name: summary(np.concatenate([raw[f'{role}__{seed}__{layer}__{name}'].ravel() for layer in range(4)])) for name in METRICS}
        summaries.append(dict(role=role, seed=seed, selected_epoch=checkpoint['selected_epoch'], layers=layers, pooled=pooled))
        checks.append(dict(role=role, seed=seed, observation_hook_max_logit_error=error, model_state_unchanged=True))
        print('DIAGNOSTIC_COMPLETE', role, seed, 'ratio_median', pooled['position_to_content_rms']['median'], 'entropy', pooled['total_entropy_fraction']['median'], flush=True)
        del model, first, original_logits, observed_logits, values, logits
        torch.cuda.empty_cache()
    np.savez_compressed(OUT / 'per_object_statistics.npz', **raw)
    result = dict(created_unix=time.time(), plan_sha256=sha(OUT / 'plan.json'),
        summaries=summaries, checks=checks, selected_inputs=identities,
        per_object_statistics_sha256=sha(OUT / 'per_object_statistics.npz'),
        original_fit_manifest_sha256=sha(ROOTS['distance_mlp'] / 'data/manifest.json'),
        total_models=9, independent_fit_objects=120, layers=4, heads=HEADS,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        scope=plan['scope'], new_fits=0, model_changed=False, fresh_confirmation=False,
        official_test_access=False, goal_complete=False, interpretation_pending=True)
    write(OUT / 'result.json', result)
    plot(result)
    print('ATTENTION_BALANCE_DIAGNOSTIC_FINISHED', flush=True)


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [('position_to_content_rms', 'Position/content row-centered RMS (median)', True),
             ('total_entropy_fraction', 'Total attention entropy / log(512) (median)', False),
             ('attention_tv_change', 'Attention TV change from content only (median)', False)]
    labels = [f"{r['role']} / {r['seed']}" for r in result['summaries']]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    for ax, (name, title, logarithmic) in zip(axes, names):
        matrix = np.asarray([[r['metrics'][name]['median'] for r in row['layers']] for row in result['summaries']])
        display = np.log10(matrix) if logarithmic else matrix
        im = ax.imshow(display, aspect='auto', cmap='viridis', vmin=None if logarithmic else 0, vmax=None if logarithmic else 1)
        for i in range(len(labels)):
            for j in range(8): ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=6, color='white' if display[i,j] < (display.max()+display.min())/2 else 'black')
        ax.set(yticks=range(len(labels)), yticklabels=labels, xticks=range(8), xticklabels=[f'L{i+1}H{j+1}' for i in range(4) for j in range(2)], title=title)
        ax.tick_params(axis='x', labelrotation=45)
        fig.colorbar(im, ax=ax, label='log10 ratio' if logarithmic else 'fraction')
    fig.savefig(OUT / 'attention_balance.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('prepare', 'run'))
    args = parser.parse_args(); globals()[args.action]()
