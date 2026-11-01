"""Development-only V-path audit; no training, selection or test-data access."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F

from experiments.round5_probability_audit import probability_metrics
from models.round5_cross_attention import Round5CrossAttentionMatcher


FIELDS = ('q', 'k', 'qmean', 'qcov', 'kmean', 'kcov', 'k_to_q', 'k_to_qcov', 'q_to_k', 'q_to_kcov')
LABELS = ('gt_valid', 'pixcontinuous', 'key_indices', 'key_weights')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_development(config, split, count):
    if split not in ('train', 'val'):
        raise ValueError('this diagnostic cannot open the test split')
    source = config[split+'_sources'][0]
    directory = Path(config['arguments']['data'])/split/source['scene'].replace('/', '__')
    for name, key in [('inputs.npz', 'input_sha256'), ('labels.npz', 'label_sha256')]:
        if sha(directory/name) != source[key]:
            raise ValueError('development cache changed since checkpoint training')
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['split'] != split or manifest['scene'] != source['scene']:
        raise ValueError('development scene/split mismatch')
    with np.load(directory/'inputs.npz', allow_pickle=False) as inputs:
        data = {key:torch.from_numpy(inputs[key][:count]).float() for key in FIELDS}
    with np.load(directory/'labels.npz', allow_pickle=False) as labels:
        supervision = {key:labels[key][:count] for key in LABELS}
    identities = [dict(scene=source['scene'], query=q, key=k) for q, k in manifest['pairs'][:count]]
    return data, supervision, identities, dict(source,
        manifest_sha256=sha(directory/'manifest.json'), selected_pair_indices=list(range(count)))


def gradients(model, data, labels, zero_values):
    model.zero_grad(set_to_none=True)
    logits = model(data, zero_values=zero_values)
    valid = torch.from_numpy(labels['gt_valid']).bool()
    indices = torch.from_numpy(labels['key_indices']).long()
    weights = torch.from_numpy(labels['key_weights']).float()
    ce = -(F.log_softmax(logits, dim=-1).gather(-1, indices.clamp_min(0))*weights).sum(-1)
    if not valid.any():
        raise ValueError('development batch has no valid supervision')
    loss = ce[valid].mean()
    loss.backward()
    records = []
    for index, layer in enumerate(model.layers):
        value = layer.qkv.weight.grad
        output = layer.output.weight.grad
        if not torch.isfinite(value).all() or not torch.isfinite(output).all():
            raise ValueError('nonfinite development gradient')
        records.append(dict(layer=index, q_projection_gradient_norm=float(value[:96].norm()),
            k_projection_gradient_norm=float(value[96:192].norm()),
            v_projection_gradient_norm=float(value[192:].norm()),
            output_projection_gradient_norm=float(output.norm())))
    return dict(loss=float(loss.detach()), valid_rows=int(valid.sum()), layers=records)


def main():
    torch.set_num_threads(1)
    run = ROOT/'results/round5_full_dev/normalized_101'
    out = ROOT/'results/round5_dev_path_audit'
    if out.exists() and any(out.iterdir()):
        raise ValueError('refusing overwrite development path audit')
    config = json.loads((run/'config.json').read_text())
    completed = json.loads((run/'completed.json').read_text())
    checkpoint = run/'checkpoints/best.pt'
    checkpoint_hash = sha(checkpoint)
    if checkpoint_hash != completed['checkpoint_sha256']:
        raise ValueError('development checkpoint hash mismatch')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload['method'] != 'normalized' or payload['seed'] != 101:
        raise ValueError('unexpected development checkpoint identity')
    model_sources = ['models/round5_cross_attention.py', 'models/gaussian_pe.py',
        'models/gaussian_spectral_pe.py', 'models/geometry.py']
    for path in model_sources:
        if sha(ROOT/path) != payload['sourcehash'][path]:
            raise ValueError('model implementation changed since development checkpoint')
    model = Round5CrossAttentionMatcher('normalized').eval()
    model.load_state_dict(payload['model'], strict=True)
    before = {key:value.clone() for key, value in model.state_dict().items()}
    validation, val_labels, identities, val_source = load_development(config, 'val', 10)
    rows, maximum_logit_difference, maximum_probability_difference = [], 0., 0.
    with torch.no_grad():
        for index, identity in enumerate(identities):
            data = {key:value[index:index+1] for key, value in validation.items()}
            ordinary = model(data)
            zero = model(data, zero_values=True)
            probability, zero_probability = ordinary.softmax(-1)[0].numpy(), zero.softmax(-1)[0].numpy()
            maximum_logit_difference = max(maximum_logit_difference, float((ordinary-zero).abs().max()))
            maximum_probability_difference = max(maximum_probability_difference,
                float(np.abs(probability-zero_probability).max()))
            supervision = {key:value[index] for key, value in val_labels.items()}
            rows.append(dict(identity,
                ordinary=probability_metrics(probability=probability, **supervision),
                zero_values=probability_metrics(probability=zero_probability, **supervision)))
    train, train_labels, _, train_source = load_development(config, 'train', 8)
    ordinary_gradient = gradients(model, train, train_labels, zero_values=False)
    disabled_gradient = gradients(model, train, train_labels, zero_values=True)
    if any(r['v_projection_gradient_norm'] <= 1e-10 or r['output_projection_gradient_norm'] <= 1e-10
           for r in ordinary_gradient['layers']):
        raise ValueError('ordinary trained checkpoint has no measured V/output gradient')
    if any(r['v_projection_gradient_norm'] != 0 or r['output_projection_gradient_norm'] != 0
           for r in disabled_gradient['layers']):
        raise ValueError('zero_values did not remove the V/output path')
    if maximum_logit_difference <= 1e-6:
        raise ValueError('disabling values did not change validation logits')
    if any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items()):
        raise ValueError('gradient diagnostic unexpectedly changed model state')
    if sha(checkpoint) != checkpoint_hash:
        raise ValueError('checkpoint changed during diagnostic')
    totals = {}
    for mode in ('ordinary', 'zero_values'):
        n = sum(row[mode]['n_valid'] for row in rows)
        totals[mode] = dict(n_valid=n, **{key:sum(row[mode][key] for row in rows)/n
            for key in ('ce_sum', 'pck8_count', 'pck4_count', 'epe_argmax_sum', 'epe_soft_sum')})
    result = dict(scope='Development-only path diagnostic after confirmation architecture/budget freeze; not model selection or formal test evidence.',
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        development_seed=101, selected_step=payload['step'], validation_pairs=10, train_batch_pairs=8,
        sources={'train':train_source, 'validation':val_source}, rows=rows,
        pooled_ten_pair_metrics=totals, ordinary_train_gradient=ordinary_gradient,
        zero_value_train_gradient=disabled_gradient,
        maximum_validation_logit_absolute_change=maximum_logit_difference,
        maximum_validation_probability_absolute_change=maximum_probability_difference,
        optimizer_steps=0, all_model_state_unchanged=True,
        limitations='Removing V removes complete attention updates, not only covariance effects; this is an architecture path check, not a retrained ablation or an attribution of accuracy to PE.',
        source_sha256={path:sha(ROOT/path) for path in model_sources},
        diagnostic_source_sha256=sha(__file__))
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result, indent=2))
    (out/'README.md').write_text('# Development-only V-path audit\n\n'
        'Uses the frozen normalized development checkpoint (seed101), first10 validation pairs and first8 train pairs only. '
        'No formal test data, optimizer update, architecture selection or confirmation model is used.\n\n'
        f'Maximum validation logit/probability change when zeroing V: {maximum_logit_difference:.6g} / {maximum_probability_difference:.6g}. '
        f'Full V-gradient norms by layer: {[r["v_projection_gradient_norm"] for r in ordinary_gradient["layers"]]}; '
        'disabled-path V/output gradients are exactly zero. All model parameters/buffers and checkpoint bytes remain unchanged.\n\n'
        f'Pooled10-pair descriptive metrics: {totals}. These quantify path activity, not generalization or PE-specific attribution.\n')
    print(json.dumps(dict(validation=totals, gradients=ordinary_gradient['layers'],
        maximum_logit_change=maximum_logit_difference)))


if __name__ == '__main__':
    main()
