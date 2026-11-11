"""One FP32 native-tail check of the nonzero BF16 joint-permutation difference."""
import argparse
import json
import sys
import time

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, DATA, COMMIT, sha256
from experiments.gaussian_memory_bound_data import OUT, BASE_CACHE, CACHE, SEEDS, load_plan, load_priors, device_tree, write_json
from experiments.gaussian_memory_bound_evaluate import load_adapter, content_permutation
from experiments.gaussian_memory_incremental_evaluate import score
from experiments.gaussian_memory_view_data import load_backbone
from models.gaussian_memory_bound_read import bound_native_tail, attention_prior


@torch.no_grad()
def main(device):
    torch.set_num_threads(4)
    plan = load_plan(); rid = plan['training_ids'][0]; seed = SEEDS[0]
    basename = f'precision_diagnostic_{device}'
    protocol = {
        'sample': [rid, 5, 0], 'seed': seed, 'dtype': 'FP32 tail; original cached prefix values unchanged',
        'device': device,
        'reason': 'The fixed BF16 joint-permutation check was nonzero; distinguish finite-precision sensitivity from broken slot identity.',
        'comparisons': ['intact', 'joint_content_and_prior_permutation', 'content_only_permutation'],
        'scope': 'One existing training sample; implementation diagnosis only, no efficacy threshold or added scientific gate.',
        'no_retraining_or_parameter_change': True,
    }
    target = OUT / f'{basename}_protocol.json'
    if target.exists():
        assert json.loads(target.read_text()) == protocol
    else:
        write_json(target, protocol)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    identity_path = BASE_CACHE / rid / 'identity.json'; identity = json.loads(identity_path.read_text())
    for name in ('initial.pt', 'q5_c0.pt'):
        assert sha256(BASE_CACHE / rid / name) == identity['sha256'][name]
    sample_cpu = torch.load(BASE_CACHE / rid / 'q5_c0.pt', map_location='cpu', weights_only=False)
    sample = device_tree(sample_cpu, device)
    # Cached mixed-precision values are retained, but subsequent operations use FP32.
    for key in ('prefix', 'early4'):
        sample[key] = sample[key].float()
    initial = device_tree(torch.load(BASE_CACHE / rid / 'initial.pt', map_location='cpu', weights_only=False), device)
    content = initial['memory']['features'].float()
    prior = attention_prior(load_priors([rid], device=device)[rid, 5], 'bound')
    if device == 'cuda':
        adapter, checkpoint_hash = load_adapter('bound', seed); backbone = load_backbone()
    else:
        from safetensors.torch import load_file
        from models.gaussian_memory_bound_read import BoundMemoryAdapter
        from experiments.gaussian_memory_bound_train import source_identity
        sys.path.insert(0, str(ROOT / 'third_party/vggt'))
        from vggt.models.vggt import VGGT
        assert (ROOT / 'third_party/vggt/PINNED_COMMIT').read_text().strip() == COMMIT
        backbone = VGGT(enable_point=False, enable_track=False).eval().requires_grad_(False)
        state = load_file(str(DATA / 'model.safetensors'))
        mismatch = backbone.load_state_dict(state, strict=False)
        assert not mismatch.missing_keys and all(k.startswith(('point_head.', 'track_head.')) for k in mismatch.unexpected_keys)
        del state
        checkpoint = OUT / f'bound_{seed}/step_2000.pt'; checkpoint_hash = sha256(checkpoint)
        assert read_status_hash(seed) == checkpoint_hash
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['source_sha256'] == source_identity()
        adapter = BoundMemoryAdapter().eval().requires_grad_(False)
        adapter.load_state_dict(state['model']); del state
    permutation = content_permutation(rid, len(content), content.device)
    columns = torch.cat((permutation, permutation.new_tensor([len(content)])))
    predictions = {}; elapsed = {}
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()
    for name, values, weights in [('intact', content, prior),
                                 ('joint_content_and_prior_permutation', content[permutation], prior[..., columns]),
                                 ('content_only_permutation', content[permutation], prior)]:
        if device == 'cuda': torch.cuda.synchronize()
        start = time.monotonic()
        with torch.autocast('cuda', enabled=False):
            depth, _ = bound_native_tail(backbone, sample, adapter, values, weights)
        if device == 'cuda': torch.cuda.synchronize()
        elapsed[name] = time.monotonic() - start
        predictions[name] = depth.float().cpu()
        assert torch.isfinite(predictions[name]).all() and (predictions[name] > 0).all()
        print(json.dumps({'fp32_case_complete': name, 'device': device, 'seconds': elapsed[name]}), flush=True)
    intact = predictions['intact']
    differences = {name: {'max_abs': float((value - intact).abs().max()),
                          'mean_abs': float((value - intact).abs().mean())}
                   for name, value in predictions.items() if name != 'intact'}
    result = {'complete': True, 'protocol': protocol, 'source_sha256': sha256(__file__),
              'checkpoint_sha256': checkpoint_hash, 'cache_identity_sha256': sha256(identity_path),
              'prior_identity_sha256': sha256(CACHE / rid / 'identity.json'),
              'bf16_reference': json.loads((OUT / 'binding_interface.json').read_text()),
              'differences': differences, 'prediction_mean': float(intact.mean()),
              'scores': {name: score(value, sample_cpu) for name, value in predictions.items()},
              'seconds': elapsed, 'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3 if device == 'cuda' else None,
              'claim_limit': 'Single cached sample, FP32 tail only; not an all-data precision study. CPU changes both dtype and backend versus BF16 GPU, so it cannot isolate dtype alone.'}
    write_json(OUT / f'{basename}.json', result)
    print(json.dumps({'precision_diagnostic_complete': True, 'differences': differences}), flush=True)


def read_status_hash(seed):
    return json.loads((OUT / f'bound_{seed}/status.json').read_text())['checkpoint_sha256']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    main(parser.parse_args().device)
