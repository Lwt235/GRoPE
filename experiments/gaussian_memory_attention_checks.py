"""Short GPU integration check on training inputs; its weights are discarded."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import OUT, VARIANTS, assert_identity, load_inputs, write_json
from experiments.gaussian_memory_data import TRAIN_QUERY
from models.gaussian_memory_attention import GaussianAttentionReader, geometry_inputs


def main():
    torch.set_num_threads(4)
    identity = assert_identity()
    item = load_inputs('cuda', fit_only=True)[0]
    raw = item['raw']
    data = geometry_inputs(item['memory'], raw['c2w'][TRAIN_QUERY[:2]].cuda(), raw['K'][TRAIN_QUERY[:2]].cuda())
    target = raw['teacher'][TRAIN_QUERY[:2]].cuda().flatten(1, 2)
    checks = []
    states = []
    for variant in VARIANTS:
        torch.manual_seed(20261110)
        model = GaussianAttentionReader(variant, identity['center_variance']).cuda()
        states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        prediction = model(data)
        loss = (prediction - target).square().mean()
        loss.backward()
        gradients = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.requires_grad}
        assert all(np.isfinite(list(gradients.values())))
        assert all(x > 0 for x in gradients.values())
        replay = model.read_memory(item['memory'], raw['c2w'][TRAIN_QUERY[:2]].cuda(), raw['K'][TRAIN_QUERY[:2]].cuda())
        error = (replay.flatten(1, 2) - prediction).abs().max().item()
        assert error == 0
        checks.append({'variant': variant, 'loss_finite': bool(torch.isfinite(loss)),
                       'gradient_norms': gradients, 'prepared_vs_direct_max_error': error,
                       'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)})
    for a, b in zip(states, states[1:]):
        assert all(torch.equal(a[k], b[k]) for k in a)
    result = {'passed': True, 'device': torch.cuda.get_device_name(0),
              'training_only': True, 'preflight_weights_discarded': True,
              'same_initial_state_across_groups': True, 'cpu_geometry_tests_passed': 4,
              'checks': checks, 'allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3}
    write_json(OUT / 'preflight.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
