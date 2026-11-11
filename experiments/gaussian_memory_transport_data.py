"""Read-only reuse of source data; fixed states and matched reader initializations."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_oracle_data import OUT as PREVIOUS, SEEDS, checked as old_checked, load_record, input_seeds
from models.gaussian_memory import detached_memory
from models.gaussian_memory_transport import GeometryTransport, frozen_memory

OUT = ROOT / 'results/gaussian_memory/geometry_transport1'
REGIMES = ('predicted', 'sensor')
KERNELS = ('center', 'gaussian')
READERS = ('latent', 'transport', 'prior_residual')


def cases():
    return [dict(regime='none', kernel='center', reader='local', seed=s) for s in SEEDS] + [
        dict(regime=g, kernel=k, reader=r, seed=s)
        for s in SEEDS for g in REGIMES for k in KERNELS for r in READERS]


def name(case):
    return '{regime}_{kernel}_{reader}_{seed}'.format(**case)


def model_path(case, initial=False):
    return OUT / name(case) / ('initial.pt' if initial else 'step_2000.pt')


def state_path(rid, regime):
    return OUT / 'memories' / f'{rid}_{regime}.pt'


def make_model(case, plan):
    return GeometryTransport(case['reader'], case['kernel'], plan['center_variance'], plan['local_width'])


def load_memory(rid, regime, device='cpu'):
    return torch.load(state_path(rid, regime), map_location=device, weights_only=False)['memory']


def prepare():
    old = old_checked(True); OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'plan.json').exists()
    (OUT/'memories').mkdir(); identities = dict(old['input_sha256']); memories = {}
    for meta in old['catalog']:
        d = load_record(meta)
        for regime in REGIMES:
            memory = detached_memory(frozen_memory(input_seeds(d, regime, 'cpu')))
            path = state_path(meta['id'], regime)
            torch.save({'memory': memory, 'unit': d['raw']['diagnostics']['unit_meters'],
                        'reference_frames': [0,4,8,12], 'query_RGB_used': False, 'query_depth_used': False,
                        'regime': regime, 'reference_sensor_used': regime == 'sensor',
                        'raw_sha256': sha256(meta['raw'])}, path)
            identities[str(path)] = sha256(path); memories[str(path)] = sha256(path)
    source = dict(old['source_sha256'])
    new_sources = list(ROOT.glob('experiments/gaussian_memory_transport_*.py')) + [ROOT/'models/gaussian_memory_transport.py']
    source.update({str(p.relative_to(ROOT)): sha256(p) for p in new_sources})
    plan = {**{k:old[k] for k in ('catalog','training_ids','primary_places','steps','batch','lr','weight_decay','center_variance','gradient_clip')},
            'local_width': 67, 'old_local_width': old['local_width'], 'seeds': SEEDS, 'cases': cases(),
            'input_sha256': identities, 'source_sha256': source,
            'document_sha256': sha256(ROOT/'docs/gaussian_memory_geometry_transport_plan.md'),
            'previous_plan_sha256': sha256(PREVIOUS/'plan.json'), 'confirmation_claim': False,
            'memory_frozen': True, 'parameters': {}}
    for c in cases():
        # A reader family shares all parameters across sources and position biases.
        torch.manual_seed(c['seed']); net = make_model(c, plan)
        count = sum(p.numel() for p in net.parameters()); plan['parameters'][c['reader']] = count
        directory = model_path(c).parent; assert not directory.exists(); directory.mkdir()
        config = {**c, 'parameter_count': count, 'memory_frozen': True, 'steps': 2000}
        torch.save({'model': net.state_dict(), 'config': config}, model_path(c, True))
        write_json(directory/'config.json', config)
    assert plan['parameters']['latent'] == plan['parameters']['transport']
    assert plan['parameters']['local'] == plan['parameters']['prior_residual']
    write_json(OUT/'plan.json', plan); write_json(OUT/'memory_identity.json', {'sha256': memories, 'states':len(memories)})
    print(json.dumps({'prepared':True,'states':len(memories),'fits':len(cases()),'parameters':plan['parameters']}),flush=True)


def checked(inputs=False):
    plan = json.loads((OUT/'plan.json').read_text())
    assert sha256(ROOT/'docs/gaussian_memory_geometry_transport_plan.md') == plan['document_sha256']
    for p,digest in plan['source_sha256'].items():assert sha256(ROOT/p)==digest,p
    if inputs:
        for p,digest in plan['input_sha256'].items():assert sha256(p)==digest,p
    return plan


if __name__ == '__main__':prepare()
