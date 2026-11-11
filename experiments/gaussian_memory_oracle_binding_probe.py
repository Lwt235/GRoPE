"""Pre-run recovery of the original reference-only compression memberships."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_data import merge_seeds
from experiments.gaussian_memory_attention_train import write_json


def main():
    torch.set_num_threads(4)
    old = ROOT / 'results/gaussian_memory/surface1/plan.json'
    plan = json.loads(old.read_text()); out = ROOT / 'results/gaussian_memory/reference_geometry1'
    dest = out / 'bindings'; dest.mkdir(parents=True, exist_ok=True); rows = []
    for meta in plan['catalog']:
        raw = torch.load(meta['raw'], map_location='cpu', weights_only=False)
        direct = torch.load(meta['direct'], map_location='cuda', weights_only=False)['memory']
        mean, chol, near, weights = merge_seeds(direct['means'], direct['covars'], direct['opacities'] > 0)
        error = float((mean.cpu()-raw['seed_mean']).abs().max())
        chol_error = float((chol.cpu()-raw['seed_chol']).abs().max())
        row = {'id': meta['id'], 'mean_max_diff': error, 'chol_max_diff': chol_error}
        rows.append(row)
        torch.save({'near': near.cpu(), 'weights': weights.cpu(), 'reconstructed_mean': mean.cpu(),
                    'reconstructed_chol': chol.cpu(), 'diagnostics': row, 'direct_sha256': sha256(meta['direct']),
                    'raw_sha256': sha256(meta['raw'])}, dest / (meta['id']+'.pt'))
    write_json(out / 'binding_probe.json', {'rows': rows, 'scientific_result': False,
               'old_plan_sha256': sha256(old), 'source_sha256': sha256(__file__)})
    print(json.dumps({'max_mean_diff': max(r['mean_max_diff'] for r in rows),
                      'max_chol_diff': max(r['chol_max_diff'] for r in rows),
                      'nonzero_rows': [r for r in rows if r['mean_max_diff'] > 1e-5]}), flush=True)


if __name__ == '__main__': main()
