"""Validation-only scalar calibration, then fresh synthetic test scenes.

Motivation: a common small learning rate need not optimize geometry channels of
different numerical scales equally well. Keep the trained content network fixed,
choose scalar shape weight multipliers using validation only, and test on scenes
not used in either exploratory round. Not a new end-to-end training run.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.matching_probe import Matcher, aggregate, evaluate
from experiments.synthetic import SCENARIOS, make_dataset


METHODS = ['center', 'rope3d_axis', 'rope3d_axis_normal', 'rope3d_axis_scale',
           'rope3d_axis_logcov', 'rope3d_axis_projected_logcov',
           'rope3d_axis_bounded_logcov', 'rope3d_axis_moments', 'rope3d_axis_random_control']


def inverse_softplus(x):
    return torch.log(torch.expm1(torch.tensor(x))) if x > 0 else torch.tensor(-30.)


def main(args):
    torch.set_num_threads(1)
    seeds = [11, 22, 33, 44, 55]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results, selections, hashes = {}, [], {}
    for seed in seeds:
        val = make_dataset(200000+seed, 96)
        tests = {s: make_dataset(900000+seed, 192, scenario=s) for s in SCENARIOS}
        for method in METHODS:
            file = Path(args.checkpoints) / f'{seed}-{method}.pt'
            checkpoint = torch.load(file, map_location='cpu', weights_only=True)
            hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
            model = Matcher(method)
            model.load_state_dict(checkpoint['state_dict'])
            initial = F.softplus(model.shape_weight).item()
            multipliers = [1.] if method in ['center', 'rope3d_axis'] else [0., .25, .5, 1., 2., 4., 8., 16.]
            best, chosen = -1, None
            candidates = []
            for multiplier in multipliers:
                with torch.no_grad():
                    model.shape_weight.copy_(inverse_softplus(initial * multiplier))
                val_score = float(evaluate(model, val)['r1'].mean())
                candidates.append({'multiplier': multiplier, 'val_r1': val_score})
                if val_score > best:
                    best, chosen = val_score, multiplier
            with torch.no_grad():
                model.shape_weight.copy_(inverse_softplus(initial * chosen))
            for scenario, data in tests.items():
                scores = evaluate(model, data)
                for metric, values in scores.items():
                    results[f'{seed}/{method}/{scenario}/{metric}'] = values
            record = {'seed': seed, 'method': method, 'chosen_multiplier': chosen,
                      'effective_shape_weight': initial*chosen, 'val_r1': best, 'grid': candidates}
            selections.append(record)
            print(json.dumps({k:v for k,v in record.items() if k!='grid'}), flush=True)
    rows = aggregate(results, seeds, METHODS, SCENARIOS)
    np.savez_compressed(out/'per_scene.npz', **results)
    (out/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    with (out/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    manifest = {'test_seed_rule': '900000 + training_seed', 'validation_seed_rule': '200000 + training_seed',
                'scenes_per_seed_per_scenario': 192, 'seeds': seeds,
                'grid': [0., .25, .5, 1., 2., 4., 8., 16.], 'selection': selections,
                'checkpoint_hashes': hashes, 'torch': torch.__version__,
                'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for folder in ['models','experiments'] for p in sorted(Path(folder).glob('*.py'))}}
    for filename in manifest['source_hashes']:
        target = out/'source'/filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(filename).read_bytes())
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('COMPLETE', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', default='results/revision/checkpoints')
    parser.add_argument('--out', default='results/fresh_calibrated')
    main(parser.parse_args())
