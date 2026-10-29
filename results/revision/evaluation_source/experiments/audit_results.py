"""Re-evaluate existing checkpoints after the spectral-log numerical fix.

No retraining or checkpoint selection. Preserve old metrics and training source
snapshots, and record the post-training evaluation source separately.
"""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from experiments.matching_probe import Matcher, aggregate, evaluate
from experiments.synthetic import make_dataset


def main():
    torch.set_num_threads(1)
    audit = {}
    source_files = [p for folder in ['models','experiments'] for p in sorted(Path(folder).glob('*.py'))]
    for round in ['pilot','revision']:
        root = Path('results')/round
        manifest = json.loads((root/'manifest.json').read_text())
        old = dict(np.load(root/'per_scene.npz'))
        backup = root/'pre_numerical_audit'
        backup.mkdir(exist_ok=True)
        for file in ['per_scene.npz','summary.json','summary.csv']:
            if not (backup/file).exists():
                shutil.copyfile(root/file, backup/file)
        output = {}
        changes = []
        seeds = [11,22,33,44,55]
        scenarios = list(manifest['scenarios'])
        for seed in seeds:
            tests = {s: make_dataset(300000+seed,192,scenario=s) for s in scenarios}
            for method in manifest['methods']:
                ckpt = torch.load(root/'checkpoints'/f'{seed}-{method}.pt', map_location='cpu', weights_only=True)
                model = Matcher(method)
                model.load_state_dict(ckpt['state_dict'])
                for scenario, data in tests.items():
                    values = evaluate(model,data)
                    for metric, array in values.items():
                        key = f'{seed}/{method}/{scenario}/{metric}'
                        assert np.isfinite(array).all(), key
                        output[key] = array
                        if metric=='r1' and not np.array_equal(array,old[key]):
                            changes.append({'key':key,'changed_scenes':int((array!=old[key]).sum()),
                                            'mean_r1_before':float(old[key].mean()),'mean_r1_after':float(array.mean())})
        rows = aggregate(output,seeds,manifest['methods'],scenarios)
        np.savez_compressed(root/'per_scene.npz',**output)
        (root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
        import csv
        with (root/'summary.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        record = {'reason':'floor eigenvalues before scalar logarithm; float32 extreme anisotropy can produce small negative eigenvalues',
                  'training_changed':False,'checkpoint_selection_changed':False,
                  'all_metrics_finite_after_audit':True,
                  'nonfinite_before':{k:int((~np.isfinite(v)).sum()) for k,v in old.items() if not np.isfinite(v).all()},
                  'r1_changes':changes,
                  'evaluation_source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
        for p in source_files:
            target=root/'evaluation_source'/p
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(p.read_bytes())
        manifest['evaluation_audit']=record
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        audit[round]=record
        print(round,'finite metrics; R1 changed:',changes,flush=True)
    Path('results/numerical_audit.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':
    main()
