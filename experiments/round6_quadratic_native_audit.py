"""Native float32 softmax/CE supplement on declared validation prefixes."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from experiments.round6_quadratic_audit import (
    ROOT, FIELDS, SCENES, CONFIGS, sha, pair_score,
)
from models.gaussian_quadratic_pe import GaussianQuadraticMatcher


def main():
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    out = ROOT/'results/round6_quadratic_audit'
    original = json.loads((out/'audit.json').read_text())
    # First and last are fixed by archive index, never by a prediction metric.
    rows = []
    for source in original['data_sources']:
        path = ROOT/source['path']
        manifest = json.loads((path/'manifest.json').read_text())
        if manifest['scene'] not in SCENES or manifest['split'] != 'val':
            raise ValueError('only previously inspected validation places allowed')
        if sha(path/'inputs.npz') != source['input_sha256'] or sha(path/'labels.npz') != source['label_sha256']:
            raise ValueError('changed archive')
        ids = [0, source['pairs']-1]
        with np.load(path/'inputs.npz') as z:
            d = {k:torch.from_numpy(z[k][ids]) for k in FIELDS}
        with np.load(path/'labels.npz') as z:
            labels = torch.from_numpy(z['soft_labels'][ids]).double()
            valid = torch.from_numpy(z['gt_valid'][ids]).bool()
            mass = torch.from_numpy(z['sample_count'][ids]).double()*torch.from_numpy(z['visible_fraction'][ids]).double()
        double = {k:v.double() for k,v in d.items()}
        for method,width,temp in CONFIGS:
            model = GaussianQuadraticMatcher(method,width,temp)
            scores = dict(lift32=model(d),pair32=pair_score(d,method,width,temp),
                          reference64=pair_score(double,method,width,temp,reference=True))
            result = {}
            for left,right in [('lift32','pair32'),('lift32','reference64'),('pair32','reference64')]:
                a,b = scores[left],scores[right]
                p,q = a.softmax(-1).double(),b.softmax(-1).double()
                ace = -(labels*a.log_softmax(-1).double()).sum(-1)
                bce = -(labels*b.log_softmax(-1).double()).sum(-1)
                delta = ace-bce
                result[left+'_vs_'+right] = dict(max_probability_difference=float((p-q).abs().max()),
                    max_valid_row_ce_difference=float(delta[valid].abs().max()),
                    mean_absolute_visible_mass_weighted_ce_difference=float((delta.abs()*mass).sum()/mass.sum()),
                    valid_argmax_mismatches=int(((a.argmax(-1)!=b.argmax(-1))&valid).sum()))
            rows.append(dict(scene=source['scene'],grid=source['grid'],indices=ids,method=method,
                bandwidth=width,temperature=temp,comparisons=result))
    files = ('experiments/round6_quadratic_native_audit.py','experiments/round6_quadratic_audit.py',
             'models/gaussian_quadratic_pe.py','models/gaussian_measure_pe.py')
    hashes = {f:sha(ROOT/f) for f in files}
    destination = out/'source_snapshot/experiments/round6_quadratic_native_audit.py'
    shutil.copy2(__file__,destination)
    (out/'native_precision_subset.json').write_text(json.dumps(dict(
        scope='native float32 softmax/log_softmax; first and last pair per old validation scene/grid',
        source_sha256=hashes,parent_audit_sha256=sha(out/'audit.json'),
        pair_instances=16,configurations=5,records=rows,heldout_read=False),indent=2))
    print('DONE',len(rows),'records',flush=True)


if __name__ == '__main__':main()
