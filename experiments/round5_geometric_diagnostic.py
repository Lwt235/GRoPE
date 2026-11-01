"""Parameter-free same-input projected-center diagnostic, not primary gate.

Declared after training/validation development and before test evaluation.
No learned appearance, covariance, ground-truth mask, or sensor depth chooses
the prediction; labels select only the evaluation population.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.round5_train import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--splits',nargs='+',default=['train','val']);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError('refuse overwrite diagnostic')
    yy,xx=np.mgrid[:18,:24];grid=np.stack([xx.flatten()*8+3.5,yy.flatten()*8+3.5],-1)
    summaries=[]
    for split in args.splits:
        for d in sorted((ROOT/'results/round5_inputs/pairs'/split).iterdir()):
            if not d.is_dir():continue
            manifest=json.loads((d/'manifest.json').read_text())
            if sha(d/'inputs.npz')!=manifest['input_sha256'] or sha(d/'labels.npz')!=manifest['label_sha256']:raise ValueError('cache hash mismatch')
            data=np.load(d/'inputs.npz');labels=np.load(d/'labels.npz');rows=[]
            for i,(qi,ki) in enumerate(manifest['pairs']):
                distance=((data['qmean'][i,:,None]-data['k_to_q'][i,None,:])**2).sum(-1)
                pred=grid[distance.argmin(-1)]
                valid=labels['gt_valid'][i];err=np.linalg.norm(pred-labels['pixcontinuous'][i],axis=-1)[valid]
                rows.append({'q_index':qi,'k_index':ki,'n_valid':len(err),'pck8_count':int((err<=8).sum()),
                             'pck4_count':int((err<=4).sum()),'epe_sum':float(err.sum())})
            n=sum(r['n_valid'] for r in rows)
            summaries.append({'scene':manifest['scene'],'split':split,'n_valid':n,'rows':rows,
                              'pck8':sum(r['pck8_count'] for r in rows)/n,
                              'pck4':sum(r['pck4_count'] for r in rows)/n,
                              'epe':sum(r['epe_sum'] for r in rows)/n,
                              'inputs_sha256':manifest['input_sha256'],'labels_sha256':manifest['label_sha256']})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({'source_sha256':sha(__file__),'role':'parameter-free same-input diagnostic; not primary success gate','scenes':summaries},indent=2))
    for split in args.splits:
        rs=[r for r in summaries if r['split']==split]
        print(split,'scene_macro_pck8',float(np.mean([r['pck8'] for r in rs])))

if __name__=='__main__':main()
