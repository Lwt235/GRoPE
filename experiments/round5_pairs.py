"""Build separate forward-input and supervision caches for frozen SUN3D pairs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.round5_cache import DATA,CACHE,native_cameras,projected_support,hash_file
from experiments.round5_rgbd_data import load_sensor_depth,pair_supervision,token_grid

OUT=ROOT/'results/round5_inputs/pairs'

def input_pair(q,k):
    data={'q':q['features'],'k':k['features']}
    diagnostics={}
    for name,source,target in [('q',q,q),('k',k,k),('k_to_q',k,q),('q_to_k',q,k)]:
        if name in ('q','k'):
            # The source image's token footprint is known exactly. Do not
            # reproject a 3D moment approximation to invent native coordinates.
            mu=(token_grid()/8).astype('float32')
            cov=np.broadcast_to(np.eye(2)*(63/(12*64)),(432,2,2)).astype('float32')
            diag={'behind_or_near':0,'cov_clipped':0,'mean_clipped':0}
        else:
            mu,cov,diag=projected_support(source['mean3'],source['cov3'],source['c2w'],target['c2w'],target['K'])
        mean_name=name+'mean' if name in ('q','k') else name
        cov_name=name+'cov'
        data[mean_name]=mu;data[cov_name]=cov
        diagnostics[name]=diag
    return data,diagnostics

def main():
    p=argparse.ArgumentParser();p.add_argument('--splits',nargs='+',default=['train','val']);args=p.parse_args()
    selection=json.loads((DATA/'selection.json').read_text())
    summaries=[]
    for scene in selection['scenes']:
        if scene['split'] not in args.splits:continue
        dest=OUT/scene['split']/scene['scene'].replace('/','__');dest.mkdir(parents=True,exist_ok=True)
        K,poses=native_cameras(scene)
        frames={f['index']:f for f in scene['frames']}
        cached={}
        for i,frame in frames.items():
            fpath=CACHE/scene['scene']/f'{i:07d}.npz'
            provenance=json.loads(fpath.with_suffix('.json').read_text())
            if provenance['cache_sha256']!=hash_file(fpath):raise ValueError('corrupt frame cache')
            prod=provenance['producer']
            if (prod['source_sha256']!=hash_file(ROOT/'experiments/round5_cache.py') or
                prod['rgb_sha256']!=hash_file(DATA/scene['scene']/'image'/frame['rgb']) or
                prod['K_sha256']!=scene['K_sha256'] or prod['pose_sha256']!=scene['pose_sha256'] or
                prod['original_index']!=i):raise ValueError('stale frame cache')
            cached[i]=dict(np.load(fpath))
        # Supervision is constructed only after forward arrays are materialized.
        inputs=[];diags=[]
        for qi,ki in scene['pairs']:
            x,d=input_pair(cached[qi],cached[ki]);inputs.append(x);diags.append(d)
        np.savez_compressed(dest/'inputs.npz',**{key:np.stack([x[key] for x in inputs]) for key in inputs[0]})
        labels=[];statuses=Counter();oracle=[]
        for qi,ki in scene['pairs']:
            dq=load_sensor_depth(DATA/scene['scene']/'depth'/frames[qi]['depth'])
            dk=load_sensor_depth(DATA/scene['scene']/'depth'/frames[ki]['depth'])
            label=pair_supervision(dq,dk,K,K,poses[qi],poses[ki])
            labels.append(label)
            statuses.update(label.status.tolist())
            valid=label.gt_valid
            pix=label.pixcontinuous[valid]
            nearest=(np.rint((pix-3.5)/8).clip([0,0],[23,17])*8+3.5)
            oracle.extend(np.linalg.norm(nearest-pix,axis=-1).tolist())
        keys=['gt_valid','pixcontinuous','key_indices','key_weights','status']
        np.savez_compressed(dest/'labels.npz',**{key:np.stack([getattr(x,key) for x in labels]) for key in keys})
        summary={'scene':scene['scene'],'split':scene['split'],'pairs':scene['pairs'],
                 'input_sha256':hash_file(dest/'inputs.npz'),'label_sha256':hash_file(dest/'labels.npz'),
                 'status_counts':{str(k):v for k,v in statuses.items()},'n_valid':len(oracle),
                 'grid_oracle_pck8':float(np.mean(np.array(oracle)<=8)) if oracle else None,
                 'grid_oracle_pck4':float(np.mean(np.array(oracle)<=4)) if oracle else None,
                 'projection_diagnostics':diags,'label_metadata':labels[0].metadata}
        (dest/'manifest.json').write_text(json.dumps(summary,indent=2))
        summaries.append(summary)
        print(scene['scene'],'pairs',len(inputs),'valid',len(oracle),'fraction',round(len(oracle)/(432*len(inputs)),3),flush=True)
    (OUT/('summary_'+'_'.join(args.splits)+'.json')).write_text(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
