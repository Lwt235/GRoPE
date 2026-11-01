"""CPU-only context-loss portion of the fixed R41 component diagnostic.

This does not replace or restart training, render on CPU, or claim the pending
component image experiment is complete. GPUs were occupied by other users.
"""
from pathlib import Path
import time
import numpy as np
import torch
from models.round41_gs_readout import GSReadout
from experiments.round41_gs_data import SCENES,SEEDS,load_scene,query,read,write,sha
from experiments.round41_gs_train import schedule
from experiments.round41_shape_components import TRAIN,DEST,NAMES,frozen,logits


@torch.no_grad()
def main():
    torch.set_num_threads(4);frozen();out=DEST/'cpu_context_attempt2'
    if out.exists():raise ValueError('preserve context diagnostic')
    out.mkdir();(out/'source.py').write_text(Path(__file__).read_text())
    write(out/'started.json',dict(source_sha256=sha(__file__),unix=time.time(),
        device='cpu',threads=4,reason='all GPUs occupied by other users; no training or CPU renderer fallback'))
    rows=[]
    for scene in SCENES:
        data=load_scene(scene,TRAIN,device='cpu');_,validation=schedule(data)
        for seed in SEEDS:
            start=time.time();path=TRAIN/'runs'/scene/'decoupled'/str(seed)
            state=torch.load(path/'best.pt',map_location='cpu',weights_only=True)['state_dict']
            model=GSReadout('decoupled');model.load_state_dict(state);model.eval()
            sums={n:0. for n in NAMES};elements=0
            for target,k,ids in validation:
                key=data['views'][k];q=query(data,k,ids,model,target=target)
                encoded=model.encode(key['feature'],key['anchor'],key['cov'])
                for name,score in logits(model,q,key,encoded).items():
                    error=(score.softmax(-1)@key['rgb']-data['views'][target]['center_rgb'][ids]).square()
                    sums[name]+=float(error.double().sum())
                elements+=len(ids)*3
            mse={k:v/elements for k,v in sums.items()}
            original=read(path/'run.json')['context_selection_mse'];difference=abs(mse['full']-original)
            row=dict(scene=scene,seed=seed,mse=mse,elements=elements,
                original_full_mse=original,full_replay_error=difference,
                replay_warning=difference>1e-7,seconds=time.time()-start)
            rows.append(row);write(out/f'{scene}_{seed}.json',row)
            print(scene,seed,mse,'replay error',difference,flush=True)
    write(out/'summary.json',dict(records=rows,mean_mse={n:float(np.mean([r['mse'][n] for r in rows])) for n in NAMES},
        trained=False,target_rgb_read=False,component_rendering_complete=False,primary_result_unchanged=True))


if __name__=='__main__':main()
