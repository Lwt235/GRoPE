"""CPU previews of the first two fixed old-development scene re-observations."""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_scene_data import OUT,CACHE,query_path,read
from experiments.gaussian_scene_bind import load_tensor
from experiments.gaussian_memory_incremental_data import write_json
from experiments.gaussian_memory_setup import sha256


def main():
    torch.set_num_threads(4);p=read(OUT/'plan.json')
    reference=read(OUT/'references/renderer.json')
    assert sha256(reference['prediction_path'])==reference['prediction_sha256']
    payload=load_tensor(reference['prediction_path'])
    lookup={tuple(key[:3]):i for i,key in enumerate(payload['keys'])}
    folder=OUT/'scene_preview';folder.mkdir(exist_ok=True);records=[]
    for meta in p['catalog'][:2]:
        rid=meta['id'];fig,axes=plt.subplots(5,4,figsize=(10,12))
        for col,q in enumerate(p['queries']):
            query=load_tensor(query_path(rid,q));i=lookup[(rid,q,'full')]
            image=F.interpolate(query['rgb'][None].float()/255,(210,210),mode='area')[0].permute(1,2,0).numpy()
            target=query['target'].numpy();valid=query['valid'].numpy()
            vmax=float(np.percentile(target[valid],98))
            axes[0,col].imshow(image);axes[0,col].set_title(f'{rid}: query{q}')
            axes[1,col].imshow(payload['rgb'][i].numpy().clip(0,1))
            axes[2,col].imshow(payload['alpha'][i].numpy(),vmin=0,vmax=1,cmap='gray')
            axes[3,col].imshow(np.where(valid,target,np.nan),vmin=.1,vmax=vmax,cmap='viridis')
            axes[4,col].imshow(np.where(valid,payload['depth_metric'][i].numpy(),np.nan),vmin=.1,vmax=vmax,cmap='viridis')
            for ax in axes[:,col]:ax.set_xticks([]);ax.set_yticks([])
        for ax,label in zip(axes[:,0],['Observed RGB','Saved scene RGB/DC','Scene alpha','Sensor depth','Direct scene depth']):ax.set_ylabel(label)
        fig.suptitle('One source-only saved field, four given query cameras; first fixed old places\nCarrier diagnostic, not learned native utility; depth fill at uncovered pixels is source-only unit',fontsize=10)
        fig.tight_layout(rect=(0,0,1,.96));path=folder/f'{rid}.png';fig.savefig(path,dpi=140);plt.close(fig)
        records.append(dict(id=rid,figure=str(path),figure_sha256=sha256(path),queries=p['queries'],
            saved_field_sha256=sha256(CACHE/rid/'scene_memory.pt'),selection='first2fixeddevelopment catalog entries',
            independent_saved_scene_used=True))
    write_json(folder/'manifest.json',dict(complete=True,records=records,visual_review_pending=True,
        scope='Fixed source carrier previews only, no configuration selection or PE effectiveness claim'))


if __name__=='__main__':main()
