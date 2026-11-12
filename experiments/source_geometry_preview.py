"""Fixed saved-field preview while learned comparisons are still running; no new inference."""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.source_geometry_data import *

def main():
    p=plan();rid=p['catalog'][0]['id'];maps={m:load(branch_cache(m)/rid/'diagnostic_maps.pt') for m in MODES}
    folder=OUT/'preview';folder.mkdir(exist_ok=True);fig,axes=plt.subplots(3,6,figsize=(16,8),layout='constrained')
    titles=['Observed RGB','Pred field RGB','Sensor field RGB','Sensor target','Pred field depth','Sensor field depth']
    for row,(view,q) in enumerate([('source',0),('query',2),('query',11)]):
        key=f'{view}_{q}';target=maps['sensor'][key+'_target'].numpy();valid=maps['sensor'][key+'_valid'].numpy();limits=np.percentile(target[valid],[2,98]) if valid.any() else [.1,10.]
        axes[row,0].imshow(maps['sensor'][key+'_observed_rgb'].numpy().clip(0,1))
        for col,mode in [(1,'pred'),(2,'sensor')]:axes[row,col].imshow(maps[mode][key+'_final_rgb'].numpy().clip(0,1))
        axes[row,3].imshow(np.where(valid,target,np.nan),vmin=limits[0],vmax=limits[1],cmap='viridis')
        for col,mode in [(4,'pred'),(5,'sensor')]:
            d=maps[mode][key+'_final_depth'].numpy();alpha=maps[mode][key+'_final_alpha'].numpy();im=axes[row,col].imshow(np.where(alpha>1e-4,d,np.nan),vmin=limits[0],vmax=limits[1],cmap='viridis')
        fig.colorbar(im,ax=list(axes[row,3:]),shrink=.7,label='metres')
        for col in range(6):
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(titles[col])
        axes[row,0].set_ylabel(f'{rid}\n{view}{q}')
    fig.suptitle('Fixed saved-field preview | same cameras, source0 and independent query2/query11\nSource target uses joint-valid mask; white: masked target or uncovered field. Not a learned-method result.')
    path=folder/'fixed_scene.png';fig.savefig(path,dpi=140);plt.close(fig)
    write_json(folder/'manifest.json',dict(complete=True,file=path.name,sha256=sha256(path),id=rid,views=['source0','query2','query11'],actual_review_pending=True,scope='Predetermined scene only, all12places retained in numerical summaries; no new inference or method selection.'))
    print(path)

if __name__=='__main__':main()
