"""Complete locked geometry-label coverage while GPU resources are unavailable.

Reads already saved labels only; no projection, new prediction or threshold fit.
"""
import time,resource,collections
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import native_track_memory_data as d

def main():
    start=time.monotonic();torch.set_num_threads(2);rows=[];reject=collections.Counter();values=[];train={m['id'] for m in d.plan()['training_catalog']}
    for meta in d.metas():
        rid=meta['id']
        for a in d.SOURCES:
            for q in d.QUERIES:
                obj=d.load(d.case(rid,a,q));lab=obj['labels'];v=lab['valid'];rows.append(dict(id=rid,place=meta['place'],cohort='train32' if rid in train else 'dev12',anchor=a,query=q,query_set='original' if q in d.QUERIES[:4] else 'other',valid_GT=int(v.sum()),zero_support=obj['inputs']['zero_support'],zero_attention_keys=not bool(obj['inputs']['GS']['key_mask'].any()),initializer_valid=obj['inputs']['alignment_valid'],stage_counts=lab['stage_counts']))
                for x in lab['stage_counts']:reject[x['name']]+=x['rejected']
                if v.any():values.append(lab['residuals'][v].numpy())
    residual=np.concatenate(values);stats={name:dict(median=float(np.median(residual[:,i])),p95=float(np.percentile(residual[:,i],95)),maximum=float(residual[:,i].max())) for i,name in enumerate(('forward_depth_metres','reverse_depth_metres','anchor_cycle_pixels210','query_cycle_pixels210'))};summary=dict(complete=True,cases=len(rows),all_grid_points=len(rows)*64,valid_GT_points=sum(r['valid_GT'] for r in rows),zero_GT_pairs=sum(r['valid_GT']==0 for r in rows),zero_raster_pairs=sum(r['zero_support'] for r in rows),zero_attention_key_pairs=sum(r['zero_attention_keys'] for r in rows),invalid_initializer_pairs=sum(not r['initializer_valid'] for r in rows),ordered_rejection_counts=dict(reject),valid_label_residuals=stats,all_cases=rows,no_geometry_recomputed=True,no_model_calls=True,no_threshold_changes=True,CUDA_initialized=torch.cuda.is_initialized())
    d.write(d.OUT/'scoring/label_geometry_summary.json',summary)
    out=d.OUT/'figures';out.mkdir(exist_ok=True);fig,ax=plt.subplots(1,2,figsize=(13,5));ax[0].barh(list(reject),list(reject.values()),color='#4c78a8');ax[0].invert_yaxis();ax[0].set(xlabel='Points rejected at ordered stage',title='Fixed sensor / pose visibility labels');ids=[m['id'] for m in d.metas()];matrix=np.asarray([[r['valid_GT'] for r in rows if r['id']==rid] for rid in ids]);im=ax[1].imshow(matrix,aspect='auto',vmin=0,vmax=64,cmap='viridis');ax[1].axhline(31.5,c='white',lw=1)
    for x in (7.5,15.5,23.5):ax[1].axvline(x,c='white',lw=.6)
    ax[1].set(yticks=np.arange(0,44,3),yticklabels=ids[::3],xticks=[3.5,11.5,19.5,27.5],xticklabels=['anchor0','anchor3','anchor6','anchor9'],xlabel='Eight fixed queries per anchor',title='All44 places; train32 above white line');fig.colorbar(im,ax=ax[1],label='Valid labels /64');fig.tight_layout();fig.savefig(out/'coverage.png',dpi=180);plt.close(fig)
    d.write(d.OUT/'CPU_label_runtime.json',dict(complete=True,wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized(),new_model_calls=0,new_geometry_projection=0));print({k:summary[k] for k in ('cases','valid_GT_points','zero_GT_pairs','zero_raster_pairs','invalid_initializer_pairs')},flush=True)
if __name__=='__main__':main()
