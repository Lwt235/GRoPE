"""Read-only old-val interpretation of selected RGB Gaussian generators."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.round70_nvs_data import ROOT,OUT,NVSData
from experiments.round70_nvs_run import check,SEEDS
from experiments.round65_common import read,write,sha
from models.round70_nvs import CompactNVS,Camera,patchify,HEIGHT,WIDTH,PATCH,TOKENS
from models.round62_rgb_relations import gaussian_from_raw

ROLES=('rgb_raymap','rgb_prope','rgb_distance','rgb_direct','rgb_relative5','rgb_hrope')
DEST=OUT/'geometry_diagnostic1'


def matrix_log(cov):
    eig,vec=np.linalg.eigh(cov.astype('float64'))
    return (vec*np.log(np.maximum(eig,1e-15))[...,None,:])@np.swapaxes(vec,-1,-2)


def measurements(pm,pc,tm,tc,coverage):
    dz=np.abs(pm[...,2]-tm[...,2]);rel=dz/np.maximum(tm[...,2],1e-12)
    dl=matrix_log(pc)-matrix_log(tc)
    trace=np.trace(dl,axis1=-2,axis2=-1)
    shape=dl-trace[...,None,None]*np.eye(3)/3
    fields={'depth_abs_m':dz*3.,'depth_abs_relative':rel,
            'log_cov_frobenius':np.linalg.norm(dl,axis=(-2,-1)),
            'log_cov_traceless_frobenius':np.linalg.norm(shape,axis=(-2,-1))}
    result={}
    for name,valid in [('any_observed',coverage>0),('at_least_half_observed',coverage>=.5)]:
        per_view={key:np.sum(np.where(valid,val,0),axis=-1)/valid.sum(-1).clip(1) for key,val in fields.items()}
        nonempty=valid.any(-1)
        result[name]={'patch_count':int(valid.sum()),'reference_views':int(nonempty.sum()),
            'mean':{key:float(val[nonempty].mean()) for key,val in per_view.items()},
            'per_reference_view':{key:val.tolist() for key,val in per_view.items()}}
    pe=np.linalg.eigvalsh(pc.astype('float64'));te=np.linalg.eigvalsh(tc.astype('float64'))
    valid=coverage>0
    result['geometry_quantiles_05_50_95']={
        'predicted_depth_m':np.quantile(pm[...,2][valid]*3,[.05,.5,.95]).tolist(),
        'observed_depth_m':np.quantile(tm[...,2][valid]*3,[.05,.5,.95]).tolist(),
        'predicted_trace_m2':np.quantile(pe.sum(-1)[valid]*9,[.05,.5,.95]).tolist(),
        'observed_trace_m2':np.quantile(te.sum(-1)[valid]*9,[.05,.5,.95]).tolist(),
        'predicted_condition_number':np.quantile((pe[...,-1]/pe[...,0])[valid],[.05,.5,.95]).tolist(),
        'observed_condition_number':np.quantile((te[...,-1]/te[...,0])[valid],[.05,.5,.95]).tolist()}
    return result


def generate(model,ds,initial=False):
    means=[];covs=[]
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(ds),8):
            inp,_=ds.batch(np.arange(start,min(start+8,len(ds))),'rgb_prope','cpu')
            assert set(inp)=={'ref_rgb','K','c2w'}
            rays=model.create_rays(Camera(inp['K'],inp['c2w'],WIDTH,HEIGHT))
            x=model.input_tokenizer(torch.cat((patchify(inp['ref_rgb'],PATCH),patchify(rays,PATCH)[:,:2]),-1))
            raw=torch.zeros(*x.shape[:-1],7) if initial else model.geometry_head(x)
            yy,xx=torch.meshgrid((torch.arange(HEIGHT//PATCH)+.5)*PATCH,
                                (torch.arange(WIDTH//PATCH)+.5)*PATCH,indexing='ij')
            pixels=torch.stack((xx.flatten(),yy.flatten(),torch.ones(TOKENS)),-1)
            camera_rays=torch.einsum('bvij,nj->bvni',torch.linalg.inv(inp['K'][:,:2]),pixels)
            mean,cov=gaussian_from_raw(raw,camera_rays)
            means.append(mean.numpy());covs.append(cov.numpy())
    return np.concatenate(means),np.concatenate(covs)


def run():
    check();selections=read(OUT/'selections.json');assert selections['all_30_selected_before_old_eval']
    torch.set_num_threads(4)
    DEST.mkdir(exist_ok=False)
    ds=NVSData('val');refs=ds.arrays['triples'][:,[0,2]]
    tm,tc,coverage=[ds.arrays[k][refs] for k in ('mean','cov','coverage')]
    np.savez_compressed(DEST/'reference.npz',mean=tm,cov=tc,coverage=coverage,reference_indices=refs)
    report={'scope':'old-val diagnostic, no training or model selection','source_sha256':sha(Path(__file__)),
        'plan_sha256':sha(ROOT/'docs/round70_geometry_diagnostic_plan.md'),
        'selection_sha256':sha(OUT/'selections.json'),'precision':'CPU float32 generator; float64 metric arithmetic',
        'all_geometry_is_reference_only':True,'fresh_confirmation':False,'goal_complete':False,'runs':[]}
    torch.manual_seed(11);model=CompactNVS('rgb_prope');pm,pc=generate(model,ds,True)
    report['initial_fixed_generator']=measurements(pm,pc,tm,tc,coverage)
    for role in ROLES:
        for seed in SEEDS:
            checkpoint=OUT/'runs'/role/str(seed)/'best.pt'
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model=CompactNVS(role);model.load_state_dict(saved['model'])
            pm,pc=generate(model,ds)
            assert np.isfinite(pm).all() and np.isfinite(pc).all()
            np.savez_compressed(DEST/f'{role}_{seed}.npz',mean=pm,cov=pc)
            row=dict(role=role,seed=seed,checkpoint_sha256=sha(checkpoint),
                     selected_epoch=saved['epoch'],**measurements(pm,pc,tm,tc,coverage))
            report['runs'].append(row);print(role,seed,row['any_observed']['mean'],flush=True)
    write(DEST/'result.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for a,key in zip(ax,['depth_abs_relative','log_cov_frobenius','log_cov_traceless_frobenius']):
        a.axhline(report['initial_fixed_generator']['any_observed']['mean'][key],color='gray',ls='--',label='initial fixed')
        for i,role in enumerate(ROLES):
            y=[r['any_observed']['mean'][key] for r in report['runs'] if r['role']==role]
            a.scatter([i]*3,y);a.plot([i-.2,i+.2],[np.mean(y)]*2,color='black')
        a.set_xticks(range(6),[x[4:] for x in ROLES],rotation=30,ha='right');a.set_title(key+' (lower = closer to sensor moments)')
        a.grid(axis='y',alpha=.2)
    ax[0].legend();fig.suptitle('Old-val reference region consistency: interpretation only, not a new success gate')
    fig.savefig(DEST/'geometry.png',dpi=150);plt.close(fig)
    (DEST/'report.md').write_text('# R70 高斯参数解释检查（待实际图表审阅）\n\n'
        '18个已选RGB模型，全部旧val32三元组；不添加训练或选择几何最好checkpoint。\n'
        '传感器矩是区域观测参照，不是最优3DGS真值。CPU float32路径不用于代替正式bf16画质评价。\n'
        '详见result.json的两种预设覆盖条件与逐参考视图统计；geometry.png需实际查看。\n')
    write(DEST/'completion.json',dict(completed=True,visual_review_pending=True,scientific_review_pending=True,
                                   result_sha256=sha(DEST/'result.json'),unix=time.time(),goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-for-selections',action='store_true');args=p.parse_args()
    if args.wait_for_selections:
        while not (OUT/'selections.json').exists():
            q=read(OUT/'queue_status.json')
            if q['state']=='stopped':raise RuntimeError('main experiment stopped')
            proc=Path('/proc')/str(q['pid'])/'stat'
            if not proc.exists():raise RuntimeError('main controller disappeared')
            if proc.read_text().rsplit(')',1)[1].split()[19]!=q['start_ticks']:raise RuntimeError('controller identity changed')
            time.sleep(10)
    run()
