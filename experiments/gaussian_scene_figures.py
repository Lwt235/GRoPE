"""Inspectable result, source-carrier and fixed/worst-case depth figures."""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_scene_data import OUT,CACHE,query_path,query_image,source_path,read,KINDS
from experiments.gaussian_scene_bind import load_tensor
from experiments.gaussian_scene_config import freeze
from experiments.gaussian_memory_incremental_data import write_json


def plot_results(p,summary,folder):
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ai,(condition,region,metric,title) in enumerate([
        ('missing','roi','centered_log_mse','Missing ROI: centered log MSE (primary)'),
        ('missing','roi','bias_squared','Missing ROI: squared log bias'),
        ('missing','roi','log_mse','Missing ROI: raw log MSE'),
        ('full','roi','centered_log_mse','Full input ROI: centered log MSE'),
        ('missing','whole','absrel','Missing input: whole-image AbsRel'),
        ('full','whole','absrel','Full input: whole-image AbsRel')]):
        ax=axes.flat[ai]
        for i,kind in enumerate(KINDS):
            row=summary['summaries'][f'{condition}/{region}/{metric}/{kind}']
            ax.bar(i,row['mean'],color='#467da7' if kind!='G' else '#bd6038',alpha=.8)
            ax.scatter([i]*3,row['per_seed'],color='black',s=15,zorder=3)
        ax.set_xticks(range(5),KINDS);ax.set_title(title,fontsize=10);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Saved Gaussian scene memory: old-development comparison; known cameras',fontsize=13)
    fig.tight_layout();fig.savefig(folder/'outcomes.png',dpi=160);plt.close(fig)


def prediction_lookup(kind,seed):
    payload=load_tensor(CACHE/'evaluation'/f'{kind}_{seed}.pt')
    return {tuple(key):depth for key,depth in zip(payload['keys'],payload['depth_metric'])}


def plot_cases(p,rows,folder):
    selected=[]
    for rid,q in [(p['catalog'][0]['id'],2),(p['catalog'][0]['id'],11),(p['catalog'][1]['id'],5)]:selected.append((rid,q,'missing'))
    effects=[]
    for meta in p['catalog']:
        for q in p['queries']:
            by={kind:[r['metrics']['roi']['centered_log_mse'] for r in rows if r['id']==meta['id'] and r['query']==q and r['condition']=='missing' and r['kind']==kind] for kind in ('P','G')}
            effects.append((np.mean(by['G'])-np.mean(by['P']),meta['id'],q))
    worst=max(effects,key=lambda x:(x[0],x[1],x[2]));selected.append((worst[1],worst[2],'missing'))
    predictions={kind:prediction_lookup(kind,p['seeds'][0]) for kind in ('L','T','P','G')}
    rendered=load_tensor(CACHE/'evaluation/renderer_reference.pt')
    render={tuple(key[:3]):depth for key,depth in zip(rendered['keys'],rendered['depth_metric'])}
    fig,axes=plt.subplots(len(selected),7,figsize=(16,3.0*len(selected)));case_records=[]
    for r,key in enumerate(selected):
        rid,q,c=key;query=load_tensor(query_path(rid,q));target=query['target'].numpy();valid=query['valid'].numpy()
        limits=np.percentile(target[valid],[2,98]);limits[0]=max(.1,limits[0])
        image=query_image(query,c).permute(1,2,0).numpy()
        axes[r,0].imshow(image);axes[r,0].set_ylabel(f'{rid} / q{q}\n'+('largest G-P error difference' if r==3 else 'fixed case')+f'\n{limits[0]:.2f}–{limits[1]:.2f} m',fontsize=9)
        case_records.append(dict(id=rid,query=q,condition=c,depth_vmin_m=float(limits[0]),depth_vmax_m=float(limits[1]),
                                 valid_target_pixels=int(valid.sum())))
        panels=[target]+[predictions[kind][key].numpy() for kind in ('L','T','P','G')]+[render[key].numpy()]
        for col,value in enumerate(panels,1):axes[r,col].imshow(np.where(valid,value,np.nan),vmin=limits[0],vmax=limits[1],cmap='viridis')
        for ax in axes[r]:ax.set_xticks([]);ax.set_yticks([])
    for ax,title in zip(axes[0],['Current RGB','Sensor target','L','T','P','G','Direct field depth']):ax.set_title(title)
    fig.suptitle('One final seed shown; selection uses all3 seeds; row depth limits: sensor2–98th percentiles, metres',fontsize=11)
    fig.tight_layout();fig.savefig(folder/'depth_cases.png',dpi=150);plt.close(fig)
    write_json(folder/'case_selection.json',dict(fixed_and_worst=selected,worst_G_minus_P_centered_roi=float(worst[0]),
        seed_shown=p['seeds'][0],worst_selected_by='mean3seeds, missing ROI centered log MSE G minus P, largest signed difference',
        no_case_exclusion=True,cases=case_records,
        depth_display='Same limits across all depth panels in each row; clipped to sensor2–98th percentiles with minimum0.1m; metrics use unclipped predictions'))


def plot_carrier(p,folder):
    rid=p['catalog'][0]['id'];maps=load_tensor(CACHE/rid/'carrier_maps.pt')
    fig,axes=plt.subplots(3,4,figsize=(11,8))
    source=load_tensor(source_path(rid));rows=[('source',0),('query',2),('query',11)]
    for row,(view,index) in enumerate(rows):
        payload=source if view=='source' else load_tensor(query_path(rid,index))
        image=payload['rgb'][0] if view=='source' else payload['rgb']
        axes[row,0].imshow(image.permute(1,2,0).numpy());axes[row,0].set_ylabel(f'{rid} {view}{index}')
        for col,name in enumerate(('raw_dc','coarse_initial','coarse_final'),1):axes[row,col].imshow(maps[f'{view}_{index}_{name}'].numpy().clip(0,1))
        for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
    for ax,title in zip(axes[0],['Observed RGB','Raw source GS/DC','2048 before fitting','2048 after300 steps']):ax.set_title(title,fontsize=10)
    fig.suptitle('Same saved source-only scene, source and two independent queries; known cameras',fontsize=11)
    fig.tight_layout();fig.savefig(folder/'scene_reobservation.png',dpi=150);plt.close(fig)


def main():
    p=freeze();summary=read(OUT/'scoring/summary.json');rows=read(OUT/'scoring/metrics.json')['rows']
    assert summary['complete'] and read(OUT/'carrier/summary.json')['complete']
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    plot_results(p,summary,folder);plot_cases(p,rows,folder);plot_carrier(p,folder)
    write_json(folder/'manifest.json',dict(complete=True,files=['outcomes.png','depth_cases.png','scene_reobservation.png'],
        visual_review_pending=True,scope='Diagnostic research figures; not evidence of a completed visual review'))


if __name__=='__main__':main()
