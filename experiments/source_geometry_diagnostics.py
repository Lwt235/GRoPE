"""Saved-field quality under source and query cameras; targets used for evaluation only."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.source_geometry_data import *
from experiments.gaussian_scene_score import score_torch,score_numpy
from experiments.gaussian_scene_carrier import feature_metrics
from models.gaussian_scene_memory import rasterize_field,contribution_weights

@torch.no_grad()
def main():
    torch.set_num_threads(4);p=plan();start=time.monotonic()
    for mode in MODES:
        for meta in p['training_catalog']+p['catalog']:
            rid=meta['id'];marker=branch_out(mode)/'diagnostics'/f'{rid}.json'
            if marker.exists():
                r=read(marker);assert sha256(r['maps_path'])==r['maps_sha256'];continue
            fieldpath=branch_cache(mode)/rid/'scene_memory.pt';fieldhash=sha256(fieldpath)
            field=load(fieldpath,'cuda');initial=load(branch_cache(mode)/rid/'coarse_initial.pt','cuda')
            source=load(source_path(rid),'cuda');source_features=load(CACHE/'inputs'/rid/'source_features.pt','cuda')['features']
            baseline=source_features.float().mean((0,1,2));records=[];maps={};maxcheck=0.
            views=[('source',j,index) for j,index in enumerate(SOURCES)]
            if meta in p['catalog']:views += [('query',q,q) for q in QUERIES]
            for view,j,index in views:
                if view=='source':
                    c2w=source['c2w'][j];K=source['K'][j].clone();K[:2]*=210/448
                    target=F.interpolate(source['depth_sensor'][j][None,None],(210,210),mode='nearest-exact')[0,0]
                    valid=F.interpolate(source['common_valid'][j][None,None].float(),(210,210),mode='nearest-exact')[0,0].bool()
                    rgb=F.interpolate(source['rgb'][j:j+1].float()/255,(210,210),mode='area')[0].permute(1,2,0)
                    teacher=source_features[j].reshape(1024,1024);Kf=source['K'][j].clone();Kf[:2]/=14
                else:
                    query=load(query_path(rid,j),'cuda');c2w=query['c2w'];K=query['K_metric'];target=query['target'];valid=query['valid']
                    rgb=F.interpolate(query['rgb'][None].float()/255,(210,210),mode='area')[0].permute(1,2,0)
                    teacher=load(patch_path(rid,j,'full'),'cuda')['patches'][0];Kf=query['K'].clone();Kf[:2]/=14
                row=dict(view=view,frame=index,valid_pixels=int(valid.sum()))
                for label,state in [('initial',initial),('final',field)]:
                    rendered,alpha,_=rasterize_field(state,c2w[None],K[None],210,210,mode='RGB+ED')
                    covered=alpha[0,...,0]>.0001
                    depth=torch.where(covered,rendered[0,...,3].clamp_min(1e-6),field['unit'])
                    masks={'whole':valid.cpu(),'covered':(valid&covered).cpu()}
                    a=score_torch(depth.cpu(),target.cpu(),masks);b=score_numpy(depth.cpu(),target.cpu(),masks)
                    for region in a:
                        for metric,value in a[region].items():
                            other=b[region][metric];assert (value is None)==(other is None)
                            if value is not None:maxcheck=max(maxcheck,abs(value-other))
                    row[label]=dict(rgb_mse=float((rendered[0,...,:3]-rgb).square().mean()),
                        covered_fraction_of_valid=float((covered&valid).sum()/valid.sum()) if bool(valid.any()) else None,depth_metrics=a)
                    maps[f'{view}_{index}_{label}_rgb']=rendered[0,...,:3].cpu()
                    maps[f'{view}_{index}_{label}_depth']=depth.cpu()
                    maps[f'{view}_{index}_{label}_alpha']=alpha[0,...,0].cpu()
                weights,_=contribution_weights(field,c2w[None],Kf[None],32,32)
                metrics,mse,cosine=feature_metrics(weights[0].reshape(1024,-1),field,teacher)
                global_mse=float((baseline[None]-teacher.float()).square().mean())
                global_cos=float(F.cosine_similarity(baseline[None],teacher.float(),dim=-1).mean())
                row['features']=dict(metrics,source_mean_baseline_all_patch_mse=global_mse,source_mean_baseline_cosine=global_cos)
                maps[f'{view}_{index}_observed_rgb']=rgb.cpu();maps[f'{view}_{index}_target']=target.cpu();maps[f'{view}_{index}_valid']=valid.cpu()
                records.append(row)
            assert maxcheck<1e-9 and sha256(fieldpath)==fieldhash
            path=branch_cache(mode)/rid/'diagnostic_maps.pt';save_tensor(path,maps)
            record=dict(complete=True,id=rid,mode=mode,split='development' if meta in p['catalog'] else 'training',
                source_state_sha256=fieldhash,source_state_unchanged=True,plan_sha256=sha256(branch_out(mode)/'plan.json'),
                maps_path=str(path),maps_sha256=sha256(path),records=records,independent_depth_metric_max_abs=maxcheck,
                scope='Quality diagnostic; source target mask is common source validity, query mask is all original valid GT. Covered-only denominators explicit; no query labels in field construction.')
            write_json(marker,record);print(dict(diagnostics_complete=mode,id=rid),flush=True)
            del field,initial,source,source_features,maps;torch.cuda.empty_cache()
    write_json(OUT/'diagnostics_complete.json',dict(complete=True,fields=48,seconds=time.monotonic()-start,no_native_sample_exclusion=True))

if __name__=='__main__':main()
