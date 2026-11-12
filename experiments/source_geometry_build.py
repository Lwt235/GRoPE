"""Paired source-camera backprojection, fixed2048 seeds and source-only300-step fit."""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from experiments.source_geometry_data import *
from models.gaussian_scene_memory import rasterize_field,field_to_device

CONFIG=dict(slots=2048,steps=300,fit_size=224,feature_grid=32,
    candidate_rule='Nearest joint-valid224px center in each7x7 source feature cell; source-major uniform quantiles; if fewer2048 cells use all joint-valid pixels',
    paired_candidate_indices=True,source_depth_valid_range_m=[.1,10.],
    initial_opacity=.5,initial_tangent_std='depth*7/sqrt(12)/focal_at224',
    initial_thickness_ratio=.1,initial_rotation='given source camera rotation, WXYZ',
    means_lr_unit_fraction=.001,rotations_lr=.001,log_scales_lr=.02,opacity_logits_lr=.02,color_logits_lr=.02,
    minimum_scale_fraction=1e-5,maximum_scale_fraction=2.,rgb_l1_weight=1.,log_depth_l1_weight=.1,
    coverage_mse_weight=.1,view_schedule='step modulo four',queries_used=False)

def sampled_inputs(source):
    depths={m:F.interpolate(source['depth_'+m][:,None],(224,224),mode='nearest-exact')[:,0] for m in MODES}
    valid=F.interpolate(source['common_valid'][:,None].float(),(224,224),mode='nearest-exact')[:,0].bool()
    # Identical selected source image indices for both depth modes. No3D FPS or depth-dependent ranking.
    local=(torch.arange(7,device=valid.device)+.5-3.5).square()
    distance=(local[:,None]+local[None,:]).reshape(49)
    cells=valid.reshape(4,32,7,32,7).permute(0,1,3,2,4).reshape(-1,49)
    best=torch.where(cells,distance[None],torch.inf).argmin(-1)
    cell_ids=torch.where(cells.any(-1))[0]
    view=cell_ids//1024;cy=(cell_ids%1024)//32;cx=cell_ids%32
    selected=best[cell_ids];indices=view*224*224+(cy*7+selected//7)*224+cx*7+selected%7
    fallback=len(indices)<CONFIG['slots']
    if fallback:indices=torch.where(valid.reshape(-1))[0]
    if len(indices)<CONFIG['slots']:raise ValueError(f'Only{len(indices)} joint-valid source pixels; preserve failed case for diagnosis, no scene exclusion')
    quantile=((torch.arange(CONFIG['slots'],device=valid.device)+.5)*len(indices)/CONFIG['slots']).long()
    return depths,valid,indices[quantile],dict(valid_cells=len(cell_ids),candidate_count=len(indices),dense_fallback=fallback)

def initial_field(source,mode,depths,indices):
    device=source['rgb'].device;view=indices//(224*224);y=(indices%(224*224))//224;x=indices%224
    K=source['K'].clone();K[:,:2]/=2
    pixels=torch.stack((x+.5,y+.5,torch.ones_like(x)),1).float()
    d=depths[mode][view,y,x]
    rays=torch.einsum('nij,nj->ni',torch.linalg.inv(K)[view],pixels)
    camera=rays*d[:,None]
    means=torch.einsum('nij,nj->ni',source['c2w'][view,:3,:3],camera)+source['c2w'][view,:3,3]
    quaternion=torch.tensor(Rotation.from_matrix(source['c2w'][:,:3,:3].cpu().numpy()).as_quat()[:,[3,0,1,2]],device=device,dtype=torch.float32)
    sx=d*7/math.sqrt(12)/K[view,0,0];sy=d*7/math.sqrt(12)/K[view,1,1]
    scales=torch.stack((sx,sy,.1*(sx*sy).sqrt()),-1).clamp_min(float(source['unit'])*1e-5)
    rgb=F.interpolate(source['rgb'].float()/255.,(224,224),mode='area')
    colors=rgb.permute(0,2,3,1).reshape(-1,3)[indices]
    field=dict(means=means,rotations=quaternion[view],scales=scales,colors=colors,opacities=torch.full_like(d,.5))
    # Independent analytical unprojection check in the generating source image.
    cam=torch.einsum('nij,nj->ni',source['c2w'][view,:3,:3].transpose(-1,-2),means-source['c2w'][view,:3,3])
    projected=torch.einsum('nij,nj->ni',K[view],cam);projected=projected[:,:2]/projected[:,2:]
    assert torch.max(torch.abs(projected-pixels[:,:2]))<.01
    return field,float(torch.max(torch.abs(projected-pixels[:,:2])))

def fit_one(source,mode,depths,valid,indices,candidate):
    rid=source['id'];folder=branch_cache(mode)/rid;folder.mkdir(parents=True,exist_ok=True)
    marker=branch_out(mode)/'construction'/f'{rid}.json'
    if marker.exists():
        r=read(marker);assert r['complete'] and sha256(folder/'coarse_geometry.pt')==r['geometry_sha256'];return r
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.monotonic()
    initial,reprojection=initial_field(source,mode,depths,indices);save_tensor(folder/'coarse_initial.pt',initial)
    unit=source['unit'];K=source['K'].clone();K[:,:2]/=2
    images=F.interpolate(source['rgb'].float()/255,(224,224),mode='area')
    with torch.no_grad():_,initial_alpha,_=rasterize_field(initial,source['c2w'],K,224,224)
    params=dict(means=torch.nn.Parameter(initial['means'].clone()),rotations=torch.nn.Parameter(initial['rotations'].clone()),
        log_scales=torch.nn.Parameter(initial['scales'].log()),opacity_logits=torch.nn.Parameter(torch.logit(initial['opacities'])),
        color_logits=torch.nn.Parameter(torch.logit(initial['colors'].clamp(.0001,.9999))))
    optimizer=torch.optim.Adam([dict(params=[params[n]],lr=lr) for n,lr in
        [('means',.001*float(unit)),('rotations',.001),('log_scales',.02),('opacity_logits',.02),('color_logits',.02)]])
    def current():return dict(means=params['means'],rotations=F.normalize(params['rotations'],dim=-1),scales=params['log_scales'].exp(),opacities=params['opacity_logits'].sigmoid(),colors=params['color_logits'].sigmoid())
    log=[]
    for step in range(300):
        v=step%4;optimizer.zero_grad(set_to_none=True)
        result,alpha,_=rasterize_field(current(),source['c2w'][v:v+1],K[v:v+1],224,224,mode='RGB+ED')
        rgb=(result[0,...,:3].permute(2,0,1)-images[v]).abs().mean()
        covered=(alpha[0,...,0].detach()>.0001)&valid[v]
        depth=(result[0,...,3].clamp_min(1e-6)[covered].log()-depths[mode][v][covered].log()).abs().mean() if covered.any() else result.sum()*0
        coverage=(alpha-initial_alpha[v:v+1]).square().mean();loss=rgb+.1*depth+.1*coverage
        if not torch.isfinite(loss):raise FloatingPointError((rid,mode,step))
        loss.backward()
        if step==0:
            gradients={n:float(t.grad.norm()) if t.grad is not None else None for n,t in params.items()}
            assert all(v is not None and np.isfinite(v) and v>0 for v in gradients.values()),gradients
        optimizer.step()
        with torch.no_grad():params['log_scales'].clamp_(min=float(torch.log(unit*1e-5)),max=float(torch.log(unit*2)))
        log.append(dict(step=step+1,view=v,loss=float(loss),rgb=float(rgb),depth=float(depth),coverage=float(coverage)))
    final={k:v.detach() for k,v in current().items()};final.update(unit=unit,source_c2w=source['c2w'],source_intrinsic=source['K'])
    save_tensor(folder/'coarse_geometry.pt',final);save_tensor(folder/'selected_indices.pt',dict(indices=indices))
    torch.cuda.synchronize()
    r=dict(complete=True,id=rid,mode=mode,configuration=CONFIG,source_only=True,queries_used=False,
        source_input_sha256=sha256(source_path(rid)),geometry_sha256=sha256(folder/'coarse_geometry.pt'),
        selected_indices_sha256=sha256(folder/'selected_indices.pt'),initial_reprojection_max_pixels=reprojection,
        initial_covariance_convention='camera R diag(squared scales) camera R^T, WXYZ quaternion',candidate=candidate,
        first_gradients=gradients,log=log,source_fit_seconds=time.monotonic()-begin,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
    write_json(marker,r);print(dict(source_fit_complete=rid,mode=mode,seconds=r['source_fit_seconds']),flush=True);return r

def main():
    a=argparse.ArgumentParser();a.add_argument('--pilot',action='store_true');args=a.parse_args()
    torch.set_num_threads(4);assert torch.cuda.is_available()
    p=parent_plan();rows=p['training_catalog'][:2] if args.pilot else p['training_catalog']+p['catalog']
    for meta in rows:
        prepare_source(meta,'training' if meta in p['training_catalog'] else 'development')
        source=load(source_path(meta['id']),'cuda');depths,valid,indices,candidate=sampled_inputs(source)
        for mode in MODES:fit_one(source,mode,depths,valid,indices,candidate)
        del source,depths,valid,indices;torch.cuda.empty_cache()

if __name__=='__main__':main()
