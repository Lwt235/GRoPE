"""Source-camera registration and one bounded RGB/depth/coverage coarse fit."""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from experiments.gaussian_scene_data import OUT, CACHE, input_plan, source_path, read
from experiments.gaussian_scene_setup import write
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_incremental_data import save_tensor
from models.gaussian_scene_memory import rasterize_field

PILOT_CONFIG = dict(slots=2048, steps=300, fit_size=224, source_opacity_min=.01,
                    minimum_scale_fraction=1e-5, maximum_scale_fraction=2.,
                    means_lr_unit_fraction=.001, log_scales_lr=.02,
                    rotations_lr=.001, opacity_logits_lr=.02, color_logits_lr=.02,
                    rgb_l1_weight=1., log_depth_l1_weight=.1, coverage_mse_weight=.1,
                    view_schedule='step modulo four', query_data_used=False)


def register_source_cameras(predicted, given):
    """Rotation mean and median paired-baseline scale; uses source cameras only."""
    a, b = predicted.double(), given.double()
    cross = (b[:, :3, :3] @ a[:, :3, :3].transpose(-1, -2)).sum(0)
    u, _, vh = torch.linalg.svd(cross)
    sign = torch.eye(3, dtype=torch.float64)
    sign[-1, -1] = torch.det(u @ vh)
    rotation = u @ sign @ vh
    i, j = torch.triu_indices(len(a), len(a), offset=1)
    da = (a[i, :3, 3]-a[j, :3, 3]).norm(dim=-1)
    db = (b[i, :3, 3]-b[j, :3, 3]).norm(dim=-1)
    valid = (da > 1e-8) & (db > 1e-8)
    if not bool(valid.any()):
        raise ValueError('Source camera scale is unobservable from nonzero baselines')
    scale = torch.median(db[valid]/da[valid])
    translation = b[:, :3, 3].mean(0)-scale*rotation@a[:, :3, 3].mean(0)
    aligned_center = scale*(a[:, :3, 3]@rotation.T)+translation
    residual = aligned_center-b[:, :3, 3]
    aligned_rotation = rotation[None]@a[:, :3, :3]
    cosine = ((aligned_rotation.transpose(-1, -2)@b[:, :3, :3]).diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2
    diagnostics = dict(scale=float(scale), source_center_rmse_m=float(residual.square().sum(-1).mean().sqrt()),
                       source_rotation_mean_degrees=float(torch.rad2deg(torch.acos(cosine.clamp(-1,1))).mean()),
                       registration='mean source orientation + median nonzero source baseline ratio; no query pose/depth used')
    return scale.float(), rotation.float(), translation.float(), diagnostics


def quaternion_product(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw*bw-(av*bv).sum(-1, keepdim=True),
                      aw*bv+bw*av+torch.linalg.cross(av.expand_as(bv),bv)), -1)


def registered_field(raw, source):
    scale, rotation, translation, diagnostics = register_source_cameras(raw['pred_c2w'], source['c2w'])
    qxyzw = Rotation.from_matrix(rotation.numpy()).as_quat()
    q = torch.tensor(qxyzw[[3,0,1,2]], dtype=torch.float32)[None]
    field = dict(means=scale*(raw['means']@rotation.T)+translation,
                 rotations=F.normalize(quaternion_product(q,raw['rotations']),dim=-1),
                 scales=scale*raw['scales'], opacities=raw['opacities'],
                 colors=(.5+.28209479177387814*raw['harmonics'][...,0]).clamp(0,1))
    unit = scale*raw['source_depth'].median()
    return field, unit, (scale,rotation,translation), diagnostics


@torch.no_grad()
def coverage_sample(field, count):
    indices = torch.where(field['opacities'] >= PILOT_CONFIG['source_opacity_min'])[0]
    if len(indices) < count:
        raise ValueError('Fewer non-negligible source Gaussians than fixed slot count')
    centers = field['means'][indices]
    distance = torch.full((len(indices),), float('inf'), device=centers.device)
    chosen = torch.empty(count, device=centers.device, dtype=torch.long)
    current = field['opacities'][indices].argmax()
    for i in range(count):
        chosen[i] = current
        torch.minimum(distance,(centers-centers[current]).square().sum(-1),out=distance)
        current = distance.argmax()
    selected = indices[chosen]
    return {k:v[selected].contiguous() for k,v in field.items()}, selected


def fit_sources(raw, source, folder):
    field, unit, transform, diagnostics = registered_field(raw,source)
    field = {k:v.cuda() for k,v in field.items()}
    unit = unit.cuda()
    c2w = source['c2w'].cuda()
    K = source['K'].cuda().clone()
    K[:, :2] *= PILOT_CONFIG['fit_size']/448
    images = F.interpolate(source['rgb'].cuda().float()/255.,size=(224,224),mode='area')
    target_depth = F.interpolate(raw['source_depth'].permute(0,3,1,2).cuda(),size=(224,224),mode='nearest')[:,0]*transform[0].cuda()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    with torch.no_grad():
        _, full_alpha, _ = rasterize_field(field,c2w,K,224,224)
        initial, selected = coverage_sample(field,PILOT_CONFIG['slots'])
    save_tensor(folder/'coarse_initial.pt',{k:v.detach().cpu() for k,v in initial.items()})
    parameters = dict(means=torch.nn.Parameter(initial['means'].clone()),
                      rotations=torch.nn.Parameter(initial['rotations'].clone()),
                      log_scales=torch.nn.Parameter(initial['scales'].clamp_min(float(unit)*1e-5).log()),
                      opacity_logits=torch.nn.Parameter(torch.logit(initial['opacities'].clamp(.0001,.9999))),
                      color_logits=torch.nn.Parameter(torch.logit(initial['colors'].clamp(.0001,.9999))))
    groups = [dict(params=[parameters[name]],lr=lr) for name,lr in (
        ('means',.001*float(unit)),('rotations',.001),('log_scales',.02),
        ('opacity_logits',.02),('color_logits',.02))]
    optimizer = torch.optim.Adam(groups)
    def current():
        return dict(means=parameters['means'],rotations=F.normalize(parameters['rotations'],dim=-1),
                    scales=parameters['log_scales'].exp(),opacities=parameters['opacity_logits'].sigmoid(),
                    colors=parameters['color_logits'].sigmoid())
    log=[]
    for step in range(PILOT_CONFIG['steps']):
        view=step%4
        optimizer.zero_grad(set_to_none=True)
        prediction,alpha,_=rasterize_field(current(),c2w[view:view+1],K[view:view+1],224,224,mode='RGB+ED')
        rgb_loss=(prediction[0,...,:3].permute(2,0,1)-images[view]).abs().mean()
        covered=alpha[0,...,0].detach()>.0001
        depth=prediction[0,...,3].clamp_min(1e-6)
        depth_loss=(depth[covered].log()-target_depth[view][covered].clamp_min(1e-6).log()).abs().mean() if covered.any() else depth.sum()*0
        coverage_loss=(alpha-full_alpha[view:view+1]).square().mean()
        loss=rgb_loss+.1*depth_loss+.1*coverage_loss
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite source-only fit')
        loss.backward()
        if step==0:
            gradients={k:None if v.grad is None else float(v.grad.norm()) for k,v in parameters.items()}
            assert all(v is not None and np.isfinite(v) and v>0 for v in gradients.values()),gradients
        optimizer.step()
        with torch.no_grad():parameters['log_scales'].clamp_(min=float(torch.log(unit*1e-5)),max=float(torch.log(unit*2)))
        log.append(dict(step=step+1,view=view,loss=float(loss),rgb=float(rgb_loss),depth=float(depth_loss),coverage=float(coverage_loss)))
    torch.cuda.synchronize()
    seconds=time.perf_counter()-begin
    result={k:v.detach().cpu().contiguous() for k,v in current().items()}
    result.update(unit=unit.detach().cpu(),source_c2w=source['c2w'],source_intrinsic=source['K'])
    save_tensor(folder/'coarse_geometry.pt',result)
    save_tensor(folder/'source_registration.pt',dict(scale=transform[0],rotation=transform[1],translation=transform[2],selected_indices=selected.cpu()))
    write(folder/'source_fit.json',dict(complete=True,configuration=PILOT_CONFIG,registration=diagnostics,
          source_only=True,queries_used=False,source_count=4,slots=2048,first_gradients=gradients,
          log=log,source_build_seconds=seconds,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
          initial_geometry_sha256=sha256(folder/'coarse_initial.pt'),geometry_sha256=sha256(folder/'coarse_geometry.pt')))
    print({'coarse_source_fit_complete':source['id'],'seconds':seconds,'registration':diagnostics},flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);assert torch.cuda.is_available()
    p=input_plan();catalog=p['training_catalog'][:2] if args.pilot else p['training_catalog']+p['catalog']
    write(OUT/'source_fit_configuration.json',dict(status='pilot configuration; lock with complete reader/resource check before formal use',configuration=PILOT_CONFIG))
    for meta in catalog:
        folder=CACHE/meta['id'];marker=folder/'source_fit.json'
        if marker.exists() and read(marker)['complete']:continue
        source=torch.load(source_path(meta['id']),map_location='cpu',weights_only=False)
        raw=torch.load(folder/'anysplat_source_init.pt',map_location='cpu',weights_only=False)
        fit_sources(raw,source,folder)
        del raw,source;torch.cuda.empty_cache()


if __name__=='__main__':main()
