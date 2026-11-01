"""Frozen actual-GS/context-only adapter. Target RGB is never opened here."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from experiments import round9_gs_readout as R9
from experiments.real_frozen_features import official_backbone
from models.round9_gaussian_readout import bilinear_readout
from models.round40_decoupled_pe import shape_descriptor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/round41_actual_gs'
WEIGHTS=Path('/home/liwt/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
SCENES=R9.SCENES
SEEDS=(11,22,33,44,55)
VAL_VIEWS=(3,9,15,21)
TRAIN_VIEWS=tuple(i for i in range(24) if i not in VAL_VIEWS)
SOURCES=('ROUND41_ACTUAL_GS_PROTOCOL.md','models/round41_gs_readout.py',
    'experiments/round41_gs_data.py','experiments/round41_gs_train.py',
    'experiments/round41_gs_audit.py','tests/test_round41_gs.py',
    'experiments/round9_gs_readout.py','experiments/round9_analysis.py',
    'experiments/round9_context_data.py','experiments/real_frozen_features.py',
    'models/round9_gaussian_readout.py','models/gaussian_quadratic_pe.py',
    'models/round40_decoupled_pe.py','models/round7_depth_attention.py')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def freeze(out):
    if out.exists():raise ValueError('preserve previous attempt')
    old=read(ROOT/'results/round9_gs_readout/lock.json')
    sources={str(ROOT/p):sha(ROOT/p) for p in SOURCES}
    for name,digest in old['source_hashes'].items():
        if sha(name)!=digest:raise ValueError('R9 source/runtime changed '+name)
        sources[name]=digest
    sources[str(WEIGHTS)]=sha(WEIGHTS)
    data={};identities={}
    for scene in SCENES:
        base=ROOT/'results/round9_gs_readout'/scene
        context=R9.DATA/scene
        tr=read(base/'training.json');run=read(base/'run.json');manifest=read(context/'manifest.json')
        for name,digest in ((base/'state.pt',tr['state_sha256']),
                (base/'gaussians.npz',tr['gaussians_sha256']),
                (base/'predictions.npz',run['prediction_sha256']),
                (base/'colors.npz',run['colors_sha256']),
                (context/'context.npz',run['context_archive_sha256']),
                (context/'target_cameras.json',run['target_cameras_sha256']),
                (context/'manifest.json',run['data_manifest_sha256'])):
            if sha(name)!=digest:raise ValueError('changed R9 evidence '+str(name))
            data[str(name)]=digest
        data[str(base/'training.json')]=sha(base/'training.json')
        data[str(base/'run.json')]=sha(base/'run.json')
        if tr['point_count']!=12000 or len(manifest['split']['context_ids'])!=24:raise ValueError('wrong cohort')
        identities[scene]=dict(context_ids=manifest['split']['context_ids'],
            target_ids=manifest['split']['target_ids'],normalization=manifest['normalization'])
    out.mkdir(parents=True)
    for p in SOURCES:
        dst=out/'source'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,dst)
    write(out/'lock.json',dict(sources=sources,data=data,identities=identities,
        unix=time.time(),target_rgb_read=False,scenes=SCENES,seeds=SEEDS))
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')


def checked(out):
    lock=read(out/'lock.json')
    if sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip():raise ValueError('changed lock')
    for name,digest in lock['sources'].items():
        if sha(name)!=digest:raise ValueError('changed source '+name)
    return lock


def pixels(device,dtype=torch.float32,h=144,w=192):
    y,x=torch.meshgrid(torch.arange(h,device=device,dtype=dtype),torch.arange(w,device=device,dtype=dtype),indexing='ij')
    return torch.stack((x,y),-1)


def sample(image,xy):
    """General-channel bilinear sample, coordinates refer to actual pixel centers."""
    h,w=image.shape[:2]
    grid=2*xy/xy.new_tensor([w-1,h-1])-1
    return F.grid_sample(image.permute(2,0,1)[None],grid[None,:,None],
        mode='bilinear',padding_mode='border',align_corners=True)[0,:,:,0].T


def pixel_support(depth,alpha,K):
    ok=torch.isfinite(depth)&(depth>0)&(alpha>=.5)
    if not ok.any():raise ValueError('no finite supported source depth')
    median=depth[ok].median()
    z=torch.where(ok,depth,median).clamp(1e-4,1e4)
    xy=pixels(z.device,z.dtype,*z.shape)
    ray=torch.cat((xy,torch.ones_like(z)[...,None]),-1)@torch.linalg.inv(K).T
    point=ray*z[...,None]
    dv,du=torch.gradient(point,dim=(0,1),edge_order=1)
    cov=(du[...,None]*du[...,None,:]+dv[...,None]*dv[...,None,:])/12.
    return point.reshape(-1,3),cov.reshape(-1,3,3),dict(invalid_pixels=int((~ok).sum()),
        median_depth=float(median),pixels=int(z.numel()))


@torch.no_grad()
def prepare(scene,out):
    lock=checked(out);dest=out/'prepared'/scene
    if dest.exists():raise ValueError('retain prepared attempt')
    dest.mkdir(parents=True);start=time.time()
    for name,digest in lock['data'].items():
        if '/'+scene+'/' in name and sha(name)!=digest:raise ValueError('changed data')
    base=ROOT/'results/round9_gs_readout'/scene
    state={k:v.cuda() for k,v in torch.load(base/'state.pt',weights_only=True).items()}
    context=R9.load_context(scene,'cuda:0')
    if context['ids'].tolist()!=lock['identities'][scene]['context_ids']:raise ValueError('context identity')
    model,backbone_metadata=official_backbone(WEIGHTS);model=model.cuda()
    cov=R9.covariance(state['log_scales'].double(),state['quaternions'].double())
    views=[];diagnostics=[]
    for i in range(24):
        K,R,t=(context[k][i].float() for k in ('K','R','t'))
        rgb=context['rgb'][i].float()
        xyz=state['means'].double()@R.double().T+t.double()
        xy,pc,z=R9.project(state['means'].double(),cov,K.double(),R.double(),t.double())
        rendered,alpha=R9.render(state,K,R,t,depth=True)
        depth=rendered[...,3]
        inside=(z>1e-4)&(xy[:,0]>=0)&(xy[:,0]<=191)&(xy[:,1]>=0)&(xy[:,1]<=143)
        geo=bilinear_readout(torch.stack((alpha,depth,depth),-1),xy,query_valid=inside)
        visible=inside&geo.valid&(geo.rgb[:,0]>=.5)&((z-geo.rgb[:,1]).abs()<=.02+.03*geo.rgb[:,1])
        x=(rgb.permute(2,0,1)[None]-rgb.new_tensor([.485,.456,.406])[None,:,None,None])/rgb.new_tensor([.229,.224,.225])[None,:,None,None]
        x=model.maxpool(model.relu(model.bn1(model.conv1(x))))
        x=model.layer2(model.layer1(x))
        if x.shape!=(1,128,18,24):raise ValueError('feature resolution')
        feature=F.normalize(F.interpolate(x,size=(144,192),mode='bilinear',align_corners=False),dim=1)[0].permute(1,2,0)
        center_feature=F.normalize(sample(feature,xy.float()),dim=-1)*visible[:,None]
        center_rgb=sample(rgb,xy.float())
        anchor,kcov,diag=pixel_support(depth,alpha,K)
        descriptor=shape_descriptor(kcov)
        views.append(dict(feature=feature.reshape(-1,128),rgb=rgb.reshape(-1,3),
            anchor=anchor,cov=kcov,xy=pixels(K.device).reshape(-1,2),
            q_anchor=xyz.float(),q_cov=(R.double()@cov@R.double().T).float(),
            q_xy=xy.float(),q_projected_cov=pc.float(),visible=visible,
            center_feature=center_feature,center_rgb=center_rgb))
        diagnostics.append(dict(index=i,id=int(context['ids'][i]),visible=int(visible.sum()),
            key_shape_std=descriptor.std(0).cpu().tolist(),**diag))
        print('PREPARE',scene,i+1,'visible',int(visible.sum()),flush=True)
    counts=torch.stack([v['visible'] for v in views]).sum(0)
    old=read(base/'run.json')
    if int((counts>0).sum())!=old['readable_primitive_count']:raise ValueError('R9 visibility population changed')
    payload=dict(views=[{k:v.cpu() for k,v in view.items()} for view in views],
        state={k:v.cpu() for k,v in state.items()},context_ids=context['ids'].cpu())
    torch.save(payload,dest/'scene.pt')
    write(dest/'completion.json',dict(scene=scene,path=str(dest/'scene.pt'),sha256=sha(dest/'scene.pt'),
        seconds=time.time()-start,views=diagnostics,readable=int((counts>0).sum()),
        primitives=12000,backbone=backbone_metadata,target_rgb_read=False,
        scene_units='R9 normalized context-camera extent, not known metres'))


def load_scene(scene,out,device='cuda:0'):
    checked(out);rec=read(out/'prepared'/scene/'completion.json')
    if sha(rec['path'])!=rec['sha256']:raise ValueError('changed prepared arrays')
    data=torch.load(rec['path'],map_location=device,weights_only=True)
    data['train_sum']=sum(data['views'][i]['center_feature'] for i in TRAIN_VIEWS)
    data['train_count']=sum(data['views'][i]['visible'].int() for i in TRAIN_VIEWS)
    data['all_sum']=sum(v['center_feature'] for v in data['views'])
    data['all_count']=sum(v['visible'].int() for v in data['views'])
    return data


def query(data,source,ids,model,target=None):
    v=data['views'][source]
    if target is None:
        total,count=data['all_sum'][ids],data['all_count'][ids]
    else:
        total,count=data['train_sum'][ids],data['train_count'][ids]
        if target in TRAIN_VIEWS:
            total=total-data['views'][target]['center_feature'][ids]
            count=count-data['views'][target]['visible'][ids].int()
    feature=total/count.clamp_min(1)[:,None]
    return dict(feature=F.normalize(feature,dim=-1),anchor=v['q_anchor'][ids],
        cov=v['q_cov'][ids],shuffle_cov=v['q_cov'][model.q_perm[ids]],
        xy=v['q_xy'][ids],projected_cov=v['q_projected_cov'][ids])


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','prepare'))
    p.add_argument('--out',type=Path,default=OUT/'attempt1');p.add_argument('--scene',choices=SCENES)
    a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.stage=='freeze':freeze(a.out)
    elif a.scene:prepare(a.scene,a.out)
    else:p.error('--scene required')


if __name__=='__main__':main()
