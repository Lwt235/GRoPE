"""Context-only GS fitting and fixed-geometry observed-color readout.

Target RGB is never opened here. Cameras are externally supplied calibration.
Run with the already installed MiniColmap venv; no extension is built or changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from models.round9_gaussian_readout import gaussian_readout, bilinear_readout

ROOT=Path(__file__).resolve().parents[1]
EXTERNAL=Path('/home/liwt/miniColmap')
DATA=ROOT/'results/round9_context_data'
SCENES=('teatime','waldo_kitchen')
WIDTH,HEIGHT=192,144
BANDWIDTHS=(.25,.5,1.,2.)
ROLES=('full','isotropic','fixed','point')
METHODS=tuple(f'{r}_h{h:g}' for r in ROLES for h in BANDWIDTHS)+('bilinear','original_gs')
STEPS=3000
SOURCES=('ROUND9_PROTOCOL.md','experiments/round9_gs_readout.py',
    'experiments/round9_context_data.py','experiments/round9_analysis.py',
    'models/round9_gaussian_readout.py','models/gaussian_quadratic_pe.py')


def sha(path):
    with Path(path).open('rb') as f:
        h=hashlib.sha256()
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False))


def runtime():
    if str(EXTERNAL/'src') not in sys.path:sys.path.insert(0,str(EXTERNAL/'src'))
    from minicolmap.rendering.gsplat_runtime import load_project_extension
    load_project_extension()
    from gsplat import rasterization
    return rasterization


def covariance(log_scales,quaternions):
    q=quaternions/torch.linalg.vector_norm(quaternions,dim=-1,keepdim=True).clamp_min(1e-12)
    w,x,y,z=q.unbind(-1)
    r=torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(-1,3,3)
    return (r*log_scales.exp().square()[:,None,:])@r.transpose(-1,-2)


def render(state,K,R,t,colors=None,depth=False):
    rasterization=runtime()
    k=K.clone();k[0,2]+=.5;k[1,2]+=.5
    view=torch.eye(4,dtype=K.dtype,device=K.device);view[:3,:3]=R;view[:3,3]=t
    rgb=state['color_logits'].sigmoid() if colors is None else colors
    result,alpha,info=rasterization(means=state['means'].contiguous(),
        quats=state['quaternions'].contiguous(),scales=state['log_scales'].exp().contiguous(),
        opacities=state['opacity_logits'].sigmoid().contiguous(),colors=rgb.contiguous(),
        viewmats=view[None],Ks=k[None],width=WIDTH,height=HEIGHT,
        backgrounds=K.new_zeros((1,3)),packed=False,sh_degree=None,
        eps2d=.3,near_plane=1e-4,far_plane=1e6,
        rasterize_mode='classic',render_mode='RGB+ED' if depth else 'RGB')
    return result[0],alpha[0,...,0]


@torch.no_grad()
def split_primitives(state,limit=12000):
    """Fixed-budget context-only growth, shared by every readout comparison."""
    n=len(state['means']);count=min(n,limit-n)
    if count<=0:return state,0
    scales=state['log_scales'].exp()
    score=scales.amax(-1)*state['opacity_logits'].sigmoid()
    selected=torch.argsort(score,descending=True,stable=True)[:count]
    keep=torch.ones(n,dtype=torch.bool,device=scales.device);keep[selected]=False
    retained=torch.where(keep)[0];indices=torch.cat((retained,selected.repeat_interleave(2)))
    revised={k:v[indices].clone() for k,v in state.items()}
    values,vectors=torch.linalg.eigh(covariance(state['log_scales'][selected],state['quaternions'][selected]))
    offset=.5*values[:,-1].sqrt()[:,None]*vectors[:,:,-1]
    offset=torch.stack((-offset,offset),1).reshape(-1,3)
    revised['means'][len(retained):]+=offset
    revised['log_scales'][len(retained):]-=np.log(1.6)
    return {k:torch.nn.Parameter(v) for k,v in revised.items()},count


def project(means,cov,K,R,t):
    xyz=means@R.T+t;z=xyz[:,2];safe=z.clamp_min(1e-6)
    xy=(xyz@K.T)[:,:2]/safe[:,None]
    j=means.new_zeros((len(means),2,3))
    j[:,0,0]=K[0,0]/safe;j[:,0,2]=-K[0,0]*xyz[:,0]/safe.square()
    j[:,1,1]=K[1,1]/safe;j[:,1,2]=-K[1,1]*xyz[:,1]/safe.square()
    jw=j@R;c=jw@cov@jw.transpose(-1,-2)
    c=(c+c.transpose(-1,-2))*.5
    return xy,c,z


def source_hashes():
    result={str(ROOT/p):sha(ROOT/p) for p in SOURCES}
    for p in (EXTERNAL/'src/minicolmap/rendering/gsplat_runtime.py',
            EXTERNAL/'.cache/gsplat-runtime.json',EXTERNAL/'.cache/torch_extensions/gsplat_cuda/gsplat_cuda.so'):
        result[str(p)]=sha(p)
    import gsplat
    gp=Path(gsplat.__file__).parent
    for rel in ('rendering.py','cuda/_wrapper.py','cuda/_backend.py'):
        result[str(gp/rel)]=sha(gp/rel)
    return result


def data_hashes():
    return {str(DATA/s/f):sha(DATA/s/f) for s in SCENES
        for f in ('context.npz','manifest.json','target_cameras.json')}


def freeze(out):
    if out.exists():raise ValueError('refuse overwrite '+str(out))
    for name,key in [('round6_confirmation','sourcehash'),('round7_depth_confirmation','source_hashes'),('round8_oracle_development','source_hashes')]:
        prior=json.loads((ROOT/'results'/name/'lock.json').read_text())
        for p,h in prior[key].items():
            if sha(ROOT/p)!=h:raise ValueError('prior frozen source changed '+p)
    sources=source_hashes();out.mkdir(parents=True)
    for rel in SOURCES:
        dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    lock=dict(source_hashes=sources,data_hashes=data_hashes(),scenes=SCENES,steps=STEPS,
        geometry_seed=11,roles=ROLES,bandwidths=BANDWIDTHS,methods=METHODS,
        context_images_per_scene=24,target_images_per_scene=8,
        target_rgb_read_authorized=False,fresh_scene_confirmation_authorized=False,
        calibration_condition='K/R/t supplied from existing all-image SfM; no full-pipeline unseen-view claim',
        claim='context RGB/feature matching/triangulation/GS fitting exclude target observations')
    write(out/'lock.json',lock);print('FROZEN',sha(out/'lock.json'),flush=True)


def check_lock(out):
    lock=json.loads((out/'lock.json').read_text())
    if source_hashes()!=lock['source_hashes'] or data_hashes()!=lock['data_hashes']:
        raise ValueError('frozen source/data changed')
    return lock


def load_context(scene,device):
    with np.load(DATA/scene/'context.npz',allow_pickle=False) as d:
        data={k:torch.from_numpy(d[k].copy()).to(device) for k in d.files}
    k=data['K']
    if not (torch.all(k[:,0,1]==0) and torch.all(k[:,1,0]==0)
            and torch.all(k[:,2,:]==k.new_tensor([0.,0.,1.]))):
        raise ValueError('pinhole renderer/projector requires standard zero-skew K')
    return data


def train(scene,out,device):
    lock=check_lock(out);dest=out/scene
    if dest.exists():raise ValueError('refuse overwrite '+str(dest))
    dest.mkdir();data=load_context(scene,device)
    torch.manual_seed(11);torch.cuda.manual_seed_all(11);rng=np.random.default_rng(11)
    points=data['points'].float();colors=data['colors'].float().clamp(.001,.999)
    distances=cKDTree(points.cpu().numpy()).query(points.cpu().numpy(),k=min(4,len(points)))[0][:,1:]
    scales=np.sqrt(np.mean(distances**2,-1)).clip(1e-4,.04)
    quaternion=points.new_zeros((len(points),4));quaternion[:,0]=1
    state={k:torch.nn.Parameter(v) for k,v in dict(means=points.clone(),
        log_scales=torch.as_tensor(np.log(scales),device=device,dtype=points.dtype)[:,None].repeat(1,3),
        quaternions=quaternion,color_logits=torch.logit(colors),
        opacity_logits=points.new_full((len(points),),float(np.log(.1/.9)))).items()}
    lrs={'means':.00016,'log_scales':.005,'quaternions':.001,'color_logits':.025,'opacity_logits':.05}
    opt=torch.optim.Adam([{'params':[state[k]],'lr':v,'name':k} for k,v in lrs.items()],eps=1e-15)
    history=[];topology=[];batch_hash=hashlib.sha256();start=time.time()
    for step in range(1,STEPS+1):
        i=int(rng.integers(len(data['ids'])));batch_hash.update(np.asarray(i,dtype='<i8').tobytes())
        for group in opt.param_groups:
            if group['name']=='means':group['lr']=lrs['means']*.01**(step/STEPS)
        opt.zero_grad(set_to_none=True)
        pred,_=render(state,data['K'][i].float(),data['R'][i].float(),data['t'][i].float())
        loss=(pred-data['rgb'][i]).square().mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite context training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(list(state.values()),10,error_if_nonfinite=True);opt.step()
        with torch.no_grad():
            state['log_scales'].clamp_(np.log(1e-5),np.log(.2))
            state['opacity_logits'].clamp_(-10,10);state['color_logits'].clamp_(-10,10)
            state['quaternions'].div_(state['quaternions'].norm(dim=-1,keepdim=True).clamp_min(1e-12))
        if step in (500,1000,1500,2000):
            state,split_count=split_primitives(state)
            if split_count:
                opt=torch.optim.Adam([{'params':[state[k]],'lr':v,'name':k} for k,v in lrs.items()],eps=1e-15)
            topology.append(dict(step=step,split_parents=split_count,primitives=len(state['means']),optimizer_state='reset after topology change'))
        if step==1 or step%300==0:
            history.append(dict(step=step,context_image_id=int(data['ids'][i]),mse=float(loss.detach())))
            print('TRAIN',scene,step,'MSE',float(loss.detach()),'seconds',round(time.time()-start,1),flush=True)
    state={k:v.detach() for k,v in state.items()};train_metrics=[]
    with torch.no_grad():
        for i in range(len(data['ids'])):
            pred,alpha=render(state,data['K'][i].float(),data['R'][i].float(),data['t'][i].float())
            mse=float((pred-data['rgb'][i]).square().mean());train_metrics.append(dict(id=int(data['ids'][i]),mse=mse,psnr=-10*np.log10(mse),mean_alpha=float(alpha.mean())))
    torch.save({k:v.cpu() for k,v in state.items()},dest/'state.pt')
    cov=covariance(state['log_scales'],state['quaternions'])
    np.savez_compressed(dest/'gaussians.npz',means=state['means'].cpu().numpy(),
        covariances=cov.cpu().numpy(),scales=state['log_scales'].exp().cpu().numpy(),
        quaternions=state['quaternions'].cpu().numpy(),colors=state['color_logits'].sigmoid().cpu().numpy(),
        opacities=state['opacity_logits'].sigmoid().cpu().numpy())
    record=dict(scene=scene,seed=11,steps=STEPS,initial_point_count=len(points),point_count=len(state['means']),initialization='context-only newly matched and triangulated points',
        color_representation='single sigmoid RGB per primitive (view-independent, SH0 capacity)',
        topology=topology,batch_sequence_sha256=batch_hash.hexdigest(),history=history,
        context_metrics=train_metrics,seconds=time.time()-start,source_hashes=lock['source_hashes'],
        data_hashes=lock['data_hashes'],state_sha256=sha(dest/'state.pt'),gaussians_sha256=sha(dest/'gaussians.npz'),
        lock_sha256=sha(out/'lock.json'),target_rgb_read=False,checkpoint_selection='fixed final step, no validation')
    write(dest/'training.json',record);check_lock(out)
    print('TRAIN_DONE',scene,len(state['means']),'mean context PSNR',np.mean([r['psnr'] for r in train_metrics]),flush=True)


@torch.no_grad()
def readout(scene,out,device):
    lock=check_lock(out);dest=out/scene
    if (dest/'predictions.npz').exists():raise ValueError('refuse overwrite predictions')
    training=json.loads((dest/'training.json').read_text())
    if sha(dest/'state.pt')!=training['state_sha256']:raise ValueError('checkpoint changed')
    state={k:v.to(device) for k,v in torch.load(dest/'state.pt',weights_only=True).items()}
    data=load_context(scene,device);n=len(state['means']);start=time.time()
    cov=covariance(state['log_scales'].double(),state['quaternions'].double())
    sums=torch.zeros((len(METHODS)-1,n,3),dtype=torch.float64,device=device)
    counts=torch.zeros(n,dtype=torch.int64,device=device);records=[];audits=[]
    for i in range(len(data['ids'])):
        K,R,t=(data[k][i].float() for k in ('K','R','t'))
        xy,c,z=project(state['means'].double(),cov,K.double(),R.double(),t.double())
        rendered,alpha=render(state,K,R,t,depth=True)
        depth=rendered[...,3]
        geometry_image=torch.stack((alpha,depth,depth),-1)
        inside=(z>1e-4)&(xy[:,0]>=0)&(xy[:,0]<=WIDTH-1)&(xy[:,1]>=0)&(xy[:,1]<=HEIGHT-1)
        sample=bilinear_readout(geometry_image,xy,query_valid=inside)
        visible=inside&sample.valid&(sample.rgb[:,0]>=.5)&((z-sample.rgb[:,1]).abs()<=.02+.03*sample.rgb[:,1])
        ids=torch.where(visible)[0];counts[ids]+=1
        pxy=xy[ids].double();pcov=c[ids].double();image=data['rgb'][i].double()
        # Add only a fixed 1e-10 pixel^2 numerical floor to the symmetrized
        # double precision physical covariance before applying the controls.
        pcov=pcov+torch.eye(2,dtype=pcov.dtype,device=device)*1e-10
        mi=0
        for role in ROLES:
            for h in BANDWIDTHS:
                value=gaussian_readout(image,pxy,pcov,method=role,bandwidth=h,
                    fixed_variance=1.,compute_dtype=torch.float64,query_chunk_size=128,key_chunk_size=WIDTH*HEIGHT)
                if not value.valid.all():raise ValueError('shared visibility produced empty readout')
                sums[mi,ids]+=value.rgb;mi+=1
        value=bilinear_readout(image,pxy)
        if not value.valid.all():raise ValueError('bilinear support differs')
        sums[mi,ids]+=value.rgb
        # Small complete source-image operator samples are saved for an
        # independent NumPy EWA reconstruction, not used as labels.
        select=torch.linspace(0,max(0,len(ids)-1),min(8,len(ids)),device=device).long()
        if len(select):
            samplefull=gaussian_readout(image,pxy[select],pcov[select],method='full',bandwidth=.25,compute_dtype=torch.float64)
            audits.append(dict(source_id=int(data['ids'][i]),primitive_ids=ids[select].cpu().tolist(),
                mean=pxy[select].cpu().tolist(),covariance=pcov[select].cpu().tolist(),
                full_h025=samplefull.rgb.cpu().tolist()))
        records.append(dict(id=int(data['ids'][i]),visible_primitive_count=len(ids),
            mean_source_alpha=float(alpha.mean())))
        print('READOUT',scene,i+1,len(ids),'seconds',round(time.time()-start,1),flush=True)
    original=state['color_logits'].sigmoid();valid=counts>0
    colors=torch.where(valid[None,:,None],sums/counts.clamp_min(1)[None,:,None],original.double()[None]).float()
    colors=torch.cat((colors,original[None]),0)
    targetdoc=json.loads((DATA/scene/'target_cameras.json').read_text());frames=targetdoc['frames']
    predictions=[];target_alpha=[];fallback_weight=[]
    for frame in frames:
        K,R,t=(torch.tensor(frame[k],dtype=torch.float32,device=device) for k in ('K','R','t'))
        preds=[]
        for col in colors:
            pred,alpha=render(state,K,R,t,colors=col)
            preds.append(pred.cpu().numpy())
        predictions.append(np.stack(preds));target_alpha.append(alpha.cpu().numpy())
        # Alpha-composited contribution of never-readable primitives.
        contribution,_=render(state,K,R,t,colors=(~valid).float()[:,None].expand(-1,3).contiguous())
        fallback_weight.append(contribution[...,0].cpu().numpy())
    np.savez_compressed(dest/'colors.npz',colors=colors.cpu().numpy(),source_view_counts=counts.cpu().numpy(),methods=np.array(METHODS))
    np.savez_compressed(dest/'predictions.npz',predictions=np.stack(predictions,axis=1),methods=np.array(METHODS),
        target_ids=np.array([f['id'] for f in frames]),target_splits=np.array([f['split'] for f in frames]),
        alpha=np.stack(target_alpha),fallback_contribution=np.stack(fallback_weight))
    write(dest/'operator_samples.json',audits)
    record=dict(scene=scene,methods=METHODS,visibility='center in closed pixel-center rectangle; GS expected-depth approximation; alpha>=.5 and abs(z-D)<=.02+.03D in normalized context-camera units',
        views=records,point_count=n,readable_primitive_count=int(valid.sum()),fallback_primitive_count=int((~valid).sum()),
        source_view_count_quantiles=np.quantile(counts.cpu().numpy(),[0,.25,.5,.75,1]).tolist(),
        all_pixel_softmax=True,readout_compute_dtype='float64',shared_cross_view_fusion='equal average of independently normalized visible source views',
        fallback='context-trained original RGB',target_rgb_read=False,
        prediction_sha256=sha(dest/'predictions.npz'),colors_sha256=sha(dest/'colors.npz'),
        training_sha256=sha(dest/'training.json'),operator_samples_sha256=sha(dest/'operator_samples.json'),
        data_manifest_sha256=sha(DATA/scene/'manifest.json'),target_cameras_sha256=sha(DATA/scene/'target_cameras.json'),
        context_archive_sha256=sha(DATA/scene/'context.npz'),
        source_hashes=lock['source_hashes'],data_hashes=lock['data_hashes'],lock_sha256=sha(out/'lock.json'),seconds=time.time()-start)
    write(dest/'run.json',record);check_lock(out);print('READOUT_DONE',scene,record['seconds'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','train','readout'])
    p.add_argument('--out',type=Path,default=ROOT/'results/round9_gs_readout')
    p.add_argument('--scene',choices=SCENES);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.action=='freeze':freeze(a.out)
    elif a.scene is None:p.error('--scene required')
    elif a.action=='train':train(a.scene,a.out,a.device)
    else:readout(a.scene,a.out,a.device)


if __name__=='__main__':main()
