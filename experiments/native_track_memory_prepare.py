"""Useful native camera/track references and predicted-only memory geometry, three shards."""
import os,time,argparse,resource,signal
import torch
from torch.nn import functional as F
from experiments import native_track_memory_data as d
from models.native_track_memory import native_tokens,fmap,track
from experiments.native_pose_memory_data import mappose
from experiments.native_scene_exchange_data import scene_view
from experiments.camera_only_feature_data import source_geometry
from models.gaussian_scene_memory import contribution_weights

STOP=False
def stopped(*a):
    global STOP;STOP=True
def model_load(name):
    from experiments.gaussian_memory_native_run import model_load as original
    t=time.monotonic();m=original(d.plan());m.aggregator.patch_embed.cpu();m.point_head.cpu();torch.cuda.empty_cache();torch.cuda.synchronize();d.journal(name,kind='model_load',seconds=time.monotonic()-t,pretrained_large_hash=False);return m
def alignment(pose,sp,pp,index,dep):
    from streamvggt.utils.rotation import quat_to_mat
    R=quat_to_mat(pose[3:7]);c=-(R.T@pose[:3,None])[:,0];truth=sp['c2w'][index].cuda();A=truth[:3,:3]@R;sensor=pp['depth_sensor'][index].cuda();valid=pp['common_valid'][index].cuda()&torch.isfinite(dep)&(dep>1e-6)
    s=(sensor[valid]/dep[valid]).double().quantile(.5).float() if valid.any() else dep.new_tensor(float('nan'));ok=bool(torch.isfinite(s)&(s>0));s=s if ok else dep.new_tensor(1.)
    return dict(rotation=A,translation=truth[:3,3]-s*(A@c),scale=s,valid=ok),dict(valid=ok,scale=float(s),sensor_samples=int(valid.sum()),source_frame=d.SOURCES[index],query_truth_used=False)
def camera_head(m,tokens,state):
    with torch.autocast('cuda',enabled=False):poses,kv=m.camera_head(tokens,past_key_values_camera=list(state),use_cache=True)
    return poses[-1][0,0],kv
@torch.no_grad()
def local_view(cam,K):
    yy,xx=torch.meshgrid(torch.arange(32,device='cuda',dtype=torch.float64)+.5,torch.arange(32,device='cuda',dtype=torch.float64)+.5,indexing='ij');pix=torch.stack((xx,yy,torch.ones_like(xx)),-1).reshape(-1,3);kp=K.double().clone();kp[:2]/=14;p=(pix@torch.linalg.inv(kp).T).repeat_interleave(2,0);world=p@cam[:3,:3].double().T+cam[:3,3].double()
    fake=dict(means=world.float(),unit=torch.tensor(1.,device='cuda'),source_c2w=cam[None],has_content=torch.ones(2048,dtype=torch.bool,device='cuda'),rotations=torch.tensor([1.,0.,0.,0.],device='cuda').repeat(2048,1),scales=torch.ones(2048,3,device='cuda'),opacities=torch.zeros(2048,device='cuda'))
    with torch.device('cuda'):g,_=source_geometry(fake)
    v=scene_view(fake,g,dict(c2w=cam,K=K),torch.zeros(2048,device='cuda'));v['ordinary'][:,3:10]=0.;v['ordinary'][:,-1]=0.
    from models.gaussian_memory_native_pe import grid
    v['scene_pos']=grid(448,448,'cuda')[0,5:].repeat_interleave(2,0).float();return v
@torch.no_grad()
def prepare(m,meta,name):
    rid=meta['id'];mark=d.OUT/'preparation'/f'{rid}.json'
    if mark.exists():return
    sp=d.load(d.source(rid));pp=d.load(d.pair(rid));sourcepatches,sourcepatchpath=d.source_patches(rid);sourcepatches=sourcepatches.cuda();field=d.device(d.load(d.old.scene_path(rid)));g=d.device(d.load(d.fields_path(rid)));rows=[];begin=time.monotonic()
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    for ai,a in enumerate(d.SOURCES):
        if all(d.case(rid,a,q).exists() for q in d.QUERIES):continue
        t=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):anchor_tokens,kv=native_tokens(m,sourcepatches[ai:ai+1])
        d.journal(name,kind='anchor_native_prefix',id=rid,anchor=a)
        pose,camprefix=camera_head(m,anchor_tokens,[None]*m.camera_head.trunk_depth)
        with torch.autocast('cuda',enabled=False):dep=m.depth_head(anchor_tokens,torch.empty(1,1,3,448,448,device='cuda'),5)[0][0,0,...,0];anchor_fmap=fmap(m,anchor_tokens)
        align,adiag=alignment(pose,sp,pp,ai,dep);d.save(d.anchorfile(rid,a),d.cpu(dict(alignment=align,diagnostics=adiag,source_pose=pose,source_feature_reference=str(sourcepatchpath),frozen_source_only=True)));d.journal(name,kind='anchor_calibration_depth',id=rid,anchor=a,seconds=time.monotonic()-t)
        # Native4 is an independent causal prefix; original anchor remains first.
        prefix4=kv
        for j in range(4):
            if j==ai:continue
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):_,prefix4=native_tokens(m,sourcepatches[j:j+1],prefix4,1)
            d.journal(name,kind='Native4_additional_source',id=rid,anchor=a,source=d.SOURCES[j])
        for q in d.QUERIES:
            path=d.case(rid,a,q)
            if path.exists():continue
            started=time.monotonic();qp=d.load(d.old.query_path(rid,q));patches=d.load(d.old.patch_path(rid,q,'full'))['patches'].cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):qt,_=native_tokens(m,patches,kv,1)
            poseq,_=camera_head(m,qt,camprefix);_,K=pose_encoding_to_extri_intri(poseq[None,None],image_size_hw=(448,448));K=K[0,0];cam=mappose(poseq,align);d.journal(name,kind='query_native_initializer_and_Native2',id=rid,anchor=a,query=q)
            coords,vis,conf=track(m,anchor_fmap,qt,d.point_grid()[None].cuda());native2=dict(coords=torch.stack([c[0,1] for c in coords]),vis=vis[0,1],conf=conf[0,1]);del qt
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):qt4,_=native_tokens(m,patches,prefix4,4)
            coords,vis,conf=track(m,anchor_fmap,qt4,d.point_grid()[None].cuda());native4=dict(coords=torch.stack([c[0,1] for c in coords]),vis=vis[0,1],conf=conf[0,1]);d.journal(name,kind='Native4_query_and_track',id=rid,anchor=a,query=q);del qt4
            kp=K.clone();kp[:2]/=14;t0=time.monotonic();w,alpha=contribution_weights(field,cam[None],kp[None],32,32);mass=w[0].sum((0,1));d.journal(name,kind='actual_GS_raster',id=rid,anchor=a,query=q,seconds=time.monotonic()-t0);v=scene_view(field,g,dict(c2w=cam,K=K),mass);lv=local_view(cam,K);truth=d.labels(d.source210(pp,ai),d.qgeom(qp));torch.cuda.synchronize();d.journal(name,kind='predicted_GS_read_and_fields',id=rid,anchor=a,query=q,seconds=time.monotonic()-t0,zero_support=not bool(mass[field['has_content']].sum()>0))
            obj=dict(id=rid,anchor=a,query=q,inputs=d.cpu(dict(predicted_camera=cam,predicted_K=K,GS=v,L=lv,alignment_valid=align['valid'],zero_support=not bool(mass[field['has_content']].sum()>0))),labels=truth,Native2=d.cpu(native2),Native4=d.cpu(native4));d.save(path,obj)
            row=dict(id=rid,anchor=a,query=q,path=str(path),sha256=d.sha(path),bytes=path.stat().st_size,valid_GT=int(truth['valid'].sum()),alignment_valid=align['valid'],zero_support=obj['inputs']['zero_support'],seconds=time.monotonic()-started);rows.append(row);d.write(d.OUT/'prepared_pairs'/f'{rid}_{a}_{q}.json',row);print(dict(prepared=rid,anchor=a,query=q,GT=row['valid_GT']),flush=True)
            if STOP:return
        del anchor_tokens,kv,anchor_fmap,prefix4;torch.cuda.empty_cache()
    if len(list((d.OUT/'prepared_pairs').glob(f'{rid}_*.json')))==32:d.write(mark,dict(complete=True,id=rid,place=meta['place'],pairs=32,seconds=time.monotonic()-begin,field_reference=str(d.old.scene_path(rid)),source_only_geometry_reference=str(d.fields_path(rid))))
    print(dict(scene_prepared=rid),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--name',required=True);a=p.parse_args();torch.set_num_threads(2);torch.set_grad_enabled(False);signal.signal(signal.SIGTERM,stopped);d.init_config();start=time.monotonic()
    # Avoid the known memory-efficient LSE backward layout issue in later fits;
    # all references/preparation use this same official flash-or-math selection.
    torch.backends.cuda.enable_mem_efficient_sdp(False);m=model_load(a.name);print(dict(model_loaded=True,pid=os.getpid(),shard=a.shard),flush=True)
    for meta in d.metas()[a.shard::3]:
        prepare(m,meta,a.name)
        if STOP:break
    d.write(d.OUT/'worker_resources'/f'{a.name}.json',dict(phase='prepare',seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),completed=not STOP))
    if STOP:raise SystemExit(75)
if __name__=='__main__':main()
