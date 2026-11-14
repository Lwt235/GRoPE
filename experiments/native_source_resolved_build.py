"""One training-source PCA and44 original-alpha bindings; no native predictions."""
import argparse,time,resource,signal
import torch
from experiments import native_source_resolved_data as d
from models.gaussian_scene_memory import contribution_weights
STOP=False
def stopped(*a):
    global STOP;STOP=True
@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args();d.init_config();torch.set_num_threads(2);signal.signal(signal.SIGTERM,stopped);start=time.monotonic();torch.cuda.init();pca_path=d.CACHE/'PCA255.pt'
    if not pca_path.exists():
        t=time.monotonic();s=torch.zeros(1024,device='cuda',dtype=torch.float64);gram=torch.zeros(1024,1024,device='cuda',dtype=torch.float64);n=0
        for meta in d.plan()['training_catalog']:
            x,path=d.source_patches(meta['id']);x=x.reshape(-1,1024).cuda().double();s+=x.sum(0);gram+=x.T@x;n+=len(x)
        mu=s/n;cov=(gram-n*mu[:,None]*mu[None,:])/(n-1);e,U=torch.linalg.eigh(cov);U=U[:,-255:].flip(1).contiguous();torch.cuda.synchronize();obj=dict(mean=mu.cpu(),basis=U.cpu(),eigenvalues=e.flip(0).cpu(),training_ids=[m['id'] for m in d.plan()['training_catalog']],source_ids=list(d.SOURCES),source_tokens=n,query_inputs_used=False,whitening=False,science_sha256=d.SCIENCE_SHA);d.save(pca_path,obj);restored=d.load(pca_path);assert torch.equal(restored['basis'],obj['basis']) and torch.equal(restored['mean'],obj['mean']);d.write(d.OUT/'PCA.json',dict(complete=True,path=str(pca_path),sha256=d.sha(pca_path),source_only=True,training_places=32,source_tokens=n,dimensions=255,total_variance=float(e.sum()),retained_variance_fraction=float(e[-255:].sum()/e.sum()),seconds=time.monotonic()-t));d.journal(a.name,kind='global_source_only_PCA',seconds=time.monotonic()-t,source_tokens=n)
    pca=d.load(pca_path);mu=pca['mean'].cuda();U=pca['basis'].cuda()
    for meta in d.metas():
        rid=meta['id'];marker=d.OUT/'binding'/f'{rid}.json'
        if marker.exists():continue
        t=time.monotonic();field=d.device(d.load(d.old.scene_path(rid)));sp=d.load(d.source(rid));K=sp['K'].cuda().clone();K[:,:2]/=14
        x,path=d.source_patches(rid);x=x.cuda().double();w,_=contribution_weights(field,sp['c2w'].cuda(),K,32,32);w=w.reshape(4,1024,2048);torch.cuda.synchronize();d.journal(a.name,kind='actual_source_alpha_raster',id=rid,views=4)
        mass_v=w.sum(1).double().T;mass=mass_v.sum(1);safe=mass.clamp_min(1e-8);valid=field['has_content'];p=mass_v/safe[:,None];z=torch.stack([w[v].double().T@((x[v]-mu)@U) for v in range(4)],1)/safe[:,None,None];z[~valid]=0;p[~valid]=0;zi=z.sum(1)
        F=torch.cat((z,p[:,:,None]),-1).reshape(2048,1024).half().cpu().contiguous();M=torch.cat((p[:,:,None]*zi[:,None,:],p[:,:,None]),-1).reshape(2048,1024).half().cpu().contiguous()
        assert torch.isfinite(F).all() and torch.isfinite(M).all()
        for rep,content in [('F',F),('M',M)]:d.save(d.state(rid,rep),dict(content=content,has_content=valid.cpu(),representation=rep,source_ids=list(d.SOURCES),PCA_reference=str(pca_path),original_geometry_reference=str(d.old.scene_path(rid)),source_only=True,science_sha256=d.SCIENCE_SHA));assert torch.equal(d.load(d.state(rid,rep))['content'],content)
        def decoded(v):return v.float().reshape(2048,4,256)[:,:,:255].sum(1).double()@pca['basis'].T+pca['mean']
        delta=decoded(F)-decoded(M);oldmass=field.get('mass');r=dict(complete=True,id=rid,source_only=True,queries_or_labels_used=False,PCA_reference=str(pca_path),original_field=str(d.old.scene_path(rid)),source_features_reference=str(path),original_has_content_preserved=True,valid_content=int(valid.sum()),zero_total_mass=int((mass<=1e-8).sum()),source_mass_sum_max_abs_error=float((p[valid].sum(1)-1).abs().max()) if valid.any() else None,FP16_decoded_F_minus_M_max_abs=float(delta.abs().max()),FP16_decoded_F_minus_M_RMS=float(delta.square().mean().sqrt()),files=[dict(path=str(d.state(rid,rep)),sha256=d.sha(d.state(rid,rep)),bytes=d.state(rid,rep).stat().st_size) for rep in ['F','M']],seconds=time.monotonic()-t)
        d.write(marker,r);d.journal(a.name,kind='source_binding',id=rid,seconds=r['seconds']);print(dict(bound=rid,seconds=r['seconds'],decoded_F_M_max=r['FP16_decoded_F_minus_M_max_abs']),flush=True)
        del field,sp,x,w,z,p,F,M
        if STOP:raise SystemExit(75)
    d.write(d.OUT/'build_complete.json',dict(complete=True,scenes=44,PCA=str(pca_path),source_only=True,actual_new_source_rasters=44,model_or_query_calls=0));d.write(d.OUT/'worker_resources'/f'{a.name}.json',dict(phase='build',wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),completed=True));print('PCA and44source bindings complete',flush=True)
if __name__=='__main__':main()
