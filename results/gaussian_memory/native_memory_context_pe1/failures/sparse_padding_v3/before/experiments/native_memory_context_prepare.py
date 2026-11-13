"""FP64 source-only canonical graph and original compositing matrices."""
import argparse
import time
from experiments.native_memory_context_data import *
from experiments.native_render_parameter_prepare import rotation,svec
from models.gaussian_scene_memory import contribution_weights

@torch.no_grad()
def source_graph(field):
    # Every datum comes from saved source state; no query is accepted.
    u=field['unit'].double();sc=field['scales'].double()/u;assert (sc>0).all()
    cam=field['source_c2w'][0].double();p=(field['means'].double()-cam[:3,3])@cam[:3,:3]/u
    basis=cam[:3,:3].T[None]@rotation(field['rotations']);vari=sc.square()
    cov=(basis*vari[:,None,:])@basis.transpose(-1,-2)
    inv=(basis/vari[:,None,:])@basis.transpose(-1,-2)
    logcov=(basis*(2*sc.log())[:,None,:])@basis.transpose(-1,-2)
    valid=field['has_content'];n=len(p);indices=torch.nonzero(valid).flatten();assert len(indices)>0
    dist=(p[:,None]-p[None]).square().sum(-1);dist[:,~valid]=torch.inf
    # Stable order retains original index for equal squared distances. Self forced first.
    nonself=dist.clone();nonself.fill_diagonal_(torch.inf)
    others=torch.argsort(nonself,dim=1,stable=True)[:,:31]
    neighbors=torch.cat((torch.arange(n,device=p.device)[:,None],others),1)
    mask=valid[neighbors]&torch.isfinite(torch.gather(dist,1,neighbors));mask[:,0]=True
    # Invalid query slots are never updated, but self prevents NaN softmax.
    refs=torch.argsort(nonself,dim=1,stable=True)[:,:3];rv=p[refs]-p[:,None]
    refok=torch.isfinite(torch.gather(nonself,1,refs));rv=rv*refok[:,:,None]
    delta=p[neighbors]-p[:,None];d2=delta.square().sum(-1)
    def cos(a,b):return (a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-12)
    c1=cos(delta[:,:,None],rv[:,None]);c2=cos(-delta[:,:,None],rv[neighbors])
    center=torch.cat((torch.log1p(d2[:,:,None]/p.new_tensor([.02,.05,.1,.2,.5]).square()),torch.asinh(delta),c1,c2),-1)
    def rel(si,sj,ii,ij):
        ma=torch.einsum('nki,nij,nkj->nk',delta,ii,delta);mb=torch.einsum('nki,nkij,nkj->nk',delta,ij,delta)
        ta=torch.einsum('nij,nkji->nk',ii,sj);tb=torch.einsum('nkij,nji->nk',ij,si)
        return torch.stack((ma,mb,ta,tb),-1).log1p()
    full=rel(cov,cov[neighbors],inv,inv[neighbors]);iso=vari.sum(-1)/3
    sphere=torch.stack((d2/iso[:,None],d2/iso[neighbors],3*iso[neighbors]/iso[:,None],3*iso[:,None]/iso[neighbors]),-1).log1p()
    result=dict(node10=torch.cat((torch.asinh(p),svec(logcov),field['opacities'].double()[:,None]),-1).float(),neighbors=neighbors,neighbor_mask=mask,center14=center.float(),full4=full.float(),sphere4=sphere.float(),valid=valid)
    assert all(torch.isfinite(v).all() for v in result.values());return result

def main():
    a=argparse.ArgumentParser();a.add_argument('--shard',type=int,default=0);a.add_argument('--shards',type=int,default=1);args=a.parse_args();p=initialize();torch.set_num_threads(4)
    for index,meta in enumerate(p['training_catalog']+p['catalog']):
        if index%args.shards!=args.shard:continue
        rid=meta['id'];marker=OUT/'preparation'/f'{rid}.json'
        if marker.exists():assert read(marker)['complete'];continue
        t=time.monotonic();field=load(scene_path(rid),'cuda');torch.cuda.synchronize();g0=time.monotonic();graph=source_graph(field);torch.cuda.synchronize();gt=time.monotonic()-g0
        gp=graph_path(rid);gp.parent.mkdir(parents=True,exist_ok=True);save_tensor(gp,graph);files={str(gp):sha256(gp)};inputs={str(scene_path(rid)):sha256(scene_path(rid))};errors=[]
        for q in EVAL_QUERIES:
            wp=weights_path(rid,q);wp.parent.mkdir(parents=True,exist_ok=True);query=load(query_path(rid,q),'cuda');K=query['K'].clone();K[:2]/=14
            weights,_=contribution_weights(field,query['c2w'][None],K[None],32,32);w=weights[0].reshape(1024,-1).float()*field['has_content'][None]
            save_tensor(wp,dict(weights=w));files[str(wp)]=sha256(wp)
            old=load(feature_path(rid,q),'cuda');error=float((w@field['features'].float()-old['render']).abs().max());ae=float((w.sum(-1)-old['effective_alpha'].reshape(-1)).abs().max());errors.append(dict(query=q,Wf_vs_original_max_abs=error,alpha_max_abs=ae));assert error<1e-4 and ae<1e-5
            for path in [query_path(rid,q),feature_path(rid,q)]+[patch_path(rid,q,c) for c in CONDITIONS]:inputs[str(path)]=sha256(path)
        write(marker,dict(complete=True,id=rid,files=files,inputs=inputs,graph_seconds=gt,seconds=time.monotonic()-t,source_only_graph=True,contribution_checks=errors));print(dict(prepared=rid,seconds=time.monotonic()-t),flush=True)
if __name__=='__main__':main()
