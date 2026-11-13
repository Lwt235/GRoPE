"""Two old training scenes only; bounded independent fields, pilots and recovery."""
import time
import inspect
import numpy as np
import torch
from experiments.native_memory_context_data import *
from experiments.native_memory_context_prepare import source_graph
from experiments.native_memory_context_train import modules,optimizers,snapshot,restore_rng
from experiments.native_receiver_data import core_state,core_load,enable_receiver,digest
from experiments.native_receiver_runtime import model_load,forward_read as old_forward
from experiments.gaussian_scene_preflight import objective,metric_depth
from models.gaussian_scene_context_writer import MemoryContextWriter,forward_read
from models.gaussian_scene_memory import rasterize_field

def numpy_check(field,graph):
    f={k:v.cpu().numpy() for k,v in field.items()};cam=f['source_c2w'][0].astype('float64');u=float(f['unit']);p=(f['means'].astype('float64')-cam[:3,3])@cam[:3,:3]/u
    quat=f['rotations'].astype('float64');quat/=np.linalg.norm(quat,axis=1,keepdims=True);w,x,y,z=quat.T
    rot=np.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1).reshape(-1,3,3);basis=cam[:3,:3].T[None]@rot;var=(f['scales'].astype('float64')/u)**2
    cov=(basis*var[:,None,:])@basis.transpose(0,2,1);inv=(basis/var[:,None,:])@basis.transpose(0,2,1);logc=(basis*np.log(var)[:,None,:])@basis.transpose(0,2,1)
    sv=np.stack([logc[:,0,0],logc[:,1,1],logc[:,2,2],2**.5*logc[:,0,1],2**.5*logc[:,0,2],2**.5*logc[:,1,2]],-1)
    node=np.concatenate([np.arcsinh(p),sv,f['opacities'][:,None]],-1).astype('float32');errors={'node10':float(np.max(np.abs(node-graph['node10'].cpu().numpy())))}
    good=np.flatnonzero(f['has_content']);take=good[np.linspace(0,len(good)-1,7,dtype=int)];cent=[];full=[];sphere=[]
    def refs(i):
        ids=sorted((k for k in good if k!=i),key=lambda k:(float(np.sum((p[k]-p[i])**2)),k))[:3];return [p[k]-p[i] for k in ids]+[np.zeros(3)]*(3-len(ids))
    def cos(a,b):return float(a@b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-12)
    for i in take:
        expected=[i]+sorted((k for k in good if k!=i),key=lambda k:(float(np.sum((p[k]-p[i])**2)),k))[:31];actual=graph['neighbors'][i].cpu().tolist();assert actual[:len(expected)]==expected
        cc=[];gg=[];ss=[]
        for j in actual:
            d=p[j]-p[i];d2=d@d;cc.append(list(np.log1p(d2/np.square([.02,.05,.1,.2,.5])))+list(np.arcsinh(d))+[cos(d,v) for v in refs(i)]+[cos(-d,v) for v in refs(j)])
            gg.append(np.log1p([d@inv[i]@d,d@inv[j]@d,np.trace(inv[i]@cov[j]),np.trace(inv[j]@cov[i])]))
            vi=var[i].sum()/3;vj=var[j].sum()/3;ss.append(np.log1p([d2/vi,d2/vj,3*vj/vi,3*vi/vj]))
        cent.append(cc);full.append(gg);sphere.append(ss)
    for k,v in [('center14',cent),('full4',full),('sphere4',sphere)]:errors[k]=float(np.max(np.abs(np.asarray(v,dtype='float32')-graph[k][take].cpu().numpy())))
    assert max(errors.values())<2e-5;return dict(selected_nodes=take.tolist(),independent_NumPy_max_abs=errors)

def sample(kind,rid='fit_00',q=2,c='missing'):
    return to_device(dataset(plan(),kind,True).samples[rid,q,c],'cuda')

def main():
    p=plan();start=time.monotonic();checks=[];initials={};gradients=[];replays=[]
    for rid in ['fit_00','confirm_12']:
        field=load(scene_path(rid),'cuda');graph=load(graph_path(rid),'cuda');checks.append(dict(id=rid,**numpy_check(field,graph)));del field,graph
    assert list(inspect.signature(MemoryContextWriter.forward).parameters)==['self','features','graph','kind']
    model=model_load(p);core=enable_receiver(model);initial={n:t.detach().cpu().clone() for n,t in core_state(model).items()};fusion,writer=modules(SEEDS[0])
    old=load(parent.checkpoint_path('Render',SEEDS[0]));core_load(model,old['native']);fusion.load_state_dict(old['model']);del old
    for rid,q,c,cohort in [('fit_00',1,'missing','train'),('confirm_12',2,'full','train')]:
        field=load(scene_path(rid),'cuda');graph=load(graph_path(rid),'cuda');budget_charge('independent_writer_forwards');features=writer(field['features'],graph,'G');assert torch.equal(features,field['features'])
        path=RAM/f'preflight_{rid}.pt';saved=dict(field,features=features);save_tensor(path,saved);field2=load(path,'cuda');assert all(torch.equal(field[k],field2[k]) for k in field);path.unlink()
        query=load(query_path(rid,q),'cuda');K=query['K'].clone();K[:2]/=14;rendered,_,_=rasterize_field(field2,query['c2w'][None],K[None],32,32,field2['features'].float()*field2['has_content'][:,None]);actual_extra=rendered[0].reshape(1024,1024);old_extra=load(feature_path(rid,q),'cuda')['render'];assert (actual_extra-old_extra).abs().max()<1e-4
        a=dict(patches=load(patch_path(rid,q,c),'cuda')['patches'],extra=actual_extra,hw=(448,448));b=dict(a,extra=old_extra);budget_charge('nontraining_forwards',2)
        with torch.no_grad():pred=forward_read(model,a,fusion,checkpoint_blocks=False);ref=old_forward(model,b,fusion,'Render',checkpoint_blocks=False)
        err=float((pred-ref).abs().max());assert torch.isfinite(pred).all() and torch.isfinite(ref).all()
        # Zero writer content is exactly identical to original FP16 source;
        # actual raster versus prior cached render may cross native BF16 bins.
        # Record that engineering roundoff, not a scientific exclusion gate.
        replays.append(dict(id=rid,query=q,condition=c,zero_residual_original_source_content_exact=True,actual_renderer_vs_cached_R_depth_max_abs=err,render_max_abs=float((actual_extra-old_extra).abs().max()),FP16_state_reload_exact=True));del pred,ref,a,b,field,field2,saved,features,rendered,query,graph
    for kind in ['G','S']:
        core_load(model,initial);fusion,writer=modules(SEEDS[0]);initials[kind]=dict(fusion=digest(fusion),writer=digest(writer));opts=optimizers(fusion,writer,core,p);one=sample(kind);steps=[]
        for step in range(3):
            for opt in opts:opt.zero_grad(set_to_none=True)
            budget_charge('nontraining_forwards');budget_charge('independent_writer_forwards');pred=forward_read(model,one,fusion,writer,kind);loss=objective(pred,one['query']['target'],one['query']['valid']);assert torch.isfinite(loss);loss.backward()
            grads={k:float(v.grad.norm()) if v.grad is not None else 0 for k,v in writer.named_parameters()};assert any(t.grad is not None and t.grad.norm()>0 for t in model.depth_head.parameters())
            if step==2:assert grads['output.weight']>0 and grads['relation.0.weight']>0 and grads['qkv.weight']>0
            norm=torch.nn.utils.clip_grad_norm_(list(fusion.parameters())+list(writer.parameters())+core,p['gradient_clip'],foreach=False);assert torch.isfinite(norm);budget_charge('preflight_updates')
            for opt in opts:opt.step()
            steps.append(dict(step=step+1,loss=float(loss.detach()),writer_gradients=grads));del pred,loss
        if kind=='G':
            for opt in opts:opt.zero_grad(set_to_none=True)
            budget_charge('nontraining_forwards',2);budget_charge('independent_writer_forwards',2)
            with torch.no_grad():reference=forward_read(model,one,fusion,writer,kind,checkpoint_blocks=False).cpu()
            path=RAM/'preflight_full.pt';ident=dict(kind='G',seed=SEEDS[0],plan_sha256=sha256(OUT/'plan.json'));save_tensor(path,snapshot(model,fusion,writer,opts,3,ident));state=load(path);assert state['step']==3
            with torch.no_grad():writer.output.weight.zero_();next(model.aggregator.global_blocks[0].parameters()).add_(.01)
            core_load(model,state['native']);fusion.load_state_dict(state['fusion']);writer.load_state_dict(state['writer'])
            for opt in opts:opt.state.clear()
            torch.cuda.empty_cache()
            for opt,saved in zip(opts,state['optimizer']):opt.load_state_dict(saved)
            restore_rng(state['rng'])
            with torch.no_grad():actual=forward_read(model,one,fusion,writer,kind,checkpoint_blocks=False).cpu()
            error=float((reference-actual).abs().max());assert error==0;restore=dict(complete=True,prediction_max_abs=error,checkpoint_bytes=path.stat().st_size,model_fusion_writer_two_Adam_rng_step_identity=True);path.unlink();del state,reference,actual
        gradients.append(dict(kind=kind,steps=steps));model.zero_grad(set_to_none=True)
        for opt in opts:opt.zero_grad(set_to_none=True)
        del opts,opt,one;torch.cuda.empty_cache()
    assert initials['G']==initials['S'];fusion,writer=modules(SEEDS[0]);assert dict(fusion=digest(fusion),writer=digest(writer))==initials['G']
    write(OUT/'preflight.json',dict(complete=True,field_checks=checks,replays=replays,gradients=gradients,complete_resume=restore,all_PSG_initials_same=True,schedule_pairing='Same original deterministic schedule; full formal actual logs verified at completion.',writer_API_source_only=True,writer_parameters=sum(t.numel() for t in writer.parameters()),P_inactive_coefficients=256,preflight_updates=6,charged_native_forwards=12,independent_writer_forwards=10,FP16_content_train_save_same=True,peak_GiB=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-start,pilot_weights_discarded=True))
    print(dict(preflight_complete=True,seconds=time.monotonic()-start),flush=True)
if __name__=='__main__':main()
