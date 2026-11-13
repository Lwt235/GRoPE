"""Limited old-input field, integration, gradient and complete-resume checks."""
import random
import time
import numpy as np
import torch
from experiments.native_render_parameter_data import *
from experiments.native_render_parameter_prepare import fields
from experiments.native_render_parameter_train import modules,optimizers,snapshot,restore_rng
from experiments.native_receiver_data import core_state,core_load,enable_receiver
from experiments.native_receiver_runtime import model_load,forward_read as old_forward
from experiments.gaussian_scene_preflight import objective,metric_depth
from models.gaussian_scene_parameter_pe import forward_read
from models.gaussian_scene_memory import contribution_weights


def numpy_fields(field,camera,K,weight):
    f={k:v.cpu().numpy() if torch.is_tensor(v) else v for k,v in field.items()};u=float(f['unit']);cam=camera.cpu().numpy().astype('float64');K=K.cpu().numpy().astype('float64')
    p=(f['means'].astype('float64')-cam[:3,3])@cam[:3,:3]/u;w=weight.cpu().numpy().reshape(1024,-1).astype('float64')*f['has_content'][None];a=w.sum(1,keepdims=True);take=a[:,0]>0;den=np.where(take[:,None],a,1)
    def avg(value):return w@value/den
    mean=avg(p);four=[]
    for k in range(5):four.extend((avg(np.sin(np.pi*2**k*p)),avg(np.cos(np.pi*2**k*p))))
    def sv(x):return np.stack([x[...,0,0],x[...,1,1],x[...,2,2],np.sqrt(2)*x[...,0,1],np.sqrt(2)*x[...,0,2],np.sqrt(2)*x[...,1,2]],-1)
    # Independent selected-pixel direct centered summation, not moment subtraction.
    selected=np.array([0,31,127,255,511,777,1023]);scatter=np.stack([np.einsum('j,jc,jd->cd',w[i],p-mean[i],p-mean[i])/den[i,0] for i in selected])
    yy,xx=np.meshgrid(np.arange(32)+.5,np.arange(32)+.5,indexing='ij');pix=np.stack((xx,yy,np.ones_like(xx)),-1).reshape(-1,3);rays=pix@np.linalg.inv(K).T@cam[:3,:3].T;rays/=np.linalg.norm(rays,axis=1,keepdims=True)
    delta=(f['means'].astype('float64')-cam[:3,3])/u;along=rays@delta.T;radius=np.maximum(np.sum(delta*delta,axis=1)[None]-along*along,0)
    old=np.concatenate([np.log1p(radius[...,None]/np.square([.02,.05,.1,.2,.5])),np.arcsinh(along)[...,None],np.broadcast_to(np.log1p(np.linalg.norm(delta,axis=1))[None,:,None],(1024,len(p),1))],-1)
    center=np.concatenate([mean[selected]]+[x[selected] for x in four]+[np.arcsinh(sv(scatter)),np.einsum('ij,ijc->ic',w[selected],old[selected])/den[selected]],-1)
    quat=f['rotations'].astype('float64');quat/=np.linalg.norm(quat,axis=1,keepdims=True);qw,x,y,z=quat.T
    rot=np.stack([1-2*(y*y+z*z),2*(x*y-qw*z),2*(x*z+qw*y),2*(x*y+qw*z),1-2*(x*x+z*z),2*(y*z-qw*x),2*(x*z-qw*y),2*(y*z+qw*x),1-2*(x*x+y*y)],-1).reshape(-1,3,3)
    scale=f['scales'].astype('float64');basis=cam[:3,:3].T[None]@rot;logs=2*np.log(scale/u);lc=(basis*logs[:,None,:])@np.swapaxes(basis,-1,-2);ls=np.log(np.sum(scale*scale,1)/(3*u*u))[:,None,None]*np.eye(3)
    full=avg(sv(lc))[selected];sphere=avg(sv(ls))[selected]
    for value in [center,full,sphere]:value[~take[selected]]=0
    return selected,dict(center46=center.astype('float32'),full6=full.astype('float32'),sphere6=sphere.astype('float32'))


def main():
    p=plan();torch.set_num_threads(4);start=time.monotonic();RAM.mkdir(mode=0o700,exist_ok=True);assert __import__('shutil').disk_usage(RAM).free>20*2**30
    checks=[]
    for rid in ['fit_00','confirm_12']:
        q=2;field=load(scene_path(rid),'cuda');query=load(query_path(rid,q),'cuda');K=query['K'].clone();K[:2]/=14;w,_=contribution_weights(field,query['c2w'][None],K[None],32,32);selected,expected=numpy_fields(field,query['c2w'],K,w[0]);actual=load(geometry_path(rid,q));errs={key:float(np.max(np.abs(v-actual[key][selected].numpy()))) for key,v in expected.items()}
        assert max(errs.values())<2e-5,errs;checks.append(dict(id=rid,query=q,pixel_indices=selected.tolist(),independent_NumPy_field_max_abs=errs));del field,query,w
    model=model_load(p);core=enable_receiver(model);initial={n:t.detach().cpu().clone() for n,t in core_state(model).items()};fusion,pe=modules(SEEDS[0]);replays=[]
    state=load(parent.checkpoint_path('Render',SEEDS[0]));core_load(model,state['native']);fusion.load_state_dict(state['model']);del state
    for rid,q,c,cohort in [('fit_00',1,'missing','train'),('fidelity_00',2,'full','development')]:
        sample=to_device(sample_cpu('P',rid,q,c),'cuda');marker=read(parent.OUT/'evaluation'/f'{cohort}_Render_{SEEDS[0]}.json');payload=load(marker['prediction_path']);index=[tuple(k) for k in payload['keys']].index((rid,q,c))
        budget_charge('nontraining_forwards',2)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            expected=old_forward(model,sample,fusion,'Render',checkpoint_blocks=False);actual=forward_read(model,sample,fusion,pe,checkpoint_blocks=False)
        zero=float((expected-actual).abs().max());replay=float((metric_depth(actual).float().cpu()-payload['depth_metric'][index]).abs().max());assert zero==0 and replay<=1e-5
        replays.append(dict(id=rid,query=q,condition=c,zero_PE_max_abs=zero,inherited_R_metric_max_abs=replay,source_sha256=marker['prediction_sha256']));del sample,payload,expected,actual
    gradients=[];initials={};restore_record={}
    recovered_G=(RAM/'preflight_complete_state.pt').exists() and read(OUT/'budget.json')['preflight_updates']==3
    for kind in ['G','S']:
        core_load(model,initial);fusion,pe=modules(SEEDS[0]);initials[kind]=dict(fusion={n:t.detach().cpu().clone() for n,t in fusion.state_dict().items()},pe={n:t.detach().cpu().clone() for n,t in pe.state_dict().items()})
        opts=optimizers(fusion,pe,core,p);sample=to_device(sample_cpu(kind,'fit_00',2,'missing'),'cuda');steps=[]
        for step in range(0 if kind=='G' and recovered_G else 3):
            for opt in opts:opt.zero_grad(set_to_none=True)
            budget_charge('nontraining_forwards');trace={};depth=forward_read(model,sample,fusion,pe,trace=trace);loss=objective(depth,sample['query']['target'],sample['query']['valid']);assert torch.isfinite(loss);loss.backward()
            native={group:[float(getattr(model.aggregator,group)[i].attn.qkv.weight.grad.norm()) for i in range(24)] for group in ['frame_blocks','global_blocks']};assert all(np.isfinite(v) and v>0 for vv in native.values() for v in vv)
            assert all(t.grad is None for t in model.aggregator.patch_embed.parameters()) and any(t.grad is not None and t.grad.norm()>0 for t in model.depth_head.parameters())
            first=float(pe.network[0].weight.grad[:,46:].norm());last=float(pe.network[-1].weight.grad.norm());assert last>0 and trace['special_increment_max']==0
            if step==2:assert first>0 and trace['qk_increment_norm']>0
            norm=torch.nn.utils.clip_grad_norm_(list(fusion.parameters())+list(pe.parameters())+core,p['gradient_clip'],foreach=False);assert torch.isfinite(norm);budget_charge('preflight_updates')
            for opt in opts:opt.step()
            steps.append(dict(step=step+1,loss=float(loss),PE_last_grad=last,extra_six_first_grad=first,trace=trace,native_gradients=native))
        if kind=='G':
            if recovered_G:
                state=load(RAM/'preflight_complete_state.pt');assert state['step']==3
                core_load(model,state['native']);fusion.load_state_dict(state['fusion']);pe.load_state_dict(state['pe'])
                for opt,saved in zip(opts,state['optimizer']):opt.load_state_dict(saved)
                restore_rng(state['rng']);del state
            for opt in opts:opt.zero_grad(set_to_none=True)
            budget_charge('nontraining_forwards',2)
            with torch.no_grad():reference=forward_read(model,sample,fusion,pe,checkpoint_blocks=False).detach().cpu()
            path=RAM/'preflight_complete_state.pt';identity=dict(kind=kind,seed=SEEDS[0],schedule_sha256='preflight-discarded',configuration_sha256=sha256(OUT/'plan.json'));save_tensor(path,snapshot(model,fusion,pe,opts,3,identity))
            state=load(path);assert state['step']==3 and all(state[n]==v for n,v in identity.items())
            with torch.no_grad():next(model.aggregator.global_blocks[0].parameters()).add_(.01);pe.network[-1].weight.zero_()
            core_load(model,state['native']);fusion.load_state_dict(state['fusion']);pe.load_state_dict(state['pe'])
            for opt in opts:opt.state.clear()
            torch.cuda.empty_cache()
            for opt,saved in zip(opts,state['optimizer']):opt.load_state_dict(saved)
            restore_rng(state['rng'])
            with torch.no_grad():replay=forward_read(model,sample,fusion,pe,checkpoint_blocks=False).detach().cpu()
            error=float((reference-replay).abs().max());assert error==0;restore_record=dict(complete=True,prediction_max_abs=error,checkpoint_bytes=path.stat().st_size,all_native_fusion_PE_two_Adam_rng_step_identity_restored=True);path.unlink();del state,reference,replay
        gradients.append(dict(kind=kind,steps=steps,recovered_G_pilot_checks_before_saved_step=bool(kind=='G' and recovered_G),prior_assertion_source='failures/preflight_restore_memory_v1/preflight_before.py' if kind=='G' and recovered_G else None));model.zero_grad(set_to_none=True)
        for opt in opts:opt.zero_grad(set_to_none=True)
        del opts,opt,sample,depth,loss;torch.cuda.empty_cache()
    assert all(torch.equal(t,initials['S'][part][n]) for part,st in initials['G'].items() for n,t in st.items())
    # P's extra six inputs are explicitly zero; full initial state is otherwise identical.
    f,ppe=modules(SEEDS[0]);assert all(torch.equal(t.cpu(),initials['G']['pe'][n]) for n,t in ppe.state_dict().items());assert position(load(geometry_path('fit_00',2)),'P')[:,46:].count_nonzero()==0
    write(OUT/'preflight.json',dict(complete=True,field_checks=checks,replays=replays,gradients=gradients,complete_resume=restore_record,common_fusion_PE_initialization=True,P_inactive_coefficients=384,total_trainable=637383906+787712+68928,preflight_updates=6,preflight_charged_forwards_including_failed_attempt=10 if recovered_G else 8,baseline_replay_and_zero_check_forwards_including_failed_attempt=8 if recovered_G else 4,pilot_weights_discarded=True,peak_GiB=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-start))
    print(dict(preflight_complete=True,seconds=time.monotonic()-start,peak_GiB=torch.cuda.max_memory_allocated()/2**30),flush=True)


if __name__=='__main__':
    from experiments.native_render_parameter_train import rng_state
    main()
