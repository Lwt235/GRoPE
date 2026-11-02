"""Frozen native-3D Transformer PE screen with matched center geometry."""
import argparse
import hashlib
import math
from pathlib import Path
import random
import shutil
import time
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from models.round46_native3d_pe import make_model,ROLES
from models.round5_cross_attention import METHODS
from experiments import round5_train as R5
from experiments import round43_frequency_train as R43
from experiments import round45_phase_warp_train as R45
from experiments.round46_native3d_data import capture_frame_records,build,load,for_role
from experiments.round43_frequency_train import ROOT,DATA,SEEDS,sha,read,write,checked

OUT=ROOT/'results/round46_native3d/attempt1'
OLD_OUT=ROOT/'results/round43_frequency/attempt1'
SOURCES=tuple(dict.fromkeys(R45.SOURCES+('ROUND46_NATIVE3D_PROTOCOL.md',
 'models/round46_native3d_pe.py','experiments/round46_native3d_train.py',
 'experiments/round46_native3d_audit.py','tests/test_round46_native3d.py','experiments/round46_native3d_data.py','experiments/round5_cache.py')))


def freeze(out):
    if out.exists():raise ValueError('retain existing attempt')
    prior=R43.checked(OLD_OUT)
    for e in prior['entries']:
        for name,digest in e['hashes'].items():
            if sha(name)!=digest:raise ValueError('changed old data')
    seal=read(OLD_OUT/'audit_training/completion.json')
    verified=read(OLD_OUT/'metrics/verification.json')['input_hashes']
    inherited=[];identities={}
    for rec in seal['records']:
        role,seed=rec['role'],rec['seed']
        if role!='point':continue
        path=OLD_OUT/'runs'/role/str(seed);record=read(path/'run.json')
        if sha(path/'run.json')!=rec['run_sha256']:raise ValueError('changed inherited run')
        for name,digest in record['hashes'].items():
            if sha(path/name)!=digest:raise ValueError('changed inherited training')
        eval_=read(path/'old_eval.json')
        if sha(path/'old_eval.json')!=verified[str(path/'old_eval.json')]:raise ValueError('changed old predictions')
        if sha(path/'best.pt')!=rec['checkpoint_sha256'] or eval_['checkpoint_sha256']!=rec['checkpoint_sha256']:
            raise ValueError('inherited checkpoint identity')
        if sha(path/'old_eval_statistics.npz')!=eval_['statistics_sha256']:raise ValueError('inherited statistics')
        names=('run.json','best.pt','history.json','validation.json','score_sample.npz','old_eval.json','old_eval_statistics.npz')
        inherited.append(dict(role='legacy_point2d',seed=seed,path=str(path),hashes={f:sha(path/f) for f in names},numeric=rec['numeric']))
        identity=[record['initial_shared_sha256'],record['batch_sha256']]
        if str(seed) in identities and identities[str(seed)]!=identity:raise ValueError('unpaired inherited baselines')
        identities[str(seed)]=identity
    if len(inherited)!=5:raise ValueError('five historical point controls required')
    frames=capture_frame_records(prior['entries'])
    sources={str(ROOT/p):sha(ROOT/p) for p in SOURCES}
    out.mkdir(parents=True)
    for name in sources:
        target=out/'source'/Path(name).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,target)
    write(out/'lock.json',dict(sources=sources,entries=prior['entries'],roles=ROLES,seeds=SEEDS,steps=1000,
        inherited=inherited,baseline_identity=identities,prior_lock_sha256=sha(OLD_OUT/'lock.json'),
        frames=frames,old_eval_is_not_fresh=True,fresh_data=False,unix=time.time()))
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')


def preflight(out):
    checked(out);torch.manual_seed(20261102)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(
        str(ROOT/'tests'),pattern='test_round46_native3d.py'))
    write(out/'preflight.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,real_arrays_read=False))
    if not result.wasSuccessful():raise SystemExit(1)


def smoke(out):
    from experiments.round46_native3d_audit import check_sample
    from models.round46_native3d_pe import TWO_D
    checked(out)
    if not read(out/'preflight.json')['passed']:raise ValueError('preflight failed')
    rows=[]
    for role in ROLES:
        torch.manual_seed(731);dim=2 if role in TWO_D else 3
        data={k:torch.randn(2,7,128,device='cuda:0') for k in ('q','k')}
        data.update({k:torch.randn(2,7,dim,device='cuda:0')*6 for k in ('qmean','kmean','k_to_q','q_to_k')})
        data.update({k:torch.zeros(2,7,dim,dim,device='cuda:0') for k in ('qcov','kcov','k_to_qcov','q_to_kcov')})
        torch.manual_seed(17);reference=make_model('center2d' if dim==2 else 'point3d').cuda()
        with torch.no_grad():initial_output=reference(data)
        del reference
        torch.manual_seed(17);model=make_model(role).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001)
        with torch.no_grad():initial_error=float((model(data)-initial_output).abs().max())
        if initial_error!=0:raise ValueError('CUDA point initialization mismatch')
        loss=model(data).square().mean();loss.backward()
        gradients={n:float(p.grad.norm()) for n,p in model.named_parameters() if n.startswith('position_field.')}
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if not torch.isfinite(loss) or any(not np.isfinite(x) for x in gradients.values()):raise ValueError('smoke')
        path=out/('smoke_'+role+'.npz');score_sample(model,data,path)
        numeric=check_sample(role,model.state_dict(),path)
        if numeric['warning']:raise ValueError('toy reference discrepancy '+str(numeric))
        rows.append(dict(role=role,loss=float(loss),gradient_norm=float(norm),position_gradients=gradients,
            initial_point_max_error=initial_error,numeric=numeric))
    write(out/'smoke.json',dict(passed=True,rows=rows,gpu=torch.cuda.get_device_name(),real_arrays_read=False))


def data(out):
    from experiments.round46_native3d_audit import audit_data
    build(out,('train','val'))
    for split in ('train','val'):audit_data(out,split)


def old_data(out):
    from experiments.round46_native3d_audit import audit_data
    build(out,('test',));audit_data(out,'test')


def sha_tensor(value):return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def parameter_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if name.startswith('position_field.'):continue
        h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def score_sample(model,data,path):
    # First complete validation pair; eight fixed query rows, all candidate keys.
    d=R5.batch(data,slice(0,1));layer=model.layers[0]
    q=model.input_projection(d['q']);k=model.input_projection(d['k'])
    qq,_,_=layer._project(q);_,kk,_=layer._project(k)
    qq,kk=F.normalize(qq,dim=-1),F.normalize(kk,dim=-1)
    qp=model._position_encode(qq,d['qmean'],d['qcov'])
    kp=model._position_encode(kk,d['k_to_q'],d['k_to_qcov'])
    temp=layer.log_temperature.clamp(max=math.log(60)).exp()
    ids=torch.linspace(0,qq.shape[-2]-1,min(8,qq.shape[-2]),device=qq.device).long()
    probability=(temp*(qp[:,:,ids]@kp.transpose(-1,-2))).softmax(-1)
    values=dict(query_ids=ids,q=qq[0,:,ids],k=kk[0],qmean=d['qmean'][0,ids],
        kmean=d['k_to_q'][0],qcov=d['qcov'][0,ids],kcov=d['k_to_qcov'][0],probability=probability[0])
    np.savez_compressed(path,**{k:v.cpu().numpy() for k,v in values.items()})


def fit(role,seed,train,train_ids,val,val_ids,out):
    dest=out/'runs'/role/str(seed)
    if dest.exists():raise ValueError('retain run')
    dest.mkdir(parents=True);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=make_model(role).cuda();initial=parameter_digest(model)
    initial_position={n:p.detach().cpu().clone() for n,p in model.position_field.named_parameters()}
    initial_projection=sha_tensor(model.position_field.projection) if hasattr(model.position_field,'projection') else None
    lock=checked(out)
    if initial!=lock['baseline_identity'][str(seed)][0]:raise ValueError('backbone initialization changed')
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed+50000);batch_digest=hashlib.sha256();history=[];losses=[];best=float('inf')
    start=time.time();torch.cuda.reset_peak_memory_stats()
    for step in range(1001):
        if step%100==0:
            rows=R5.evaluate(model,val,val_ids,role,seed,'val');ce=R5.macro(rows)
            history.append(dict(step=step,ce=ce,pck8=R5.macro(rows,'pck8_count')))
            if ce<best:
                best,best_step,best_rows=ce,step,rows
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print('VAL',role,seed,step,ce,flush=True)
        if step==1000:break
        model.train();index=rng.integers(len(train_ids),size=8);batch_digest.update(index.astype('<i8').tobytes())
        d=R5.batch(train,index);opt.zero_grad(set_to_none=True)
        logits=model({k:d[k] for k in R5.FIELDS});ce=R5.per_query_ce(logits,d)
        loss=torch.where(d['gt_valid'],ce,0).sum()/d['gt_valid'].sum().clamp_min(1)
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward()
        if step==0:first={n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.named_parameters()}
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        losses.append(dict(step=step+1,loss=float(loss),gradient_norm=float(norm)))
        for group in opt.param_groups:group['lr']=.001*(.2+.8*.5*(1+math.cos(math.pi*(step+1)/1000)))
        if torch.cuda.max_memory_allocated()>16*2**30:raise MemoryError('16GiB allocation ceiling')
    model.load_state_dict(state);model.eval()
    torch.save(dict(role=role,seed=seed,step=best_step,state_dict=state),dest/'best.pt')
    write(dest/'validation.json',best_rows);write(dest/'history.json',dict(validation=history,training=losses))
    score_sample(model,val,dest/'score_sample.npz')
    changes={n:float((p.detach().cpu()-initial_position[n]).norm()) for n,p in model.position_field.named_parameters()}
    if batch_digest.hexdigest()!=lock['baseline_identity'][str(seed)][1]:raise ValueError('batch sequence changed')
    record=dict(role=role,seed=seed,steps=1000,best_step=best_step,val_ce=best,
        initial_shared_sha256=initial,batch_sha256=batch_digest.hexdigest(),first_gradient=first,
        parameters=sum(p.numel() for p in model.parameters()),
        positional_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith('position_field.')),
        initial_projection_sha256=initial_projection,position_parameter_changes=changes,
        selected_shared_changed=parameter_digest(model)!=initial,seconds=time.time()-start,
        peak_allocated=torch.cuda.max_memory_allocated(),
        hashes={f:sha(dest/f) for f in ('best.pt','history.json','validation.json','score_sample.npz')})
    write(dest/'run.json',record);print('DONE',role,seed,record['seconds'],flush=True)


def train(seeds,out):
    checked(out)
    if not read(out/'preflight.json')['passed'] or not read(out/'smoke.json')['passed']:raise ValueError('checks required')
    data,ids=load('train',out,'cuda:0');val,vi=load('val',out,'cuda:0')
    for seed in seeds:
        started=time.time();write(out/f'seed_{seed}_started.json',dict(unix=started,gpu=torch.cuda.get_device_name()))
        offset=SEEDS.index(seed)
        try:
            for role in ROLES[offset:]+ROLES[:offset]:
                fit(role,seed,for_role(data,role),ids,for_role(val,role),vi,out)
                if time.time()-started>10800:raise TimeoutError('3GPU-hour shard budget')
        except BaseException as e:
            write(out/f'seed_{seed}_failure.json',dict(error=repr(e),unix=time.time()));raise
        write(out/f'seed_{seed}_complete.json',dict(completed=True,seconds=time.time()-started))


@torch.no_grad()
def export(seeds,out):
    checked(out);seal=read(out/'audit_training/completion.json')
    if len(seal['records'])!=60:raise ValueError('all selected checkpoints required')
    bundle,ids=load('test',out,'cuda:0')
    yy,xx=torch.meshgrid(torch.arange(18,device='cuda:0'),torch.arange(24,device='cuda:0'),indexing='ij')
    grid=torch.stack((xx.flatten()*8+3.5,yy.flatten()*8+3.5),-1)
    for seed in seeds:
        for role in ROLES:
            data=for_role(bundle,role)
            dest=out/'runs'/role/str(seed);record=read(dest/'run.json')
            if sha(dest/'best.pt')!=record['hashes']['best.pt']:raise ValueError('changed checkpoint')
            if (dest/'old_eval.json').exists():raise ValueError('retain old eval')
            state=torch.load(dest/'best.pt',map_location='cuda:0',weights_only=True)['state_dict']
            model=make_model(role).cuda();model.load_state_dict(state);model.eval();allrows=[]
            saved={}
            for variant,zero in (('primary',False),('zero_v',True)):
                chunks={k:[] for k in ('argmax','soft_xy','log_partition','label_logits')}
                for start in range(0,len(ids),8):
                    d=R5.batch(data,slice(start,start+8));logits=model({k:d[k] for k in R5.FIELDS},zero_values=zero)
                    if not torch.isfinite(logits).all():raise ValueError('nonfinite eval logits')
                    arg=logits.argmax(-1);soft=logits.softmax(-1)@grid
                    values=dict(argmax=arg,soft_xy=soft,log_partition=logits.logsumexp(-1),
                        label_logits=logits.gather(-1,d['key_indices'].clamp_min(0)))
                    ce=R5.per_query_ce(logits,d);err=(grid[arg]-d['pixcontinuous']).norm(dim=-1)
                    se=(soft-d['pixcontinuous']).norm(dim=-1);mask=d['gt_valid']
                    stats=dict(ce_sum=torch.where(mask,ce,0).sum(-1),pck8_count=((err<=8)&mask).sum(-1),
                        pck4_count=((err<=4)&mask).sum(-1),epe_argmax_sum=torch.where(mask,err,0).sum(-1),
                        epe_soft_sum=torch.where(mask,se,0).sum(-1),n_valid=mask.sum(-1))
                    stats={k:v.cpu().tolist() for k,v in stats.items()}
                    allrows.extend(dict(ids[start+j],variant=variant,**{k:v[j] for k,v in stats.items()}) for j in range(len(arg)))
                    for name,value in values.items():chunks[name].append(value.cpu().numpy())
                for name,values in chunks.items():saved[variant+'_'+name]=np.concatenate(values)
            np.savez_compressed(dest/'old_eval_statistics.npz',**saved)
            write(dest/'old_eval.json',dict(rows=allrows,statistics_sha256=sha(dest/'old_eval_statistics.npz'),
                checkpoint_sha256=sha(dest/'best.pt'),old_eval_not_fresh=True))
            print('EXPORT',role,seed,flush=True)
        write(out/f'export_{seed}_complete.json',dict(completed=True,fresh_data=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','preflight','data','old_data','smoke','train','export'))
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS));a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    if a.stage in ('freeze','preflight','data','old_data'):globals()[a.stage](a.out);return
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if not set(a.seeds)<=set(SEEDS):raise ValueError('unknown seed')
    if a.stage=='smoke':smoke(a.out)
    else:globals()[a.stage](a.seeds,a.out)


if __name__=='__main__':main()
