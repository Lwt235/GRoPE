"""Paired-seed two-layer attention training, validation selection, sealed testing."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from models.round5_cross_attention import Round5CrossAttentionMatcher,METHODS

FIELDS=('q','k','qmean','qcov','kmean','kcov','k_to_q','k_to_qcov','q_to_k','q_to_kcov')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_split(root,split,device,limit=None):
    arrays={key:[] for key in FIELDS+('gt_valid','pixcontinuous','key_indices','key_weights')}
    identities=[];sources=[]
    for directory in sorted((root/split).iterdir()):
        if not directory.is_dir():continue
        m=json.loads((directory/'manifest.json').read_text())
        if m['split']!=split:raise ValueError('manifest split mismatch')
        if sha(directory/'inputs.npz')!=m['input_sha256'] or sha(directory/'labels.npz')!=m['label_sha256']:
            raise ValueError(f'pair cache changed: {directory}')
        inputs=np.load(directory/'inputs.npz');labels=np.load(directory/'labels.npz')
        n=len(m['pairs']) if limit is None else min(limit,len(m['pairs']))
        for key in FIELDS:arrays[key].append(inputs[key][:n])
        for key in arrays.keys()-set(FIELDS):arrays[key].append(labels[key][:n])
        identities.extend({'scene':m['scene'],'pair_id':f'{q}:{k}','q_index':q,'k_index':k} for q,k in m['pairs'][:n])
        sources.append({'scene':m['scene'],'input_sha256':m['input_sha256'],'label_sha256':m['label_sha256'],
                        'manifest_sha256':sha(directory/'manifest.json')})
    packed={k:torch.from_numpy(np.concatenate(v)).to(device) for k,v in arrays.items()}
    packed['key_indices']=packed['key_indices'].long()
    for k in FIELDS+('pixcontinuous','key_weights'):packed[k]=packed[k].float()
    packed['gt_valid']=packed['gt_valid'].bool()
    return packed,identities,sources

def batch(data,index):return {k:v[index] for k,v in data.items()}

def per_query_ce(logits,data):
    lp=F.log_softmax(logits,dim=-1)
    # Unknown rows use -1 sentinels with zero weights; never gather at -1.
    return -(lp.gather(-1,data['key_indices'].clamp_min(0))*data['key_weights']).sum(-1)

@torch.no_grad()
def evaluate(model,data,identities,method,seed,split,batch_size=8,zero_values=False,probability_dir=None):
    model.eval();rows=[]
    audit_indices=set();scene_counts={}
    if probability_dir is not None:
        probability_dir.mkdir(parents=True,exist_ok=True)
        for i,identity in enumerate(identities):
            scene=identity['scene'];count=scene_counts.get(scene,0)
            if count<3:audit_indices.add(i)
            scene_counts[scene]=count+1
    yy,xx=torch.meshgrid(torch.arange(18,device=data['q'].device),torch.arange(24,device=data['q'].device),indexing='ij')
    grid=torch.stack([xx.flatten()*8+3.5,yy.flatten()*8+3.5],-1)
    for start in range(0,len(identities),batch_size):
        d=batch(data,slice(start,start+batch_size))
        logits=model({k:d[k] for k in FIELDS},zero_values=zero_values)
        if not torch.isfinite(logits).all():raise RuntimeError('nonfinite evaluation logits')
        ce=per_query_ce(logits,d)
        argmax=grid[logits.argmax(-1)]
        soft=logits.softmax(-1)@grid
        err=(argmax-d['pixcontinuous']).norm(dim=-1)
        se=(soft-d['pixcontinuous']).norm(dim=-1)
        mask=d['gt_valid']
        # Invalid sensor coordinates may be NaN; select with where, not NaN*0.
        sums={'ce_sum':torch.where(mask,ce,0).sum(-1),
              'pck8_count':((err<=8)&mask).sum(-1),'pck4_count':((err<=4)&mask).sum(-1),
              'epe_argmax_sum':torch.where(mask,err,0).sum(-1),
              'epe_soft_sum':torch.where(mask,se,0).sum(-1),'n_valid':mask.sum(-1)}
        values={k:v.cpu().tolist() for k,v in sums.items()}
        for j in range(len(mask)):
            rows.append(dict(identities[start+j],method=method,seed=seed,split=split,
                             **{k:v[j] for k,v in values.items()}))
            if start+j in audit_indices:
                np.savez_compressed(probability_dir/f'pair_{start+j:04d}.npz',
                    probability=logits[j].softmax(-1).cpu().numpy(),
                    gt_valid=d['gt_valid'][j].cpu().numpy(),
                    pixcontinuous=d['pixcontinuous'][j].cpu().numpy(),
                    key_indices=d['key_indices'][j].cpu().numpy(),
                    key_weights=d['key_weights'][j].cpu().numpy())
    return rows

def macro(rows,metric='ce_sum'):
    grouped={}
    for r in rows:
        acc=grouped.setdefault(r['scene'],[0.,0]);acc[0]+=r[metric];acc[1]+=r['n_valid']
    if any(n==0 for _,n in grouped.values()):raise ValueError('zero valid scene')
    return float(np.mean([v/n for v,n in grouped.values()]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'results/round5_inputs/pairs')
    p.add_argument('--methods',nargs='+',default=list(METHODS));p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55])
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int)
    p.add_argument('--test-only',action='store_true');p.add_argument('--zero-values',action='store_true')
    p.add_argument('--confirmation-lock',type=Path)
    args=p.parse_args();torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    args.out.mkdir(parents=True,exist_ok=True)
    if not args.test_only:
        train,trids,trsources=load_split(args.data,'train',args.device,args.limit)
    val,vids,vsources=load_split(args.data,'val',args.device,args.limit)
    sourcefiles=['models/round5_cross_attention.py','models/gaussian_pe.py','models/gaussian_spectral_pe.py','models/geometry.py','experiments/round5_train.py','experiments/round5_pairs.py','experiments/round5_cache.py','experiments/round5_rgbd_data.py']
    sourcehash={f:sha(ROOT/f) for f in sourcefiles}
    if args.test_only:
        if not args.confirmation_lock:raise ValueError('freeze completed validation selections before opening test')
        lock=json.loads(args.confirmation_lock.read_text())
        if lock['sourcehash']!=sourcehash:raise ValueError('code changed after confirmation lock')
        sealed={'selection_sha256':ROOT/'data/round5_sun3d/selection.json',
                'protocol_sha256':ROOT/'ROUND5_PROTOCOL.md',
                'analysis_sha256':ROOT/'experiments/round5_analysis.py'}
        for key,path in sealed.items():
            if lock[key]!=sha(path):raise ValueError(f'{key} changed after confirmation lock')
        # Audit all requested checkpoints and replay validation BEFORE opening test.
        for method in args.methods:
            for seed in args.seeds:
                run=args.out/f'{method}_{seed}';key=f'{method}_{seed}'
                frozen=lock['runs'][key]
                ckpt=run/'checkpoints'/'best.pt'
                if sha(ckpt)!=frozen['checkpoint_sha256']:raise ValueError('checkpoint changed after lock')
                loaded=torch.load(ckpt,map_location=args.device,weights_only=True)
                if (loaded['method']!=method or loaded['seed']!=seed or
                    loaded['step']!=frozen['selected_step'] or loaded['sourcehash']!=sourcehash):
                    raise ValueError('checkpoint identity/source mismatch')
                config=json.loads((run/'config.json').read_text())
                if config['val_sources']!=vsources:raise ValueError('validation cache changed after training')
                verifier=Round5CrossAttentionMatcher(method).to(args.device)
                verifier.load_state_dict(loaded['model'])
                replay=evaluate(verifier,val,vids,method,seed,'val',args.batch_size)
                if abs(macro(replay)-frozen['val_ce'])>1e-5 or abs(macro(replay,'pck8_count')-frozen['val_pck8'])>1e-7:
                    raise ValueError('validation replay mismatch')
                print('VALIDATION_REPLAY_OK',key,flush=True)
        test,tids,tsources=load_split(args.data,'test',args.device,args.limit)
        selection=json.loads(sealed['selection_sha256'].read_text())
        expected={(s['scene'],q,k) for s in selection['scenes'] if s['split']=='test' for q,k in s['pairs']}
        observed={(r['scene'],r['q_index'],r['k_index']) for r in tids}
        if observed!=expected or len(tids)!=len(expected):raise ValueError('test pair set differs from frozen selection')
    # Stop accidental overwrite of an existing completed run; --test-only is separate.
    for method in args.methods:
        for seed in args.seeds:
            run=args.out/f'{method}_{seed}';run.mkdir(parents=True,exist_ok=True)
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=Round5CrossAttentionMatcher(method).to(args.device)
            ckpt=run/'checkpoints'/'best.pt';ckpt.parent.mkdir(exist_ok=True)
            if args.test_only:
                suffix='test_zero_values' if args.zero_values else 'test'
                if (run/f'{suffix}.json').exists():raise RuntimeError(f'refusing overwrite {run}/{suffix}')
                loaded=torch.load(ckpt,map_location=args.device,weights_only=True)
                if loaded['sourcehash']!=sourcehash:raise RuntimeError('source changed after checkpoint training')
                model.load_state_dict(loaded['model'])
                rows=evaluate(model,test,tids,method,seed,'test',args.batch_size,args.zero_values,
                              run/'predictions'/suffix)
                (run/f'{suffix}.json').write_text(json.dumps({'rows':rows,'input_sources':tsources,'selected_step':loaded['step'],'sourcehash':sourcehash},indent=2))
                print(method,seed,suffix,'ce',macro(rows),'pck8',macro(rows,'pck8_count'),flush=True)
                continue
            if (run/'completed.json').exists():raise RuntimeError(f'refusing overwrite {run}')
            config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
            (run/'config.json').write_text(json.dumps({'arguments':config,'method':method,'seed':seed,'sourcehash':sourcehash,'train_sources':trsources,'val_sources':vsources,'parameters':sum(x.numel() for x in model.parameters())},indent=2))
            opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
            rng=np.random.default_rng(seed+50000);best=math.inf;history=[];start=time.time();losses=[]
            for step in range(args.steps+1):
                if step%args.eval_every==0 or step==args.steps:
                    rows=evaluate(model,val,vids,method,seed,'val',args.batch_size)
                    ce=macro(rows);record={'step':step,'val_ce':ce,'val_pck8':macro(rows,'pck8_count'),'train_ce':float(np.mean(losses)) if losses else None,'seconds':time.time()-start}
                    history.append(record);losses=[]
                    if ce<best:
                        best=ce;torch.save({'model':model.state_dict(),'step':step,'method':method,'seed':seed,'sourcehash':sourcehash},ckpt)
                        (run/'val.json').write_text(json.dumps({'rows':rows,'selected_step':step,'input_sources':vsources},indent=2))
                    (run/'history.json').write_text(json.dumps(history,indent=2))
                    print(method,seed,record,flush=True)
                if step==args.steps:break
                model.train();idx=torch.as_tensor(rng.integers(len(trids),size=args.batch_size),device=args.device)
                d=batch(train,idx);opt.zero_grad(set_to_none=True)
                logits=model({k:d[k] for k in FIELDS})
                ce=per_query_ce(logits,d);loss=torch.where(d['gt_valid'],ce,0).sum()/d['gt_valid'].sum().clamp_min(1)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(grad):raise RuntimeError('nonfinite training gradient')
                opt.step();losses.append(float(loss.detach()))
                factor=.2+.8*.5*(1+math.cos(math.pi*(step+1)/args.steps))
                for group in opt.param_groups:group['lr']=args.lr*factor
            (run/'completed.json').write_text(json.dumps({'best_val_ce':best,'seconds':time.time()-start,'max_gpu_memory':torch.cuda.max_memory_allocated(),'checkpoint_sha256':sha(ckpt),'sourcehash':sourcehash},indent=2))

if __name__=='__main__':main()
