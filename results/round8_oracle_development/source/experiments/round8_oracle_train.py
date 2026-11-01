"""Old-development 2x2 privileged routing/value diagnostic; no test loader."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from models.round8_oracle_attention import OracleDepthAttention,ROLES
from models.round7_depth_attention import depth_loss
from experiments.round7_depth_train import load_data as load_base,batch,evaluate
from experiments.round7_depth_data import ROOT,sha

SEEDS=(11,22,33,44,55)
BASE_ROOTS=(ROOT/'results/round7_depth_old',ROOT/'results/round7_depth_bm4')
ORACLE_ROOT=ROOT/'results/round8_oracle_data'
SCENES={'train':{'harvard_c3/hv_c3_1','harvard_c5/hv_c5_1','harvard_c6/hv_c6_1'},
        'val':{'harvard_c8/hv_c8_3','brown_bm_4/brown_bm_4'}}
SOURCES=('ROUND8_PROTOCOL.md','models/round8_oracle_attention.py','experiments/round8_oracle_train.py',
    'experiments/round8_oracle_data.py','experiments/round8_oracle_analysis.py',
    'experiments/round8_transport_probe.py','models/round7_depth_attention.py',
    'models/gaussian_quadratic_pe.py','experiments/round7_depth_data.py',
    'experiments/round7_depth_train.py','experiments/round6_region_data.py','experiments/round5_rgbd_data.py')


def load_development(split,device='cpu'):
    if split not in SCENES:raise ValueError('Round8 permits only fixed old train/val')
    data,ids,sources=load_base(BASE_ROOTS,split,device)
    if {r['scene'] for r in ids}!=SCENES[split] or len(ids)!=(576 if split=='train' else 128):
        raise ValueError('old development population mismatch')
    caches={};oracle_sources=[];indices={}
    for scene in sorted(SCENES[split]):
        path=ORACLE_ROOT/split/scene.replace('/','__');m=json.loads((path/'manifest.json').read_text())
        if m['scene']!=scene or m['split']!=split:raise ValueError('oracle split/scene mismatch')
        base=Path(m['base_path'])
        for file,key in [('manifest.json','base_manifest_sha256'),('inputs.npz','base_input_sha256'),('labels.npz','base_label_sha256')]:
            if sha(base/file)!=m[key]:raise ValueError('oracle reference observation/label changed')
        if m['frozen_protocol_sha256']!=sha(ORACLE_ROOT/'frozen_export_protocol.json'):
            raise ValueError('oracle export contract changed')
        if sha(path/'oracle.npz')!=m['oracle_sha256']:raise ValueError('oracle archive changed')
        with np.load(path/'oracle.npz',allow_pickle=False) as a:
            caches[scene]={f:a[f].copy() for f in a.files}
        indices[scene]={tuple(pair):i for i,pair in enumerate(m['pairs'])}
        expected={(r['q_index'],r['k_index']) for r in ids if r['scene']==scene}
        if set(indices[scene])!=expected or len(m['pairs'])!=len(expected):raise ValueError('oracle pair identity mismatch')
        oracle_sources.append(dict(path=str(path),manifest_sha256=sha(path/'manifest.json'),oracle_sha256=m['oracle_sha256']))
    fields=next(iter(caches.values())).keys()
    for field in fields:
        values=np.stack([caches[r['scene']][field][indices[r['scene']][(r['q_index'],r['k_index'])]] for r in ids])
        data[field]=torch.from_numpy(values).to(device)
    return data,ids,sources,oracle_sources


def sources_now():return {p:sha(ROOT/p) for p in SOURCES}

def freeze(out):
    if out.exists():raise ValueError('refuse overwrite diagnostic directory')
    for n,key in [(6,'sourcehash'),(7,'source_hashes')]:
        old=ROOT/('results/round6_confirmation/lock.json' if n==6 else 'results/round7_depth_confirmation/lock.json')
        for rel,digest in json.loads(old.read_text())[key].items():
            if sha(ROOT/rel)!=digest:raise ValueError('prior sealed source changed '+rel)
    _,ti,ts,ots=load_development('train');_,vi,vs,ovs=load_development('val')
    mapping=sources_now();out.mkdir(parents=True)
    for rel in mapping:
        dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    lock=dict(source_hashes=mapping,roles=list(ROLES),seeds=list(SEEDS),steps=1000,
        data_sources=dict(train=ts,val=vs,oracle_train=ots,oracle_val=ovs),
        train_pairs=len(ti),validation_pairs=len(vi),test_access_authorized=False,
        oracle_protocol_sha256=sha(ORACLE_ROOT/'frozen_export_protocol.json'),
        scope='old development privileged routing/source-depth diagnostic; no deployable or independent-confirmation claim')
    (out/'lock.json').write_text(json.dumps(lock,indent=2));print('FROZEN',sha(out/'lock.json'),flush=True)


def fit(role,seed,train,ti,val,vi,out,lock):
    dest=out/role/str(seed)
    if dest.exists():raise ValueError('refuse overwrite '+str(dest))
    dest.mkdir(parents=True);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=OracleDepthAttention(role).to(train['q'].device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed);batch_hash=hashlib.sha256();history=[];best=float('inf');start=time.time()
    for step in range(1001):
        if step%100==0:
            result,_=evaluate(model,val,vi);score=result['summary']['metrics']['AbsRel']
            history.append(dict(step=step,validation=result['summary']))
            if score<best:
                best=score;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print('VAL',role,seed,step,round(score,6),'sec',round(time.time()-start,1),flush=True)
        if step==1000:break
        model.train();ix=rng.integers(len(ti),size=8);batch_hash.update(ix.astype('<i8').tobytes())
        d=batch(train,ix);optimizer.zero_grad(set_to_none=True);pred=model(d)
        loss=depth_loss(pred,d['target'],d['valid'])
        if not torch.isfinite(loss):raise ValueError('nonfinite diagnostic loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    model.load_state_dict(best_state);result,pred=evaluate(model,val,vi)
    torch.save(dict(role=role,seed=seed,step=best_step,state_dict=best_state,source_hashes=lock['source_hashes']),dest/'best.pt')
    np.savez_compressed(dest/'val_predictions.npz',depth=pred.detach().cpu().numpy())
    (dest/'validation.json').write_text(json.dumps(result,indent=2))
    metadata=dict(role=role,seed=seed,steps=1000,best_step=best_step,best_validation_AbsRel=best,
        parameter_count=sum(p.numel() for p in model.parameters()),source_hashes=lock['source_hashes'],
        data_sources=lock['data_sources'],batch_sequence_sha256=batch_hash.hexdigest(),
        checkpoint_sha256=sha(dest/'best.pt'),prediction_sha256=sha(dest/'val_predictions.npz'),
        validation_sha256=sha(dest/'validation.json'),lock_sha256=sha(out/'lock.json'),
        history=history,seconds=time.time()-start,device=str(train['q'].device),
        GPU=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        effective_routing_training='QK learns only on fallback rows' if role in ('gt_route','gt_both') else 'normal learned QK',
        privileged_GT_input=role in ('gt_route','gt_value','gt_both'),independent_confirmation=False)
    (dest/'run.json').write_text(json.dumps(metadata,indent=2));print('DONE',role,seed,best_step,best,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','train'])
    p.add_argument('--out',type=Path,default=ROOT/'results/round8_oracle_development')
    p.add_argument('--roles',nargs='+',choices=ROLES,default=list(ROLES));p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.action=='freeze':freeze(a.out);return
    lock=json.loads((a.out/'lock.json').read_text())
    if sources_now()!=lock['source_hashes'] or lock['test_access_authorized'] is not False:raise ValueError('diagnostic lock mismatch')
    if not set(a.seeds)<=set(SEEDS):raise ValueError('unplanned seed')
    tr,ti,ts,ots=load_development('train',a.device);va,vi,vs,ovs=load_development('val',a.device)
    if dict(train=ts,val=vs,oracle_train=ots,oracle_val=ovs)!=lock['data_sources']:raise ValueError('frozen data changed')
    for role in a.roles:
        for seed in a.seeds:fit(role,seed,tr,ti,va,vi,a.out,lock)
    if sources_now()!=lock['source_hashes']:raise ValueError('source changed during training')

if __name__=='__main__':main()
