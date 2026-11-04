"""Fixed reference regions and labels under within-region observation thinning."""
import hashlib
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from experiments.round58_data import tokenize
from experiments.round60_data import scene_reader, sample_inputs, attach_supervision
from models.round65_actual_gs import shape_features
from experiments.round65_common import ROOT, read, write, sha

OUT=ROOT/'results/round68_fixed_region/attempt1'


def rng_for(identity,salt):
    digest=hashlib.sha256((salt+':'+identity).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8],'little'))


def shuffle_indices(identities,tokens=64):
    result=[]
    for identity in identities:
        views=[]
        for view in range(2):
            rng=rng_for(str(identity)+':'+str(view),'R68-shape-association-v1')
            # A uniformly sampled cyclic shift of a random cycle has no fixed point.
            order=rng.permutation(tokens);shift=int(rng.integers(1,tokens))
            permutation=np.empty(tokens,dtype=np.int64)
            permutation[order]=np.roll(order,shift)
            views.append(permutation)
        result.append(views)
    return np.asarray(result)


@torch.no_grad()
def thin_regions(batch,reference,identities):
    points=batch['points'];b,v,n,_=points.shape
    anchors=reference['anchors'].cpu().numpy()
    retained=np.zeros((b,v,n),dtype=bool)
    for i,identity in enumerate(identities):
        for view in range(v):
            selected=np.unique(anchors[i,view])
            remaining=np.setdiff1d(np.arange(n),selected)
            extra=rng_for(str(identity)+':'+str(view),'R68-retain256-v1').choice(remaining,256-len(selected),replace=False)
            retained[i,view,np.r_[selected,extra]]=True
    retained=torch.as_tensor(retained,device=points.device)
    indices=reference['neighbors'];tokens,neighbors=indices.shape[-2:]
    flat=points.reshape(b*v,n,3);ids=indices.reshape(b*v,tokens,neighbors)
    rows=torch.arange(b*v,device=points.device)[:,None,None]
    patch=flat[rows,ids]
    mask=retained.reshape(b*v,n)[rows,ids]
    count=mask.sum(-1)
    if int(count.min())<1:raise ValueError('retained anchor must keep every region observed')
    centers=(patch*mask[...,None]).sum(-2)/count[...,None]
    delta=patch-centers.unsqueeze(-2)
    covariance=torch.einsum('btki,btkj->btij',delta*mask[...,None],delta)/count[...,None,None]
    covariance+=torch.eye(3,device=points.device)*.02**2
    eigen=torch.linalg.eigvalsh(covariance.double()).float()
    radial=delta.norm(dim=-1).masked_fill(~mask,float('inf')).sort(-1).values
    qids=(torch.arange(1,9,device=points.device)*count[...,None]//8-1).clamp_min(0)
    quantiles=torch.gather(radial,-1,qids)
    features=torch.cat((.5*(eigen/.04).log(),(quantiles/.2).log1p()),-1)
    return dict(features=features.reshape(b,v,tokens,11),centers=centers.reshape(b,v,tokens,3),
        covariance=covariance.reshape(b,v,tokens,3,3),surviving_counts=count.reshape(b,v,tokens),
        retained=retained,world_centers=reference['world_centers'],overlap=reference['overlap'])


@torch.no_grad()
def geometry_scores(centers,covariance):
    b,v,n,_=centers.shape
    distances=torch.cdist(centers.double(),centers.double())/.2
    diagonal=torch.eye(n,device=centers.device,dtype=torch.bool)
    signature=distances.masked_fill(diagonal,float('inf')).sort(-1).values[...,:8]
    matches=torch.cdist(signature[:,0],signature[:,1])/np.sqrt(8)
    logshape=shape_features(covariance)
    ambiguity=[];diversity=[]
    for direction in range(2):
        scores=matches if direction==0 else matches.transpose(-1,-2)
        best,indices=scores.topk(4,largest=False,sorted=True)
        ambiguity.append(1-(best[...,1]-best[...,0])/(best[...,3]+1e-12))
        target=logshape[:,1-direction]
        candidates=target[torch.arange(b,device=centers.device)[:,None,None],indices]
        delta=candidates.unsqueeze(-2)-candidates.unsqueeze(-3)
        pair=delta.square().sum(-1).sqrt()
        upper=torch.triu(torch.ones(4,4,device=centers.device,dtype=torch.bool),diagonal=1)
        diversity.append(pair[...,upper].mean(-1))
    return torch.stack((torch.stack(ambiguity,1),torch.stack(diversity,1)),-1)


def collect(lock,split,realization,device='cpu'):
    specifications=[e for e in lock['entries'] if e['split']==split]
    collected={};sparse_collected={};identities=[];scenes=[];scales=[];gaps=[];excluded=[]
    pending=[];pending_ids=[];sample=None;started=time.time()
    def flush():
        nonlocal sample
        if not pending:return
        batch={k:torch.as_tensor(np.stack([p[k] for p in pending]),device=device) for k in pending[0]}
        reference=tokenize(batch);scores=geometry_scores(reference['centers'],reference['covariance'])
        for key in ('features','centers','covariance','world_centers','overlap'):
            collected.setdefault(key,[]).append(reference[key].cpu().numpy())
        collected.setdefault('geometry_scores',[]).append(scores.cpu().numpy())
        permutation=shuffle_indices(pending_ids);collected.setdefault('permutation',[]).append(permutation)
        if split=='old_eval':
            sparse=thin_regions(batch,reference,pending_ids)
            for key in ('features','centers','covariance','world_centers','overlap','surviving_counts'):
                sparse_collected.setdefault(key,[]).append(sparse[key].cpu().numpy())
            sparse_collected.setdefault('geometry_scores',[]).append(scores.cpu().numpy())
            sparse_collected.setdefault('permutation',[]).append(permutation)
        else:sparse=None
        if sample is None:
            sample={k:v[:2].cpu().numpy() for k,v in batch.items()}
            sample.update({'reference_'+k:v[:2].cpu().numpy() for k,v in reference.items()})
            sample['geometry_scores']=scores[:2].cpu().numpy()
            sample['permutation']=permutation[:2]
            if sparse is not None:sample.update({'sparse_'+k:v[:2].cpu().numpy() for k,v in sparse.items()})
        pending.clear();pending_ids.clear()
    for entry in specifications:
        reader,poses=scene_reader(entry,lock)
        accepted=0
        for q,k in entry['pairs']:
            identity=f"{entry['scene']}:{q}:{k}"
            clouds=[reader(q),reader(k)]
            if any(len(c['points'])<512 for c in clouds):
                excluded.append(dict(id=identity,reason='fewer_than512_observed_points'));continue
            inputs=sample_inputs(clouds,identity,realization,count=512,augment=split=='fit')
            pending.append(attach_supervision(inputs,poses[[q,k]]));pending_ids.append(identity)
            identities.append(identity);scenes.append(entry['scene']);scales.append(float(inputs['scale']));gaps.append(k-q);accepted+=1
            if len(pending)==16:flush()
        reader.cache_clear();print('REGIONS',split,realization,entry['scene'],accepted,flush=True)
    flush()
    metadata=dict(ids=np.asarray(identities),scenes=np.asarray(scenes),scales=np.asarray(scales),frame_gaps=np.asarray(gaps))
    base={k:np.concatenate(v) for k,v in collected.items()};base.update(metadata)
    sparse={k:np.concatenate(v) for k,v in sparse_collected.items()} if sparse_collected else None
    if sparse is not None:sparse.update(metadata)
    return base,sparse,sample,dict(excluded=excluded,seconds=time.time()-started)


def strata(scores,thresholds):
    return (2*(scores[...,0]>=thresholds[0])+(scores[...,1]>=thresholds[1])).astype(np.int64)


def plan():
    from experiments.round60_common import checked as old_check,OUT as OLD
    from experiments.round65_common import checked as classification_check
    old=old_check();classification=classification_check()
    if OUT.exists():raise ValueError('preserve prior preparation')
    sources=dict(old['sources']);sources.update(classification['sources'])
    names=('docs/round68_fixed_region_plan.md','experiments/round68_fixed_region_data.py',
           'models/round68_fixed_region.py','models/round61_normal_relations.py',
           'tests/test_round68_fixed_region.py')
    sources.update({name:sha(ROOT/name) for name in names})
    write(OUT/'preparation_plan.json',dict(created_unix=time.time(),sources=sources,
        original_scene_lock_path=str(OLD/'lock.json'),original_scene_lock_sha256=sha(OLD/'lock.json'),
        splits={'fit':576,'val':64,'old_eval':480},all_reference_queries_preserved_under_thinning=True,
        used_development_only=True,goal_complete=False))
    print('R68_PREPARATION_PLAN',sha(OUT/'preparation_plan.json'),flush=True)


def check():
    p=read(OUT/'preparation_plan.json')
    for name,digest in p['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('preparation source changed '+name)
    if sha(p['original_scene_lock_path'])!=p['original_scene_lock_sha256']:
        raise ValueError('original scene identities changed')
    return p


def prepare():
    p=check();lock=read(p['original_scene_lock_path'])
    destination=OUT/'data'
    if destination.exists():raise ValueError('preserve previous preparation output')
    destination.mkdir();all_arrays={};samples={};records={}
    for name,split,realization in (('fit0','fit',0),('fit1','fit',1),('val','val',0),('eval_base','old_eval',0)):
        base,sparse,sample,meta=collect(lock,split,realization)
        if len(base['ids'])!=p['splits'][split]:raise ValueError('original pair population changed')
        all_arrays[name]=base;samples[name]=sample;records[name]=meta
        if sparse is not None:
            all_arrays['eval_sparse256']=sparse
            records['eval_sparse256']=dict(meta,controlled_known_region_observation_test=True)
        if name=='fit1':
            scores=np.concatenate([all_arrays[n]['geometry_scores'].reshape(-1,2) for n in ('fit0','fit1')])
            thresholds=np.median(scores,axis=0)
            write(destination/'strata_thresholds.json',dict(thresholds=thresholds.tolist(),
                source='all fit0/fit1 input queries, including those without valid cross-view labels',
                query_count=len(scores),statistic='fixed componentwise median'))
        print('COLLECTED',name,len(base['ids']),flush=True)
    thresholds=np.asarray(read(destination/'strata_thresholds.json')['thresholds'])
    manifest={}
    for name,arrays in all_arrays.items():
        arrays['strata']=strata(arrays['geometry_scores'],thresholds)
        if any(not np.isfinite(v).all() for k,v in arrays.items() if k not in ('ids','scenes')):
            raise ValueError('nonfinite regional inputs')
        path=destination/(name+'.npz');np.savez_compressed(path,**arrays)
        overlap=arrays['overlap'];valid=np.stack((overlap.max(-1)>=.3,overlap.max(-2)>=.3),axis=1)
        counts={str(group):dict(all=int((arrays['strata']==group).sum()),
                               valid=int(((arrays['strata']==group)&valid).sum())) for group in range(4)}
        record=dict(records[name],sha256=sha(path),count=len(arrays['ids']),valid_queries=int(valid.sum()),
            all_queries=int(valid.size),valid_fraction=float(valid.mean()),strata_counts=counts,
            units='pair-normalized scene units',covariance_source='observed region moments',
            sensor_measurement_noise_in_labels=True,pose_in_forward=False)
        if 'surviving_counts' in arrays:
            record['surviving_count_quantiles']=np.quantile(arrays['surviving_counts'],[0,.1,.5,.9,1]).tolist()
        write(path.with_suffix('.json'),record)
        manifest[name]=dict(path=str(path),sha256=sha(path),record_sha256=sha(path.with_suffix('.json')))
    for name,sample in samples.items():np.savez_compressed(destination/(name+'_sample.npz'),**sample)
    a,b=(all_arrays[name] for name in ('eval_base','eval_sparse256'))
    if any(not np.array_equal(a[key],b[key]) for key in ('ids','scenes','overlap','world_centers','strata','permutation')):
        raise ValueError('sparse condition changed reference task')
    write(destination/'manifest.json',dict(caches=manifest,thresholds_sha256=sha(destination/'strata_thresholds.json'),
        preparation_plan_sha256=sha(OUT/'preparation_plan.json'),fixed_sparse_reference_verified=True,
        fresh_confirmation=False,goal_complete=False))
    print('R68_DATA_COMPLETE',[(n,read(destination/(n+'.json'))['valid_fraction']) for n in manifest],flush=True)


def load_cache(name):
    check();manifest=read(OUT/'data/manifest.json')
    if name not in manifest['caches']:raise ValueError('undeclared cache')
    record=manifest['caches'][name];path=Path(record['path'])
    if sha(path)!=record['sha256']:raise ValueError('cache changed')
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('plan','prepare'))
    args=parser.parse_args();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    globals()[args.action]()
