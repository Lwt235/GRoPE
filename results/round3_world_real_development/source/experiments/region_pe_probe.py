"""Soft regional correspondence: independently sampled support, pure Q/K PE."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from models.gaussian_pe import GaussianPEMatcher,REGION_METHODS
from models.gaussian_world_pe import GaussianWorldMatcher,METHODS as WORLD_METHODS
from experiments.region_support_data import make_dataset,CONDITIONS
from experiments.matching_probe import digest
from experiments.rendered_probe import source_snapshot


def real_dataset(root,scene,split,alternate=False):
    folder=Path(root)/scene
    manifest=json.loads((folder/'manifest.json').read_text())
    rows=[]
    ids=[]
    suffix='_alt' if alternate else ''
    for rec in manifest['pairs']:
        if rec['split']!=split: continue
        with np.load(folder/rec['file']) as f:
            d={k:f[k].copy() for k in f.files}
        label=d['soft_labels'].copy()
        key_valid=d['k_valid'+suffix].astype(bool)
        kept=(label*key_valid[None]).sum(-1)
        valid=d['label_valid']&d['q_valid'+suffix]&(kept>=.95)
        label=label*key_valid[None]/np.maximum(kept[:,None],1e-10)
        # Pixels are at 192x144, with 24px square support. All projected
        # quantities use the same dimensionless patch units as simulation.
        rows.append({'q':d['q_features'],'k':d['k_features'],
                     'mq':d['q_mean'+suffix],'mk':d['k_mean'+suffix],
                     'cq':d['q_cov'+suffix],'ck':d['k_cov'+suffix],
                     'pixq':d['q_pixel_mean']/24,'k_to_q':d['k_projected_mean'+suffix]/24,
                     'projected_qcov':d['q_pixel_cov']/24**2,
                     'projected_kcov':d['k_projected_cov'+suffix]/24**2+np.eye(2)*1e-6,
                     'soft_labels':label,'label_valid':valid,'key_valid':key_valid,
                     'q_valid':d['q_valid'+suffix].astype(bool),
                     'retained_label_mass':kept})
        ids.append({'q_id':rec['q_id'],'k_id':rec['k_id'],'file':rec['file'],'sha256':rec['npz_sha256']})
    data={k:torch.as_tensor(np.stack([r[k] for r in rows]),dtype=torch.bool if k in ['label_valid','key_valid','q_valid'] else torch.float32) for k in rows[0]}
    return data,{'scene':scene,'split':split,'alternate':alternate,'pairs':ids,
                 'valid_query_rows':int(data['label_valid'].sum()),
                 'label_rule':'at least 4 observed common tracks, valid q geometry, >=95% label mass on valid keys; condition on valid key support',
                 'scene_manifest_sha256':hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()}


def soft_loss(logits,data):
    score=logits[...,:-1]
    if 'key_valid' in data: score=score.masked_fill(~data['key_valid'][...,None,:],-10000.)
    losses=-(data['soft_labels']*score.log_softmax(-1)).sum(-1)
    return losses[data['label_valid']].mean()


@torch.no_grad()
def evaluate(model,data,batch=32):
    outputs=[]
    for start in range(0,len(data['q']),batch):
        part={k:v[start:start+batch] for k,v in data.items()}
        score=model(part)[...,:-1]
        if 'key_valid' in part: score=score.masked_fill(~part['key_valid'][...,None,:],-10000.)
        outputs.append(score.softmax(-1).cpu())
    p=torch.cat(outputs).double()
    y=data['soft_labels'].cpu().double()
    valid=data['label_valid'].cpu()
    ce=-(y*p.clamp_min(1e-30).log()).sum(-1)
    entropy=-(y*y.clamp_min(1e-30).log()).sum(-1)
    kl=ce-entropy
    js=.5*(y*(y.clamp_min(1e-30)/((p+y)/2).clamp_min(1e-30)).log()+p*(p.clamp_min(1e-30)/((p+y)/2).clamp_min(1e-30)).log()).sum(-1)
    topmass=y.gather(-1,p.argmax(-1)[...,None])[...,0]
    hit=(p.argmax(-1)==y.argmax(-1)).double()
    out={'cross_entropy':float(ce[valid].mean()),'kl':float(kl[valid].mean()),
         'js':float(js[valid].mean()),'top1_target_mass':float(topmass[valid].mean()),
         'dominant_region_accuracy':float(hit[valid].mean()),'valid_query_rows':int(valid.sum())}
    per={}
    if not valid.any(): raise ValueError('evaluation has no valid query rows')
    for name,value in [('kl',kl),('top1_target_mass',topmass),('dominant_region_accuracy',hit),('js',js)]:
        count=valid.sum(-1).numpy()
        per[name]=np.divide((value*valid).sum(-1).numpy(),count,
                            out=np.full(len(count),np.nan),where=count>0)
    per['valid_rows']=valid.sum(-1).numpy()
    return out,per


def main(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    methods=args.methods.split(',') if args.methods else REGION_METHODS
    seeds=[int(s) for s in args.seeds.split(',')]
    manifest={'config':vars(args),'source_hashes':source_snapshot(out),'records':[],'datasets':{}}
    per_scene={}
    histories=[]
    started=time.time()
    for seed in seeds:
        if args.domain=='synthetic':
            base=3100 if args.phase=='development' else 4100
            cache={}
            train,stats=make_dataset(base+seed,args.train_scenes,'oracle',cache)
            val,_=make_dataset(base+100+seed,args.val_scenes,'oracle',cache)
            tests={c:make_dataset(base+200+seed,args.test_scenes,c,cache)[0] for c in CONDITIONS}
            meta={'geometry':'independently fit rendered surface moments, not predicted RGB geometry','stats':stats}
            del cache
        else:
            train,tm=real_dataset(args.real_root,args.scene,'train')
            val,vm=real_dataset(args.real_root,args.scene,'validation')
            tests={}
            testmeta={}
            if args.phase=='confirmation':
                for scene in ['teatime','waldo_kitchen']:
                    for alternate in [False,True]:
                        key=scene+('_alt' if alternate else '')
                        tests[key],testmeta[key]=real_dataset(args.real_root,scene,'test',alternate)
            meta={'train':tm,'validation':vm,'test':testmeta,
                  'scope':'real photos and supplied trained GS; sparse observed SfM track distributions; no new-view reconstruction claim'}
        manifest['datasets'][str(seed)]={'train':digest(train),'validation':digest(val),'test':{c:digest(d) for c,d in tests.items()},'meta':meta}
        train={k:v.to(device) for k,v in train.items()}
        val={k:v.to(device) for k,v in val.items()}
        tests={c:{k:v.to(device) for k,v in d.items()} for c,d in tests.items()}
        eligible=train['label_valid'].any(-1).nonzero()[:,0].cpu()
        gen=torch.Generator().manual_seed(380000+seed)
        batches=eligible[torch.randint(0,len(eligible),(args.steps,args.batch),generator=gen)].to(device)
        for method in methods:
            torch.manual_seed(seed)
            if method in WORLD_METHODS:
                model=GaussianWorldMatcher(method).to(device)
            else:
                model=GaussianPEMatcher(method,args.features,args.resolution).to(device)
            opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
            start=time.time()
            # The small real data can overfit within the first 100 steps.
            # Include the common untrained state and frequent early validation.
            best_val,_=evaluate(model,val)
            best=best_val['cross_entropy']
            best_step=0
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            histories.append({'seed':seed,'method':method,'step':0,'validation':best_val})
            for step in range(args.steps):
                part={k:v[batches[step]] for k,v in train.items()}
                loss=soft_loss(model(part),part)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                opt.step()
                if (step+1)%args.val_every==0 or step+1==args.steps:
                    metrics,_=evaluate(model,val)
                    histories.append({'seed':seed,'method':method,'step':step+1,'train_loss':float(loss),'validation':metrics})
                    if metrics['cross_entropy']<best:
                        best=metrics['cross_entropy']
                        best_step=step+1
                        best_val=metrics
                        best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(best_state)
            test_results={}
            for condition,d in tests.items():
                metrics,per=evaluate(model,d)
                test_results[condition]=metrics
                for metric,values in per.items(): per_scene[f'{seed}/{method}/{condition}/{metric}']=values
            record={'seed':seed,'method':method,'step':best_step,'best_validation':best_val,
                    'seconds':time.time()-start,'parameters':sum(p.numel() for p in model.parameters()),
                    'temperature':float(model.logtemp.exp()),'test':test_results}
            manifest['records'].append(record)
            checkpoint=out/'checkpoints'/f'{seed}-{method}.pt'
            checkpoint.parent.mkdir(exist_ok=True)
            torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'method':method,'features':args.features,'resolution':args.resolution},checkpoint)
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            (out/'training_history.json').write_text(json.dumps(histories,indent=2)+'\n')
            np.savez_compressed(out/'per_scene.npz',**per_scene)
            print(json.dumps(record),flush=True)
    manifest['elapsed_seconds']=time.time()-started
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('COMPLETE',manifest['elapsed_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cpu')
    p.add_argument('--domain',choices=['synthetic','real'],default='synthetic')
    p.add_argument('--phase',choices=['development','confirmation'],default='development')
    p.add_argument('--out',default='results/round3_region_development')
    p.add_argument('--real-root',default='results/round3_real_support_data_adjacent')
    p.add_argument('--scene',default='teatime')
    p.add_argument('--seeds',default='91')
    p.add_argument('--methods',default='')
    p.add_argument('--features',type=int,default=256)
    p.add_argument('--resolution',type=float,default=0.)
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--val-every',type=int,default=100)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--train-scenes',type=int,default=192)
    p.add_argument('--val-scenes',type=int,default=48)
    p.add_argument('--test-scenes',type=int,default=96)
    main(p.parse_args())
