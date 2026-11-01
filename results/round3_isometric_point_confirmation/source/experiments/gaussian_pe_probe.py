"""Fresh-data, parameter-matched Gaussian per-token PE comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from models.gaussian_pe import GaussianPEMatcher,METHODS
from experiments import rendered_data
from experiments.rendered_probe import concatenate,source_snapshot
from experiments.rendered_metrics import evaluate
from experiments.matching_probe import digest
from experiments.rgb_geometry_probe import RGBGeometry,observation_from_rgb


def write_summary(records,out):
    summary=[]
    for regime,method in sorted(set((r['regime'],r['method']) for r in records)):
        rows=[r for r in records if r['regime']==regime and r['method']==method]
        for condition in rows[0]['test']:
            d={'regime':regime,'method':method,'condition':condition,'seeds':len(rows)}
            for key in rows[0]['test'][condition]:
                values=[r['test'][condition][key] for r in rows if r['test'][condition][key] is not None]
                if values:
                    d[key]=float(np.mean(values))
                    d[key+'_seed_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
            summary.append(d)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def main(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    seeds=[int(s) for s in args.seeds.split(',')]
    methods=args.methods.split(',') if args.methods else METHODS
    namespace=1100 if args.phase=='development' else 2100
    predictor_path=Path('results/round2_rgb_geometry/checkpoints/geometry-11.pt')
    predictor=RGBGeometry()
    predictor.load_state_dict(torch.load(predictor_path,map_location='cpu',weights_only=True)['state_dict'])
    manifest={'config':vars(args),'source_hashes':source_snapshot(out),'records':[],
              'datasets':{},'geometry_stats':{},'predictor_sha256':hashlib.sha256(predictor_path.read_bytes()).hexdigest(),
              'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
              'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'protocol_sha256':hashlib.sha256(Path('ROUND3_PROTOCOL.md').read_bytes()).hexdigest()}
    per_scene={}
    history=[]
    started=time.time()
    for seed in seeds:
        cache={}
        train_parts=[]
        val_parts=[]
        for condition in ['oracle','mixed','depth_fit']:
            train,_=rendered_data.make_dataset(namespace+seed,args.train_scenes,condition,cache)
            val,_=rendered_data.make_dataset(namespace+100+seed,args.val_scenes,condition,cache)
            train_parts.append(train)
            val_parts.append(val)
        tests={}
        for condition in rendered_data.CONDITIONS:
            tests[condition],stats=rendered_data.make_dataset(namespace+200+seed,args.test_scenes,condition,cache)
            manifest['geometry_stats'][f'{seed}/{condition}']=stats
        original=rendered_data.patch_observation
        try:
            rendered_data.patch_observation=lambda view,cx,cy,rng,condition:observation_from_rgb(predictor,view,cx,cy,rng,condition)
            tests['rgb_pred'],stats=rendered_data.make_dataset(namespace+200+seed,args.test_scenes,'rgb_pred',cache)
            manifest['geometry_stats'][f'{seed}/rgb_pred']=stats
        finally:
            rendered_data.patch_observation=original
        del cache
        for regime in args.regimes.split(','):
            train=train_parts[0] if regime=='oracle_train' else concatenate(train_parts)
            val=val_parts[0] if regime=='oracle_train' else concatenate(val_parts)
            manifest['datasets'][f'{regime}/{seed}']={'train':digest(train),'validation':digest(val),'test':{c:digest(t) for c,t in tests.items()}}
            # Keep only fields used by the PE or evaluator, avoiding unused pair
            # biases from round 2. No labels/visibility enter positional maps.
            keys=['q','k','mq','mk','pixq','k_to_q','projected_kcov','labels','pixel_error']
            tr={k:train[k].to(device) for k in keys}
            va={k:val[k].to(device) for k in keys}
            ts={c:{k:d[k].to(device) for k in keys} for c,d in tests.items()}
            gen=torch.Generator().manual_seed(370000+seed)
            batches=torch.randint(0,len(tr['q']),(args.steps,args.batch),generator=gen).to(device)
            for method in methods:
                torch.manual_seed(seed)
                if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
                model=GaussianPEMatcher(method,args.features,args.resolution).to(device)
                opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
                best=float('inf')
                best_state=None
                start=time.time()
                for step in range(args.steps):
                    part={k:v[batches[step]] for k,v in tr.items()}
                    logits=model(part)
                    loss=F.cross_entropy(logits.flatten(0,1),part['labels'].flatten())
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                    opt.step()
                    if (step+1)%args.val_every==0 or step+1==args.steps:
                        metrics,_,_=evaluate(model,va)
                        history.append({'seed':seed,'regime':regime,'method':method,'step':step+1,'train_loss':float(loss),'validation':metrics,'grad_norm':float(grad)})
                        if metrics['loss']<best:
                            best=metrics['loss']
                            best_step=step+1
                            best_validation=metrics
                            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
                model.load_state_dict(best_state)
                test_results={}
                for condition,data in ts.items():
                    metrics,scenes,probability=evaluate(model,data)
                    test_results[condition]=metrics
                    per_scene[f'{regime}/{seed}/{method}/{condition}']=scenes
                record={'seed':seed,'regime':regime,'method':method,'step':best_step,'best_validation':best_validation,
                        'seconds':time.time()-start,'parameters':sum(p.numel() for p in model.parameters()),
                        'peak_cuda_mb':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
                        'temperature':float(model.logtemp.exp()),'test':test_results}
                manifest['records'].append(record)
                checkpoint=out/'checkpoints'/f'{regime}-{seed}-{method}.pt'
                checkpoint.parent.mkdir(exist_ok=True)
                torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'method':method,'features':args.features},checkpoint)
                np.savez_compressed(out/'per_scene.npz',**per_scene)
                (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                (out/'training_history.json').write_text(json.dumps(history,indent=2)+'\n')
                write_summary(manifest['records'],out)
                print(json.dumps({'seed':seed,'regime':regime,'method':method,'step':best_step,
                                  'oracle_r1':test_results['oracle']['visible_r1'],'mixed_r1':test_results['mixed']['visible_r1'],
                                  'rgb_r1':test_results['rgb_pred']['visible_r1'],'unmatched_ap':test_results['oracle']['unmatched_auprc'],
                                  'seconds':record['seconds']}),flush=True)
            del tr,va,ts
    manifest['elapsed_seconds']=time.time()-started
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('COMPLETE',manifest['elapsed_seconds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--phase',choices=['development','confirmation'],default='development')
    parser.add_argument('--out',default='results/round3_development')
    parser.add_argument('--seeds',default='91')
    parser.add_argument('--methods',default='')
    parser.add_argument('--regimes',default='oracle_train,robust_train')
    parser.add_argument('--features',type=int,default=256)
    parser.add_argument('--resolution',type=float,default=0.)
    parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--train-scenes',type=int,default=192)
    parser.add_argument('--val-scenes',type=int,default=48)
    parser.add_argument('--test-scenes',type=int,default=96)
    parser.add_argument('--val-every',type=int,default=100)
    main(parser.parse_args())
