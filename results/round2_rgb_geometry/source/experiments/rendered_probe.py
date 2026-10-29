"""Two-stage rendered matching experiment with conditional reliability gates.

1. Oracle training, independently corrupted geometry at test time.
2. Shared mixed geometry training/validation, shared noisy-depth estimator test.
Gate runs require a predeclared reproducible normal gain in stage 1.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch
import torch.nn.functional as F
from models.rendered_attention import RenderedMatcher,METHODS,GATE_METHODS
from experiments.rendered_data import CONDITIONS,make_dataset
from experiments.rendered_metrics import evaluate
from experiments.matching_probe import digest


ROBUST=['position_rbf','rope3d','rope_normal','rope_normal_anisotropy',
        'rope_covariance','rope_normal_uncertainty']


def concatenate(datasets):
    return {k:torch.cat([d[k] for d in datasets]) for k in datasets[0]}


def source_snapshot(out):
    hashes={}
    for folder in ['models','experiments','tests']:
        for path in sorted(Path(folder).glob('*.py')):
            target=out/'source'/path
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(path.read_bytes())
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def gate_trigger(records,seeds):
    differences=[]
    for seed in seeds:
        picked={r['method']:r for r in records if r['seed']==seed and r['regime']=='oracle_train'}
        if 'rope_normal' not in picked or 'rope3d' not in picked: return False,[]
        differences.append(picked['rope_normal']['best_validation']['visible_r1']-picked['rope3d']['best_validation']['visible_r1'])
    return bool(np.mean(differences)>=.005 and np.mean(np.array(differences)>0)>=.8),differences


def summarize(records,out):
    rows=[]
    keys=sorted(set((r['regime'],r['method']) for r in records))
    for regime,method in keys:
        selected=[r for r in records if r['regime']==regime and r['method']==method]
        for condition in CONDITIONS:
            row={'regime':regime,'method':method,'condition':condition,'seeds':len(selected)}
            for metric in selected[0]['test'][condition]:
                values=[r['test'][condition][metric] for r in selected if r['test'][condition][metric] is not None]
                if not values: continue
                row[metric]=float(np.mean(values))
                row[metric+'_seed_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
            rows.append(row)
    (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'summary.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    seeds=[int(s) for s in args.seeds.split(',')]
    manifest={'config':vars(args),'source_hashes':source_snapshot(out),
              'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
              'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
              'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'gate_trigger_rule':'mean oracle-validation normal gain >= 0.5 percentage points and positive in at least 80% of seeds; test results do not trigger gates',
              'geometry_provider':'oracle planes / explicit corruption / local PCA from noisy rendered depth. No RGB-trained geometry predictor.',
              'paper_baseline_scope':{'hrope_calibrated_qk':'Calibrated Eq.1/4/17 attention-logit operator, normalized coordinates, supplied planes, fixed cameras. No learned 13D head or V/output/NVS stack.',
                                      'worldrope_mixture_qk':'Eq.3-10 frustum covariance, world projection, 3-component expectation. Analytic unimodal depth quadrature, not learned multi-depth or layer refinement. Q/K score only.'},
              'datasets':{},'geometry_stats':{},'records':[]}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    started=time.time()
    per_scene={}
    histories=[]
    for regime in ['oracle_train','robust_train']:
        methods=METHODS.copy() if regime=='oracle_train' else ROBUST.copy()
        if args.methods: methods=args.methods.split(',')
        if regime=='robust_train':
            permitted,differences=gate_trigger(manifest['records'],seeds)
            manifest['gate_trigger']={'enabled':permitted,'oracle_normal_gain_by_seed':differences}
            if permitted and not args.methods: methods+=GATE_METHODS
            print('GATE_TRIGGER',json.dumps(manifest['gate_trigger']),flush=True)
        for seed in seeds:
            cache={}
            if regime=='oracle_train':
                train,train_stats=make_dataset(600+seed,args.train_scenes,'oracle',cache)
                val,val_stats=make_dataset(700+seed,args.val_scenes,'oracle',cache)
            else:
                train_parts=[]
                val_parts=[]
                train_stats=[]
                val_stats=[]
                for condition in ['oracle','mixed','depth_fit']:
                    d,s=make_dataset(600+seed,args.train_scenes,condition,cache)
                    train_parts.append(d)
                    train_stats+=s
                    d,s=make_dataset(700+seed,args.val_scenes,condition,cache)
                    val_parts.append(d)
                    val_stats+=s
                train,val=concatenate(train_parts),concatenate(val_parts)
            tests={}
            test_stats={}
            for condition in CONDITIONS:
                tests[condition],test_stats[condition]=make_dataset(800+seed,args.test_scenes,condition,cache)
            key=f'{regime}/{seed}'
            manifest['datasets'][key]={'train':digest(train),'validation':digest(val),'test':{c:digest(d) for c,d in tests.items()}}
            manifest['geometry_stats'][key]={'train':train_stats,'validation':val_stats,'test':test_stats}
            # Release dense render buffers; only model tensors remain resident.
            del cache
            train={k:v.to(device) for k,v in train.items()}
            val={k:v.to(device) for k,v in val.items()}
            tests={c:{k:v.to(device) for k,v in d.items()} for c,d in tests.items()}
            generator=torch.Generator().manual_seed(240000+seed)
            batches=torch.randint(0,len(train['q']),(args.steps,args.batch),generator=generator).to(device)
            for method in methods:
                torch.manual_seed(seed)
                if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
                model=RenderedMatcher(method).to(device)
                content=[p for n,p in model.named_parameters() if n!='weights' and not n.startswith('gate.')]
                geometry=[p for n,p in model.named_parameters() if n=='weights' or n.startswith('gate.')]
                optimizer=torch.optim.AdamW([{'params':content,'lr':.002},{'params':geometry,'lr':.01}],weight_decay=.0001)
                best_loss=float('inf')
                best_state=None
                best_step=None
                best_validation=None
                model_started=time.time()
                for step in range(args.steps):
                    part={k:v[batches[step]] for k,v in train.items()}
                    logits=model(part)
                    loss=F.cross_entropy(logits.flatten(0,1),part['labels'].flatten())
                    assert torch.isfinite(loss),(regime,seed,method,step)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                    optimizer.step()
                    if (step+1)%args.val_every==0 or step+1==args.steps:
                        metrics,_,_=evaluate(model,val)
                        histories.append({'regime':regime,'seed':seed,'method':method,'step':step+1,'train_loss':loss.item(),
                                          'grad_norm':float(grad),'validation':metrics})
                        if metrics['loss']<best_loss:
                            best_loss=metrics['loss']
                            best_step=step+1
                            best_validation=metrics
                            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
                model.load_state_dict(best_state)
                test_results={}
                predictions={}
                for condition,data in tests.items():
                    metrics,scenes,probability=evaluate(model,data)
                    test_results[condition]=metrics
                    per_scene[f'{regime}/{seed}/{method}/{condition}']=scenes
                    predictions[condition+'/probability']=probability.astype(np.float16)
                    predictions[condition+'/labels']=data['labels'].cpu().numpy().astype(np.int16)
                checkpoint=out/'checkpoints'/f'{regime}-{seed}-{method}.pt'
                checkpoint.parent.mkdir(exist_ok=True)
                torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'method':method,'regime':regime,'seed':seed,'step':best_step},checkpoint)
                predfile=out/'predictions'/f'{regime}-{seed}-{method}.npz'
                predfile.parent.mkdir(exist_ok=True)
                np.savez_compressed(predfile,**predictions)
                record={'regime':regime,'seed':seed,'method':method,'step':best_step,'validation_loss':best_loss,
                        'best_validation':best_validation,
                        'parameters':sum(p.numel() for p in model.parameters()),
                        'effective_geometry_weights':F.softplus(model.weights).detach().cpu().tolist(),
                        'seconds':time.time()-model_started,
                        'peak_cuda_allocated_mb':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
                        'test':test_results}
                manifest['records'].append(record)
                print(json.dumps({'regime':regime,'seed':seed,'method':method,'step':best_step,
                                  'oracle_r1':test_results['oracle']['visible_r1'],
                                  'depth_fit_r1':test_results['depth_fit']['visible_r1'],
                                  'mixed_r1':test_results['mixed']['visible_r1'],
                                  'unmatched_ap':test_results['oracle']['unmatched_auprc'],
                                  'seconds':record['seconds']}),flush=True)
                np.savez_compressed(out/'per_scene.npz',**per_scene)
                (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                (out/'training_history.json').write_text(json.dumps(histories,indent=2)+'\n')
                summarize(manifest['records'],out)
    manifest['elapsed_seconds']=time.time()-started
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('COMPLETE',out,manifest['elapsed_seconds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--out',default='results/round2_rendered')
    parser.add_argument('--seeds',default='11,22,33,44,55')
    parser.add_argument('--train-scenes',type=int,default=192)
    parser.add_argument('--val-scenes',type=int,default=48)
    parser.add_argument('--test-scenes',type=int,default=96)
    parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--val-every',type=int,default=100)
    parser.add_argument('--methods',default='')
    run(parser.parse_args())
