"""One supervised RGB geometry predictor shared by all matching ablations.

This is a small synthetic-domain MLP, not a pretrained geometry foundation model.
It predicts camera-z depth, sign-free camera normal and log-depth variance from
RGB patches plus known image coordinates. Test GT is used only in diagnostics.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from experiments.rendered_scene import pair,COLS,ROWS,PATCH,WIDTH,HEIGHT,FOCAL
from experiments import rendered_data
from experiments.rendered_data import unit,world_rays
from experiments.rendered_metrics import evaluate
from experiments.rendered_probe import source_snapshot
from models.rendered_attention import RenderedMatcher


def input_features(view):
    cx,cy=np.meshgrid(np.arange(COLS)*PATCH+PATCH//2,np.arange(ROWS)*PATCH+PATCH//2)
    coords=np.stack([(cx.flatten()-WIDTH/2)/FOCAL,(cy.flatten()-HEIGHT/2)/FOCAL],-1)
    rgb=rendered_data.patch_features(view['rgb'])
    context=np.full((ROWS*COLS,1),view['rgb'].mean())
    return torch.tensor(np.concatenate([rgb,coords,context],-1),dtype=torch.float32)


class RGBGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(111,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,5))

    def forward(self,x):
        p=self.net(x)
        logdepth=p[...,0].clamp(-1,3)
        normal=F.normalize(p[...,1:4]+torch.tensor([0.,0.,1.],device=x.device),dim=-1)
        logvar=p[...,4].clamp(-8,1)
        return logdepth,normal,logvar


def loss(model,data):
    logdepth,normal,logvar=model(data['x'])
    error=logdepth-data['z'].log()
    nll=.5*(error.square()*(-logvar).exp()+logvar)
    normal_loss=1-(normal*data['n']).sum(-1).square()
    return (nll+.5*normal_loss).mean()


def supervised_data(seed,scenes):
    xs,zs,ns=[],[],[]
    for index in range(scenes):
        a,b,gt=pair(seed*1000+index)
        cx,cy=gt['center_x'],gt['center_y']
        for view in [a,b]:
            xs.append(input_features(view))
            zs.append(torch.tensor(view['depth'][cy,cx]))
            ns.append(torch.tensor(view['normal'][cy,cx]@view['rotation'].T))
    return {'x':torch.cat(xs),'z':torch.cat(zs),'n':torch.cat(ns)}


def observation_from_rgb(model,view,cx,cy,rng,condition):
    with torch.no_grad():
        logdepth,normal_camera,logvar=model(input_features(view))
    depth=logdepth.exp().numpy().astype(np.float64)
    sigma=np.clip(depth*np.exp(.5*logvar.numpy()),.01,1.5)
    rotation,translation=view['rotation'],view['translation']
    origin=-rotation.T@translation
    normal=unit(normal_camera.numpy().astype(np.float64)@rotation)
    ray=world_rays(cx,cy,rotation)
    mean=origin+ray*depth[:,None]
    # Predicted neighbor centers only: no GT/noisy-GT depth quality signal.
    grid=np.stack([cx,cy],-1)
    neighbor=(np.abs(grid[:,None]-grid[None,:]).max(-1)<=PATCH)&(~np.eye(len(cx),dtype=bool))
    distance=np.einsum('nmi,ni->nm',mean[None]-mean[:,None],normal)
    residual=np.sqrt((distance**2*neighbor).sum(-1)/neighbor.sum(-1))
    reference=np.tile(np.array([0.,0.,1.]),(len(cx),1))
    reference[np.abs(normal[:,2])>.85]=[1.,0.,0.]
    u=unit(np.cross(normal,reference))
    v=np.cross(normal,u)
    basis=np.stack([u,v],-1)
    camera_mean=mean@rotation.T+translation
    x,y,z=camera_mean.T
    jac=np.zeros((len(cx),2,3))
    jac[:,0,0],jac[:,1,1]=FOCAL/z,FOCAL/z
    jac[:,0,2],jac[:,1,2]=-FOCAL*x/z**2,-FOCAL*y/z**2
    jac=jac@rotation
    aa=jac@basis
    local=np.linalg.inv(aa.transpose(0,2,1)@aa+np.eye(2)*.01)*(PATCH**2/12)
    cov=basis@local@basis.transpose(0,2,1)+normal[:,:,None]*normal[:,None,:]*.005**2
    projected=jac@cov@jac.transpose(0,2,1)
    support_error=np.abs(np.log(np.maximum(np.trace(projected,axis1=1,axis2=2),1e-8)/(PATCH**2/6)))
    distance=(normal*mean).sum(-1)
    safe=np.where(np.abs(distance)<.03,np.copysign(.03,distance+1e-12),distance)
    h=rotation[None]+translation[None,:,None]*normal[:,None,:]/safe[:,None,None]
    fallback=(np.linalg.cond(h)>1e4)
    h[fallback]=rotation
    return {'mean':mean,'normal':normal,'cov':cov,
            'uncertainty':ray[:,:,None]*ray[:,None,:]*sigma[:,None,None]**2,
            'residual':residual,'support_error':support_error,'sigma':sigma,
            'homography':h,'homography_fallback':fallback.astype(np.float32),
            'depth':depth,'ray':ray,'rotation':rotation,'translation':translation,
            # These two fields are consumed only by dataset diagnostic stats.
            'true_mean':view['points'][cy,cx],'true_normal':view['normal'][cy,cx]}


def main(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    seeds=[11,22,33,44,55]
    manifest={'config':vars(args),'source_hashes':source_snapshot(out),'predictors':[],
              'scope':'Shared small RGB patch geometry predictor supervised on training scenes only; no external model or test depth inputs.'}
    if not args.eval_only:
        for seed in seeds:
            started=time.time()
            train={k:v.to(device) for k,v in supervised_data(600+seed,192).items()}
            val={k:v.to(device) for k,v in supervised_data(700+seed,48).items()}
            torch.manual_seed(seed)
            model=RGBGeometry().to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
            generator=torch.Generator().manual_seed(910000+seed)
            best=float('inf')
            best_state=None
            history=[]
            for step in range(args.steps):
                indices=torch.randint(0,len(train['x']),(512,),generator=generator).to(device)
                batch={k:v[indices] for k,v in train.items()}
                current=loss(model,batch)
                optimizer.zero_grad(set_to_none=True)
                current.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step()
                if (step+1)%100==0 or step+1==args.steps:
                    with torch.no_grad(): score=float(loss(model,val))
                    history.append({'step':step+1,'train_loss':float(current),'validation_loss':score})
                    if score<best:
                        best,best_step=score,step+1
                        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            checkpoint=out/'checkpoints'/f'geometry-{seed}.pt'
            checkpoint.parent.mkdir(exist_ok=True)
            torch.save({'state_dict':best_state,'step':best_step,'seed':seed},checkpoint)
            record={'seed':seed,'step':best_step,'validation_loss':best,'history':history,'seconds':time.time()-started}
            manifest['predictors'].append(record)
            print('RGB_PREDICTOR',json.dumps({k:v for k,v in record.items() if k!='history'}),flush=True)
        (out/'predictor_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.train_only: return
    matching_manifest=json.loads(Path(args.matching,'manifest.json').read_text())
    assert 'elapsed_seconds' in matching_manifest,'Matching training must finish before evaluation.'
    results=[]
    per_scene={}
    geo_stats={}
    original=rendered_data.patch_observation
    for seed in seeds:
        model=RGBGeometry()
        model.load_state_dict(torch.load(out/'checkpoints'/f'geometry-{seed}.pt',map_location='cpu',weights_only=True)['state_dict'])
        rendered_data.patch_observation=lambda view,cx,cy,rng,condition: observation_from_rgb(model,view,cx,cy,rng,condition)
        dataset,stats=rendered_data.make_dataset(800+seed,96,'rgb_pred')
        geo_stats[str(seed)]=stats
        for record in matching_manifest['records']:
            if record['seed']!=seed: continue
            regime,method=record['regime'],record['method']
            matcher=RenderedMatcher(method)
            checkpoint=Path(args.matching,'checkpoints',f'{regime}-{seed}-{method}.pt')
            matcher.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'])
            metrics,scenes,_=evaluate(matcher,dataset)
            results.append({'seed':seed,'regime':regime,'method':method,'metrics':metrics})
            per_scene[f'{regime}/{seed}/{method}']=scenes
        rendered_data.patch_observation=original
        print('RGB_EVALUATED',seed,flush=True)
    summary=[]
    for regime,method in sorted(set((r['regime'],r['method']) for r in results)):
        rows=[r['metrics'] for r in results if r['regime']==regime and r['method']==method]
        entry={'regime':regime,'method':method}
        for key in rows[0]:
            values=[r[key] for r in rows if r[key] is not None]
            if values:
                entry[key]=float(np.mean(values))
                entry[key+'_seed_std']=float(np.std(values,ddof=1))
        summary.append(entry)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'records.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'geometry_stats.json').write_text(json.dumps(geo_stats,indent=2)+'\n')
    np.savez_compressed(out/'per_scene.npz',**per_scene)
    print('COMPLETE_RGB_EVALUATION',len(results),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--out',default='results/round2_rgb_geometry')
    parser.add_argument('--matching',default='results/round2_rendered')
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--train-only',action='store_true')
    parser.add_argument('--eval-only',action='store_true')
    main(parser.parse_args())
