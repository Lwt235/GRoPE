"""One post-failure old-data intervention; no optimization or checkpoint choice."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp
from models.round41_gs_readout import GSReadout
from models.round40_decoupled_pe import shape_descriptor,squared_distance
from experiments.round41_gs_data import ROOT,OUT,SCENES,SEEDS,checked,load_scene,query,read,write,sha
from experiments.round41_gs_train import schedule
from experiments.round41_gs_audit import numpy_logits,arr
from experiments import round9_gs_readout as R9

TRAIN=OUT/'attempt1';DEST=ROOT/'results/round41_shape_components'
NAMES=('full','zero_shape','scale_only','shape_only')


def frozen():
    checked(TRAIN);lock=read(DEST/'lock.json')
    for name,digest in lock['artifacts'].items():
        if sha(name)!=digest:raise ValueError('changed diagnostic input '+name)
    return lock


def freeze():
    checked(TRAIN)
    if DEST.exists():raise ValueError('retain existing diagnostic')
    paths=[Path(__file__),ROOT/'ROUND41_SHAPE_COMPONENT_DIAGNOSTIC.md',
           TRAIN/'lock.json',TRAIN/'metrics/summary.json',TRAIN/'audit/completion.json']
    for scene in SCENES:
        for seed in SEEDS:paths.append(TRAIN/'runs'/scene/'decoupled'/str(seed)/'best.pt')
    DEST.mkdir();write(DEST/'lock.json',dict(unix=time.time(),training_dir=str(TRAIN),
        artifacts={str(p):sha(p) for p in paths},old_results_known=True,trained=False))
    (DEST/'source.py').write_text(Path(__file__).read_text())
    (DEST/'protocol.md').write_text((ROOT/'ROUND41_SHAPE_COMPONENT_DIAGNOSTIC.md').read_text())


def logits(model,q,k,encoded):
    base=model.logits(q,k,key_encoded=encoded,zero_shape=True)
    qs,ks=shape_descriptor(q['cov']),shape_descriptor(k['cov'])
    beta=F.softplus(model.strength_raw)/7
    scale=(qs[:,-1,None]-ks[None,:,-1]).square()
    shape=squared_distance(qs[:,:6],ks[:,:6])
    return dict(full=model.logits(q,k,key_encoded=encoded),zero_shape=base,
                scale_only=base-beta*scale,shape_only=base-beta*shape)


@torch.no_grad()
def forward():
    frozen();records=[]
    for scene in SCENES:
        data=load_scene(scene,TRAIN);_,validation=schedule(data)
        for seed in SEEDS:
            dest=DEST/scene/str(seed)
            if dest.exists():raise ValueError('preserve diagnostic forward')
            dest.mkdir(parents=True);start=time.time()
            state=torch.load(TRAIN/'runs'/scene/'decoupled'/str(seed)/'best.pt',map_location='cuda:0',weights_only=True)['state_dict']
            model=GSReadout('decoupled').cuda();model.load_state_dict(state);model.eval()
            context={n:0. for n in NAMES};elements=0
            for target,k,ids in validation:
                key=data['views'][k];q=query(data,k,ids,model,target=target)
                encoded=model.encode(key['feature'],key['anchor'],key['cov'])
                for name,score in logits(model,q,key,encoded).items():
                    err=(score.softmax(-1)@key['rgb']-data['views'][target]['center_rgb'][ids]).square()
                    context[name]+=float(err.double().sum())
                elements+=len(ids)*3
            context={k:v/elements for k,v in context.items()}
            sums=torch.zeros((2,12000,3),dtype=torch.float64,device='cuda');counts=data['all_count']
            stats={n:dict(observations=0,large_trace_mass_sum=0.,distance_squared_sum=0.) for n in NAMES}
            numeric=[]
            for k,key in enumerate(data['views']):
                ids=torch.where(key['visible'])[0];encoded=model.encode(key['feature'],key['anchor'],key['cov'])
                trace=key['cov'].diagonal(dim1=-2,dim2=-1).sum(-1)
                large=trace>=torch.quantile(trace,.9)
                for offset in range(0,len(ids),256):
                    ix=ids[offset:offset+256];q=query(data,k,ix,model)
                    scores=logits(model,q,key,encoded)
                    distance=(q['xy'][:,0,None]-key['xy'][None,:,0]).square()+(q['xy'][:,1,None]-key['xy'][None,:,1]).square()
                    for name,score in scores.items():
                        p=score.softmax(-1);st=stats[name]
                        st['observations']+=len(ix);st['large_trace_mass_sum']+=float(p[:,large].double().sum())
                        st['distance_squared_sum']+=float((p*distance).double().sum())
                        if name in ('scale_only','shape_only'):sums[('scale_only','shape_only').index(name),ix]+=(p@key['rgb']).double()
                    if k==0 and offset==0:
                        small={n:v[:8] for n,v in q.items()};base=numpy_logits(small,key,state,'center_rope')
                        full=numpy_logits(small,key,state,'decoupled')
                        beta=float(np.logaddexp(0,arr(state['strength_raw'])))/7
                        qt=np.maximum(np.trace(arr(small['cov']),axis1=-2,axis2=-1),1e-12)
                        kt=np.maximum(np.trace(arr(key['cov']),axis1=-2,axis2=-1),1e-12)
                        scale=beta*(np.log(qt)[:,None]-np.log(kt)[None])**2
                        for name,ref in (('scale_only',base-scale),('shape_only',full+scale)):
                            refp=np.exp(ref-logsumexp(ref,axis=-1,keepdims=True));saved=scores[name][:8].softmax(-1).cpu().numpy()
                            numeric.append(dict(name=name,max_p_error=float(abs(saved-refp).max()),winner_changes=int(np.sum(saved.argmax(-1)!=refp.argmax(-1)))))
            colors=torch.where((counts>0)[None,:,None],sums/counts.clamp_min(1)[None,:,None],data['state']['color_logits'].sigmoid()[None].double()).float()
            frames=read(R9.DATA/scene/'target_cameras.json')['frames'];predictions=[]
            for c in colors:
                images=[]
                for frame in frames:
                    K,R,t=(torch.tensor(frame[n],dtype=torch.float32,device='cuda') for n in ('K','R','t'))
                    image,_=R9.render(data['state'],K,R,t,colors=c);images.append(image.cpu().numpy())
                predictions.append(np.stack(images))
            np.savez_compressed(dest/'predictions.npz',predictions=np.stack(predictions),names=np.array(['scale_only','shape_only']),colors=colors.cpu().numpy())
            record=dict(scene=scene,seed=seed,path=str(dest/'predictions.npz'),sha256=sha(dest/'predictions.npz'),
                context_mse=context,attention_statistics=stats,numeric=numeric,seconds=time.time()-start)
            write(dest/'run.json',record);records.append(record)
            print('COMPONENT',scene,seed,record['seconds'],flush=True)
    write(DEST/'forward_completion.json',dict(records=records,models=10,trained=False))


def metrics():
    from experiments.round9_analysis import read_target,frame_metrics
    frozen();records=read(DEST/'forward_completion.json')['records']
    if len(records)!=10:raise ValueError('full fixed cohort required')
    if (DEST/'summary.json').exists():raise ValueError('retain diagnostic metrics')
    rows=[]
    for rec in records:
        if sha(rec['path'])!=rec['sha256']:raise ValueError('changed diagnostic predictions')
        frames=read(R9.DATA/rec['scene']/'target_cameras.json')['frames']
        with np.load(rec['path'],allow_pickle=False) as z:preds=z['predictions'];names=z['names'].tolist()
        for i,frame in enumerate(frames):
            truth,_=read_target(frame)
            for name,pred in zip(names,preds[:,i]):rows.append(dict(scene=rec['scene'],seed=rec['seed'],variant=name,split=frame['split'],id=frame['id'],**frame_metrics(pred,truth)))
    summary={name:dict(gate_mse=float(np.mean([r['mse'] for r in rows if r['variant']==name and r['split']=='gate'])),
        by_scene={s:float(np.mean([r['mse'] for r in rows if r['variant']==name and r['split']=='gate' and r['scene']==s])) for s in SCENES}) for name in ('scale_only','shape_only')}
    write(DEST/'frame_metrics.json',rows);write(DEST/'summary.json',dict(variants=summary,records=records,
        original_primary_unchanged=True,new_candidate_validated=False,fresh_confirmation=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','forward','metrics'));a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    globals()[a.stage]()


if __name__=='__main__':main()
