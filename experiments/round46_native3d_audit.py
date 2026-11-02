"""Independent native-3D geometry/phase reconstruction and full metric replay."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from scipy.special import logsumexp,ndtr
from models.round5_cross_attention import METHODS
from models.round46_native3d_pe import ROLES as NEW_ROLES
from experiments.round46_native3d_train import ROOT,OUT,SEEDS,read,write,sha,checked
from experiments.round43_frequency_audit import array
from experiments.round45_phase_warp_audit import numpy_phase as numpy_2d_phase
from experiments.round46_native3d_data import load_frames
ROLES=NEW_ROLES+('legacy_point2d',)


def numpy_phase(role,state,mean):
    if role=='gaussian2d':return numpy_2d_phase('warp_full',state,mean)
    x=array(mean);w=array(state['frequency']);prefix='position_field.'
    s={k[len(prefix):]:array(v) for k,v in state.items() if k.startswith(prefix)}
    if role=='free3d':w=w+s['frequency_scale']*s['frequency_offset']
    elif role=='affine3d':x=x+x@s['affine_offset'].T
    elif role not in ('point3d','center2d'):
        z=x/12
        if role in ('gaussian3d','iso3d','fixed3d','fixed_centers3d'):
            centers=s['centers']+s.get('mean_offset',0);L=np.zeros((64,3,3))
            for g in range(64):
                diag=np.exp(np.broadcast_to(s['log_precision'][g],(3,))) if 'log_precision' in s else np.ones(3)
                L[g]=np.diag(diag)
                if 'lower' in s:L[g][np.tril_indices(3,-1)]=s['lower'][g]
            P=4*L@L.transpose(0,2,1);delta=z[...,None,:]-centers
            basis=np.exp(-.5*np.einsum('...gi,gij,...gj->...g',delta,P,delta))
        elif role=='fourier3d':
            phase=z@(s['base_frequency']+12*s['frequency_offset']).T+s['phase_offset']
            basis=np.concatenate((np.cos(phase),np.sin(phase)),-1)
        elif role=='mlp3d':
            hidden=z@(s['base_mlp']+s['mlp_offset'])+s['bias'];basis=hidden*ndtr(hidden)
        else:raise ValueError(role)
        x=x+12*(basis/s['initial_rms'])@s['projection']
    return x@w.T


def reference_geometry(q,k):
    # Independent pinhole formulas and world -> camera solves, not pair_geometry.
    u=np.tile(np.arange(24)*8+3.5,18);v=np.repeat(np.arange(18)*8+3.5,24)
    def center(frame):
        K=np.asarray(frame['K'],dtype=float)
        if not np.allclose(K[2],[0,0,1],atol=1e-10) or abs(K[1,0])>1e-10:raise ValueError('pinhole convention')
        depth=np.nan_to_num(frame['predicted_depth'].astype(float),nan=10.,posinf=20.,neginf=.1).clip(.1,20)
        a=v.astype(int);b=u.astype(int)
        d=(depth[a,b]+depth[a+1,b]+depth[a,b+1]+depth[a+1,b+1])/4
        y=(v-K[1,2])/K[1,1];x=(u-K[0,2]-K[0,1]*y)/K[0,0]
        return np.column_stack((d*x,d*y,d))
    qx,kx=center(q),center(k)
    def other(points,source,target):
        world=points@source['c2w'][:3,:3].T+source['c2w'][:3,3]
        return np.linalg.solve(target['c2w'][:3,:3],(world-target['c2w'][:3,3]).T).T
    kin=other(kx,k,q);qin=other(qx,q,k)
    origin=(qx.sum(0)+kin.sum(0))/(len(qx)+len(kin))
    radius=max(float(np.sqrt((np.square(qx-origin).sum()+np.square(kin-origin).sum())/(len(qx)+len(kin)))),1e-3)
    qc=12*(qx-origin)/radius;kc=12*(kin-origin)/radius
    def projection(points,K):
        z=np.maximum(points[:,2],.1)
        px=(K[0,0]*points[:,0]+K[0,1]*points[:,1])/z+K[0,2]
        py=K[1,1]*points[:,1]/z+K[1,2]
        return (np.column_stack((px,py))/8).clip(-128,128)
    native=np.column_stack((u,v))/8
    return dict(qmean=qc,kmean=kc,k_to_q=kc,q_to_k=qc,qmean_2d=native,kmean_2d=native,
        k_to_q_2d=projection(kin,q['K']),q_to_k_2d=projection(qin,k['K']))


def audit_data(out,split):
    lock=checked(out);folder=out/'data'/split
    if sha(folder/'manifest.json')!=(folder/'manifest.sha256').read_text().strip():raise ValueError('geometry manifest changed')
    records=read(folder/'manifest.json')['entries'];errors={};pairs=0
    for record in records:
        if sha(record['path'])!=record['sha256']:raise ValueError('geometry bytes changed')
        original=next(e for e in lock['entries'] if e['scene']==record['scene'] and e['split']==split)
        if record['pairs']!=original['pairs']:raise ValueError('pair population')
        frames=load_frames(lock,record['scene']);oldpath=Path(original['folder'])/'inputs.npz'
        if sha(oldpath)!=original['hashes'][str(oldpath)]:raise ValueError('old visual inputs changed')
        with np.load(record['path'],allow_pickle=False) as z,np.load(oldpath,allow_pickle=False) as old:
            for name in ('q','k'):
                if not np.array_equal(z[name],old[name]):raise ValueError('visual inputs changed')
            for j,(qi,ki) in enumerate(record['pairs']):
                expected=reference_geometry(frames[qi],frames[ki])
                for name,value in expected.items():
                    actual=z[name][j].astype(float);error=float(abs(value-actual).max())
                    errors[name]=max(errors.get(name,0.),error)
                    if not np.allclose(actual,value,atol=3e-5,rtol=3e-7):raise ValueError('geometry reconstruction '+name)
                pairs+=1
    write(out/f'data_audit_{split}.json',dict(passed=True,pairs=pairs,all432tokens=True,
        max_coordinate_errors=errors,visual_features_identical=True,labels_reused=True,
        scope='All new position arrays compared against separate pinhole/world-camera formulas; same author.'))


def check_sample(role,state,path):
    with np.load(path,allow_pickle=False) as archive:z={k:archive[k] for k in archive.files}
    def encode(content,mean):
        value=array(content)
        return (value[...,0::2]+1j*value[...,1::2])*np.exp(1j*numpy_phase(role,state,mean)[None])
    q=encode(z['q'],z['qmean']);k=encode(z['k'],z['kmean'])
    temperature=np.exp(min(float(array(state['layers.0.log_temperature'])),np.log(60)))
    score=temperature*(q@k.conj().swapaxes(-1,-2)).real
    p=np.exp(score-logsumexp(score,axis=-1,keepdims=True))
    error=float(abs(p-z['probability']).max())
    return dict(p_max_error=error,winner_changes=int(np.sum(p.argmax(-1)!=z['probability'].argmax(-1))),
        warning=bool(error>5e-5),scope='first validation pair, first layer, eight query rows/all keys/two heads')


def training(out):
    lock=checked(out);dest=out/'audit_training'
    for seed in SEEDS:
        if not read(out/f'seed_{seed}_complete.json')['completed']:raise ValueError('all seeds required')
    if dest.exists():raise ValueError('retain audit')
    dest.mkdir();rows=[];projections={};warnings=[]
    for role in NEW_ROLES:
        for seed in SEEDS:
            path=out/'runs'/role/str(seed);record=read(path/'run.json')
            for name,digest in record['hashes'].items():
                if sha(path/name)!=digest:raise ValueError('changed training output')
            if [record['initial_shared_sha256'],record['batch_sha256']]!=lock['baseline_identity'][str(seed)]:raise ValueError('unpaired inherited training')
            projection=record['initial_projection_sha256']
            if projection is not None:
                key=(seed,2 if role=='gaussian2d' else 3)
                if key in projections and projections[key]!=projection:raise ValueError('different zero projection initialization')
                projections[key]=projection
            h=read(path/'history.json')['validation']
            if [x['step'] for x in h]!=list(range(0,1001,100)) or min(h,key=lambda x:x['ce'])['step']!=record['best_step']:raise ValueError('selection')
            state=torch.load(path/'best.pt',weights_only=True,map_location='cpu')['state_dict']
            numeric=check_sample(role,state,path/'score_sample.npz')
            if numeric['warning']:warnings.append(dict(role=role,seed=seed,**numeric))
            rows.append(dict(role=role,seed=seed,path=str(path),reused=False,run_sha256=sha(path/'run.json'),checkpoint_sha256=sha(path/'best.pt'),numeric=numeric))
            print('AUDIT',role,seed,numeric['p_max_error'],flush=True)
    for item in lock['inherited']:
        path=Path(item['path'])
        for name,digest in item['hashes'].items():
            if sha(path/name)!=digest:raise ValueError('changed inherited output')
        rows.append(dict(role=item['role'],seed=item['seed'],path=str(path),reused=True,
            run_sha256=item['hashes']['run.json'],checkpoint_sha256=item['hashes']['best.pt'],numeric=item['numeric']))
        if item['numeric']['warning']:warnings.append(dict(role=item['role'],seed=item['seed'],**item['numeric']))
    write(dest/'completion.json',dict(records=rows,warnings=warnings,paired=True,new_checkpoints=55,reused_checkpoints=5,
        new_old_eval_read=False,independent_PE_formulas=True,independently_authored=False,unix=time.time()))


def metrics(out):
    lock=checked(out);seal=read(out/'audit_training/completion.json');dest=out/'metrics'
    if dest.exists():raise ValueError('retain metrics')
    for seed in SEEDS:
        if not read(out/f'export_{seed}_complete.json')['completed']:raise ValueError('all exports required')
    labels={k:[] for k in ('gt_valid','pixcontinuous','key_weights')};ids=[]
    for entry in lock['entries']:
        if entry['split']!='test':continue
        path=Path(entry['folder'])/'labels.npz'
        if sha(path)!=entry['hashes'][str(path)]:raise ValueError('changed old labels')
        with np.load(path,allow_pickle=False) as z:
            for k in labels:labels[k].append(z[k])
        ids.extend(dict(scene=entry['scene'],pair_id=f'{q}:{k}',q_index=q,k_index=k) for q,k in entry['pairs'])
    labels={k:np.concatenate(v) for k,v in labels.items()};mask=labels['gt_valid'].astype(bool)
    yy,xx=np.mgrid[:18,:24];grid=np.stack((xx.ravel()*8+3.5,yy.ravel()*8+3.5),-1)
    allrows=[];max_errors={};count_changes={};sources={}
    for rec in seal['records']:
        role,seed=rec['role'],rec['seed'];path=Path(rec['path'])
        saved=read(path/'old_eval.json')
        if saved['checkpoint_sha256']!=rec['checkpoint_sha256'] or sha(path/'old_eval_statistics.npz')!=saved['statistics_sha256']:
            raise ValueError('changed evaluation artifacts')
        sources[str(path/'old_eval.json')]=sha(path/'old_eval.json')
        with np.load(path/'old_eval_statistics.npz',allow_pickle=False) as z:
            for variant in ('primary','zero_v'):
                arg=z[variant+'_argmax'];soft=z[variant+'_soft_xy'].astype(np.float64)
                lp=z[variant+'_label_logits'].astype(np.float64)-z[variant+'_log_partition'].astype(np.float64)[...,None]
                ce=-(lp*labels['key_weights']).sum(-1)
                error=np.linalg.norm(grid[arg]-labels['pixcontinuous'],axis=-1)
                se=np.linalg.norm(soft-labels['pixcontinuous'],axis=-1)
                stats=dict(n_valid=mask.sum(-1),ce_sum=np.where(mask,ce,0).sum(-1),
                    pck8_count=((error<=8)&mask).sum(-1),pck4_count=((error<=4)&mask).sum(-1),
                    epe_argmax_sum=np.where(mask,error,0).sum(-1),epe_soft_sum=np.where(mask,se,0).sum(-1))
                old=[r for r in saved['rows'] if r['variant']==variant]
                if [{k:r[k] for k in ids[0]} for r in old]!=ids:raise ValueError('evaluation population mismatch')
                if arg.shape!=mask.shape or not all(np.isfinite(v).all() for v in stats.values()):raise ValueError('bad full statistics')
                for key,values in stats.items():
                    diff=values-np.asarray([r[key] for r in old]);max_errors[key]=max(max_errors.get(key,0.),float(abs(diff).max()))
                    if key in ('n_valid','pck8_count','pck4_count'):count_changes[key]=count_changes.get(key,0)+int(np.count_nonzero(diff))
                allrows.extend(dict(ids[i],role=role,seed=seed,variant=variant,**{k:float(v[i]) for k,v in stats.items()}) for i in range(len(ids)))
    def summarize(role,seed,variant='primary'):
        scene={}
        for name in sorted({r['scene'] for r in ids}):
            rows=[r for r in allrows if (r['role'],r['seed'],r['variant'],r['scene'])==(role,seed,variant,name)]
            if len(rows)!=96:raise ValueError('all96 old pairs required')
            n=sum(r['n_valid'] for r in rows)
            scene[name]={name_:sum(r[key] for r in rows)/n for name_,key in
                (('pck8','pck8_count'),('pck4','pck4_count'),('ce','ce_sum'),('epe','epe_argmax_sum'),('soft_epe','epe_soft_sum'))}
        return dict(by_scene=scene,macro={k:float(np.mean([v[k] for v in scene.values()])) for k in next(iter(scene.values()))})
    table={r:{str(seed):summarize(r,seed) for seed in SEEDS} for r in ROLES}
    means={r:{k:float(np.mean([v['macro'][k] for v in entries.values()])) for k in next(iter(entries.values()))['macro']} for r,entries in table.items()}
    strongest='legacy_point2d';checks=[]
    for control in (strongest,'center2d','point3d','gaussian2d','fourier3d','mlp3d','affine3d','free3d'):
        delta=means['gaussian3d']['pck8']-means[control]['pck8'];ce=means['gaussian3d']['ce']/means[control]['ce']-1
        wins=sum(table['gaussian3d'][str(s)]['macro']['pck8']>table[control][str(s)]['macro']['pck8'] for s in SEEDS)
        scene={name:float(np.mean([table['gaussian3d'][str(s)]['by_scene'][name]['pck8']-table[control][str(s)]['by_scene'][name]['pck8'] for s in SEEDS])) for name in next(iter(table[control].values()))['by_scene']}
        checks.append(dict(control=control,pck8_difference_pp=100*delta,ce_relative_change=ce,winning_seeds=wins,
            scene_differences=scene,passed=bool(delta>=.01 and wins>=4 and sum(v>0 for v in scene.values())>=4 and ce<=.02)))
    components=[]
    for control in ('fixed3d','iso3d','fixed_centers3d'):
        seed_deltas=[100*(table['gaussian3d'][str(s)]['macro']['pck8']-table[control][str(s)]['macro']['pck8']) for s in SEEDS]
        scene_deltas={name:100*float(np.mean([table['gaussian3d'][str(s)]['by_scene'][name]['pck8']-table[control][str(s)]['by_scene'][name]['pck8'] for s in SEEDS])) for name in next(iter(table[control].values()))['by_scene']}
        components.append(dict(control=control,pck8_difference_pp=float(np.mean(seed_deltas)),paired_seed_pp=seed_deltas,
            scene_pp=scene_deltas,winning_seeds=sum(v>0 for v in seed_deltas),winning_scenes=sum(v>0 for v in scene_deltas.values())))
    dest.mkdir();write(dest/'frame_rows.json',allrows)
    write(dest/'summary.json',dict(means=means,by_role_seed=table,checks=checks,components=components,new_runs=55,reused_controls=5,
        zero_v={r:{str(seed):summarize(r,seed,'zero_v') for seed in SEEDS} for r in ROLES},
        promising_development_only=all(c['passed'] for c in checks),fresh_confirmation=False,goal_complete=False))
    write(dest/'verification.json',dict(rows=len(allrows),pairs_per_run=len(ids),valid_queries_per_run=int(mask.sum()),
        max_gpu_numpy_frame_sum_difference=max_errors,count_changed_frames=count_changes,
        sampled_PE_warnings=seal['warnings'],input_hashes=sources,
        scope='all-query metric replay from saved sufficient statistics; sampled PE score audit, not full independent network replay'))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('training','metrics'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    torch.set_num_threads(4);globals()[a.stage](a.out)


if __name__=='__main__':main()
