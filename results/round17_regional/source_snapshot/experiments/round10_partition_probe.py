"""Frozen Round10 A calibration, gate, timing, and reviewable predictions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from models.round10_partition_pe import (prepare_support, geometry_logits,
    pool_content, common_prediction)

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ('g8', 'g16', 'mixed')
METHODS = ('point', 'center', 'fixed', 'isotropic', 'full', 'quad4', 'quad16')
WIDTHS = (.25, .5, 1., 2.)
TEMPERATURES = (0., 5., 10.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def sources():
    paths = ['ROUND10_PROTOCOL.md', 'models/round10_partition_pe.py',
             'experiments/round10_partition_probe.py', 'experiments/round10_partition_data.py',
             'experiments/round6_region_data.py', 'experiments/round5_rgbd_data.py']
    return {p:sha(ROOT/p) for p in paths}


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k:archive[k] for k in archive.files}


def load_data(data):
    completion = json.loads((data/'completion.json').read_text())
    contract = json.loads((data/'contract.json').read_text())
    if completion['fresh_test_read'] or completion['pair_count'] != 80:
        raise ValueError('unexpected development cohort')
    if sha(data/'contract.json') != completion['contract_sha256']:
        raise ValueError('changed data contract')
    if sha(data/'partitions.npz') != contract['partitions_sha256']:
        raise ValueError('changed spatial partitions')
    if sha(data/'forward_complete.json') != completion['forward_complete_sha256']:
        raise ValueError('changed forward-before-supervision record')
    if len(completion['manifests']) != 80:
        raise ValueError('incomplete pair cohort')
    for p, digest in completion['source_hashes'].items():
        if sha(p) != digest: raise ValueError('changed data producer')
    entries = []
    for item in completion['manifests']:
        p = Path(item['path'])
        if sha(p) != item['sha256']: raise ValueError('changed pair metadata')
        m = json.loads(p.read_text())
        if m['split'] not in ('train','val') or m['forward_uses_sensor']:
            raise ValueError('unexpected split or forward source')
        if sha(p.parent/'inputs.npz') != m['input_sha256']:
            raise ValueError('changed observations')
        entries.append(dict(path=p.parent, meta=m))
    identities=[(e['meta']['scene'],*e['meta']['pair']) for e in entries]
    if len(set(identities)) != 80:
        raise ValueError('duplicate scene/pair')
    for scene,split in contract['old_scene_splits'].items():
        subset=[e for e in entries if e['meta']['scene']==scene]
        if len(subset)!=16 or any(e['meta']['split']!=split for e in subset):
            raise ValueError('changed old development scene membership')
    if sum(e['meta']['split']=='train' for e in entries)!=48:
        raise ValueError('changed calibration population')
    return load_npz(data/'partitions.npz'), entries


def partition_tensors(parts, device):
    output = {}
    for name in PARTITIONS:
        owner = parts[name+'_pixel_index'].reshape(144,192)
        atoms = owner[4::8,4::8].reshape(-1)
        output[name] = dict(
            bounds=torch.as_tensor(parts[name+'_bounds'], device=device, dtype=torch.long),
            owner=torch.as_tensor(owner.reshape(-1), device=device, dtype=torch.long),
            atoms=torch.as_tensor(atoms, device=device, dtype=torch.long),
            parent=torch.as_tensor(parts[name+'_parent_g16'],device=device,dtype=torch.long),
            mean=torch.as_tensor(parts[name+'_centers']/8, device=device,dtype=torch.float32),
            cov=torch.as_tensor(parts[name+'_covariance']/64, device=device,dtype=torch.float32),
            mass=torch.as_tensor(parts[name+'_area'],device=device,dtype=torch.float32))
    return output


def observation_tensors(inputs, device):
    return {k:torch.as_tensor(inputs[k],device=device,dtype=torch.float32)
            for k in ('q_to_k_xy','q_to_k_pixel_cov','q_features','k_features')}


def prepare(observations, partition, name, samples=1):
    p = partition[name]
    query = prepare_support(observations['q_to_k_xy']/8,observations['q_to_k_pixel_cov']/64,
                            p['owner'],p['bounds'],p['parent'],samples=samples,
                            include_full_moments=samples==1)
    query['features'] = pool_content(observations['q_features'],p['atoms'],len(p['mean']))
    return query


def prepare_key(observations, p):
    return dict(mean=p['mean'],cov=p['cov'],mass=p['mass'],
                features=pool_content(observations['k_features'],p['atoms'],len(p['mean'])))


def containing(xy, stride):
    safe = np.where(np.isfinite(xy),xy,0)
    ix = np.floor((safe+.5+1e-9)/stride).astype(np.int64)
    valid = np.isfinite(xy).all(-1)&(safe[:,0]>=-.5)&(safe[:,0]<191.5)&(safe[:,1]>=-.5)&(safe[:,1]<143.5)
    ix = np.clip(ix,[0,0],[192//stride-1,144//stride-1])
    return ix[:,1]*(192//stride)+ix[:,0],valid


def targets(labels, parts, qname, kname, device):
    xy = labels['gt_xy']
    atom,inside = containing(xy,8)
    visible = labels['gt_visible']
    if np.any(visible & ~inside): raise ValueError('visible target outside image')
    parent = parts['g16_pixel_index'].reshape(-1)
    qi = parts[qname+'_pixel_index'].reshape(-1)
    key_atom_owner = parts[kname+'_pixel_index'][4::8,4::8].reshape(-1)
    ki = key_atom_owner[atom]
    common = np.bincount(parent[visible]*432+atom[visible],minlength=108*432).reshape(108,432)
    nq,nk=len(parts[qname+'_area']),len(parts[kname+'_area'])
    direct = np.bincount(qi[visible]*nk+ki[visible],minlength=nq*nk).reshape(nq,nk)
    return dict(common=torch.as_tensor(common,device=device,dtype=torch.float64),
                direct=torch.as_tensor(direct,device=device,dtype=torch.float64),
                point_xy=torch.as_tensor(labels[qname+'_point_xy'],device=device,dtype=torch.float64),
                point_valid=torch.as_tensor(labels[qname+'_point_visible'],device=device))


def kl_counts(probability, counts):
    mass=counts.sum(-1)
    truth=counts/mass.clamp_min(1)[:,None]
    log_truth=truth.clamp_min(1e-15).log()
    p=probability.double()
    kl=(counts*(log_truth-p.clamp_min(1e-15).log())).sum()
    midpoint=(truth+p)*.5
    js=.5*(truth*(log_truth-midpoint.clamp_min(1e-15).log())
        +p*(p.clamp_min(1e-15).log()-midpoint.clamp_min(1e-15).log())).sum(-1)
    return kl,float((js*mass).sum()),float(mass.sum())


def measure(probability, common, target, key):
    kl,js,mass=kl_counts(common,target['common'])
    direct,_,_=kl_counts(probability,target['direct'])
    center=key['mean'][probability.argmax(-1)].double()*8
    good=target['point_valid']
    correct=((((center-target['point_xy'])**2).sum(-1).sqrt()<=8)&good).sum()
    return dict(kl_sum=float(kl),js_sum=js,mass=mass,direct_kl_sum=float(direct),
                pck_count=int(correct),n_point=int(good.sum()))


def macro(rows, metric='kl_sum', denominator='mass', condition=None):
    scenes=sorted(set(r['scene'] for r in rows))
    values=[]
    for scene in scenes:
        cells=[]
        for q in PARTITIONS:
            for k in PARTITIONS:
                name=q+'_'+k
                if condition is not None and name != condition:continue
                subset=[r for r in rows if r['scene']==scene and r['condition']==name]
                if not subset:raise ValueError('missing condition')
                total=sum(r[denominator] for r in subset)
                if total<=0:raise ValueError('empty metric denominator')
                cells.append(sum(r[metric] for r in subset)/total)
        values.append(float(np.mean(cells)))
    return float(np.mean(values))


def summarize(rows):
    return dict(kl=macro(rows),js=macro(rows,'js_sum'),direct_kl=macro(rows,'direct_kl_sum'),
        pck8=macro(rows,'pck_count','n_point','g8_g8'),
        by_scene={s:macro([r for r in rows if r['scene']==s]) for s in sorted({r['scene'] for r in rows})},
        by_condition={q+'_'+k:macro(rows,condition=q+'_'+k) for q in PARTITIONS for k in PARTITIONS})


def run_stage(entries, parts, partition, configurations, device, out, predictions=False):
    rows={name:[] for name in configurations}
    for index,entry in enumerate(entries):
        path,m=entry['path'],entry['meta']
        if sha(path/'labels.npz')!=m['label_sha256']:raise ValueError('changed supervision')
        observations=observation_tensors(load_npz(path/'inputs.npz'),device)
        labels=load_npz(path/'labels.npz')
        saved={}
        for qname in PARTITIONS:
            queries={budget:prepare(observations,partition,qname,budget) for budget in (1,4,16)}
            for kname in PARTITIONS:
                kp=partition[kname]; key=prepare_key(observations,kp)
                target=targets(labels,parts,qname,kname,device)
                content=queries[1]['features']@key['features'].T
                geometries={}
                for name,config in configurations.items():
                    method,width,temp=config['method'],config['h'],config['temperature']
                    budget=4 if method=='quad4' else 16 if method=='quad16' else 1
                    query=queries[budget]
                    gkey=(method,width)
                    if gkey not in geometries:
                        geometries[gkey]=geometry_logits(query,key,method,width)
                    prob=(geometries[gkey]+temp*content).softmax(-1)
                    common=common_prediction(prob,query,key,kp['atoms'])
                    row=measure(prob,common,target,key)
                    row.update(scene=m['scene'],pair=m['pair'],condition=qname+'_'+kname,
                               config=name,method=method,h=width,temperature=temp)
                    rows[name].append(row)
                    if predictions:
                        saved[name+'__'+qname+'_'+kname]=common.cpu().numpy()
                        if qname==kname=='g8':saved[name+'__point_argmax']=prob.argmax(-1).cpu().numpy()
        if predictions:
            predpath=out/'predictions'/m['scene'].replace('/','__')/(path.name+'.npz')
            predpath.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(predpath,**saved)
        if (index+1)%8==0:print('PAIRS',index+1,'/',len(entries),flush=True)
    return rows


def benchmark(entries, partition, configs, device):
    first={}
    for entry in entries:first.setdefault(entry['meta']['scene'],entry)
    timing=[]
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize()
    for scene,entry in first.items():
        obs=observation_tensors(load_npz(entry['path']/'inputs.npz'),device)
        for q in PARTITIONS:
            for k in PARTITIONS:
                for name,config in configs.items():
                    method=config['method'];budget=4 if method=='quad4' else 16 if method=='quad16' else 1
                    prep,op=[],[]
                    for rep in range(18):
                        sync();t0=time.perf_counter()
                        query=prepare(obs,partition,q,budget);key=prepare_key(obs,partition[k])
                        sync();t1=time.perf_counter()
                        logits=geometry_logits(query,key,method,config['h'])
                        logits=logits+config['temperature']*(query['features']@key['features'].T)
                        common_prediction(logits.softmax(-1),query,key,partition[k]['atoms'])
                        sync();t2=time.perf_counter()
                        if rep>=3:prep.append((t1-t0)*1000);op.append((t2-t1)*1000)
                    timing.append(dict(scene=scene,condition=q+'_'+k,method=name,
                        prepare_ms=float(np.median(prep)),operator_ms=float(np.median(op)),
                        total_ms=float(np.median(np.array(prep)+op)),repeats=15,
                        prepared_query_bytes=sum(v.numel()*v.element_size() for v in query.values() if torch.is_tensor(v)),
                        prepared_key_bytes=sum(v.numel()*v.element_size() for v in key.values() if torch.is_tensor(v)),
                        attention_score_bytes=logits.numel()*logits.element_size(),
                        memory_scope='stored support/content and score tensors; not measured allocator peak'))
        print('TIMED',scene,flush=True)
    return timing


def dense_diagnostic(entries, parts):
    rows=[]
    parent=parts['g16_pixel_index'].reshape(-1)
    for entry in entries:
        inputs=load_npz(entry['path']/'inputs.npz');labels=load_npz(entry['path']/'labels.npz')
        pred,inside=containing(inputs['q_to_k_xy'],8)
        counts=np.bincount(parent[inside]*432+pred[inside],minlength=108*432).reshape(108,432)
        mass=counts.sum(-1);p=counts/np.maximum(mass[:,None],1)
        p[mass==0]=1/432
        truth,valid=containing(labels['gt_xy'],8);visible=labels['gt_visible']
        gt=np.bincount(parent[visible]*432+truth[visible],minlength=108*432).reshape(108,432)
        kl,js,total=kl_counts(torch.from_numpy(p),torch.from_numpy(gt))
        rows.append(dict(scene=entry['meta']['scene'],pair=entry['meta']['pair'],
                         kl_sum=float(kl),mass=total,js_sum=js,empty_predicted_rows=int((mass==0).sum())))
    return rows


def gates(summary, selection, timing):
    full=summary['full'];pointname=selection['strong_point'];point=summary[pointname];quadname=selection['strong_quadrature'];quad=summary[quadname]
    times={m:float(np.mean([r['total_ms'] for r in timing if r['method']==m])) for m in METHODS}
    pointgain=1-full['kl']/point['kl'];quadgain=1-full['kl']/quad['kl']
    ratio=times['full']/times[quadname]
    condition_gains={c:1-full['by_condition'][c]/quad['by_condition'][c] for c in full['by_condition']}
    checks=dict(point_relative_gain_at_least_2pct=pointgain>=.02,
        both_scenes_improve_over_point=all(full['by_scene'][s]<point['by_scene'][s] for s in full['by_scene']),
        quadrature_accuracy_or_speed=(quadgain>=.02 and ratio<=1.25) or (quadgain>=-.01 and ratio<=.5),
        no_partition_worse_than_quadrature_by_2pct=min(condition_gains.values())>=-.02,
        pck_loss_no_more_than_half_pp=full['pck8']>=summary['point_pck']['pck8']-.005)
    return dict(passed=all(checks.values()),checks=checks,point_relative_gain=pointgain,
        strong_point=pointname,strong_quadrature=quadname,quadrature_relative_gain=quadgain,full_to_quad_time_ratio=ratio,
        partition_gains=condition_gains,incremental_pe_ms=times,
        pck_difference_pp=100*(full['pck8']-summary['point_pck']['pck8']),
        gate_quadrature_envelope=min(summary[m]['kl'] for m in ('quad4','quad16')),
        fresh_confirmation_authorized=False,general_research_goal_achieved=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'results/round10_partition_data')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.out.exists():raise ValueError('refusing overwrite')
    parts,entries=load_data(args.data)
    args.out.mkdir(parents=True)
    lock=dict(source_hashes=sources(),data_completion_sha256=sha(args.data/'completion.json'),
              partition_sha256=sha(args.data/'partitions.npz'),stage='before candidate results',
              device=args.device,torch=torch.__version__,created_unix=time.time())
    write(args.out/'lock.json',lock)
    partition=partition_tensors(parts,args.device)
    configs={f'{m}_h{h}_t{t}':dict(method=m,h=h,temperature=t) for m in METHODS for h in WIDTHS for t in TEMPERATURES}
    start=time.time()
    with torch.no_grad():
        train=run_stage([e for e in entries if e['meta']['split']=='train'],parts,partition,configs,args.device,args.out)
        train_summary={n:summarize(rows) for n,rows in train.items()}
        selected={m:min((n for n in configs if configs[n]['method']==m),key=lambda n:train_summary[n]['kl']) for m in METHODS}
        selected['point_pck']=max((n for n in configs if configs[n]['method'] in ('point','center')),key=lambda n:train_summary[n]['pck8'])
        selected['rotated']=selected['full']
        chosen={m:dict(configs[n]) for m,n in selected.items()};chosen['rotated']['method']='rotated'
        strong=min(('quad4','quad16'),key=lambda m:train_summary[selected[m]]['kl'])
        strong_point=min(('point','center'),key=lambda m:train_summary[selected[m]]['kl'])
        selection=dict(configurations=chosen,selected_names=selected,strong_quadrature=strong,strong_point=strong_point,
            calibration_summaries=train_summary,lock_sha256=sha(args.out/'lock.json'),created_unix=time.time())
        write(args.out/'calibration_rows.json',train)
        selection['calibration_rows_sha256']=sha(args.out/'calibration_rows.json')
        write(args.out/'selection.json',selection)
        (args.out/'selection.sha256').write_text(sha(args.out/'selection.json')+'\n')
        print('SELECTED',chosen,flush=True)
        val_entries=[e for e in entries if e['meta']['split']=='val']
        rows=run_stage(val_entries,parts,partition,chosen,args.device,args.out,predictions=True)
        timing=benchmark(entries,partition,{m:chosen[m] for m in METHODS},args.device)
        summary={m:summarize(r) for m,r in rows.items()}
        write(args.out/'gate_rows.json',rows);write(args.out/'timing.json',timing)
        write(args.out/'dense_predicted_diagnostic.json',dense_diagnostic(val_entries,parts))
        result=dict(summary=summary,gates=gates(summary,selection,timing),seconds=time.time()-start,
                    selection_sha256=sha(args.out/'selection.json'),source_hashes=sources())
        if result['source_hashes']!=lock['source_hashes']:raise ValueError('source changed during run')
        write(args.out/'summary.json',result)
        files={str(p.relative_to(args.out)):sha(p) for p in args.out.rglob('*') if p.is_file()}
        write(args.out/'completion.json',dict(files=files,seconds=time.time()-start,fresh_test_read=False))
        print('GATES',result['gates'],flush=True)


if __name__=='__main__':main()
