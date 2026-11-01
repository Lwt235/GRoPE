"""Round11 fixed-four-stratum covariance closure, reusing frozen B baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import round10_cell_probe as B
from models.round11_stratum_pe import prepare_gaussian_strata,gaussian_stratum_logits


A=B.A
ROOT=A.ROOT
PARTITIONS=A.PARTITIONS
METHODS=('full4','iso4')
EVAL_ROLES=('full4','iso4','iso4_matched')
WIDTHS=A.WIDTHS
TEMPERATURES=A.TEMPERATURES
TIMING_ROLES=('full4','iso4','quad16','full1')
BASELINE_FILES=('completion.json','lock.json','lock.sha256','selection.json','selection.sha256',
                'summary.json','gate_rows.json','timing.json',
                'selection_repair_lock.json','selection_repair_lock.sha256')


def sources():
    values=B.sources()
    for name in ('ROUND11_PROTOCOL.md','models/round11_stratum_pe.py','experiments/round11_stratum_probe.py',
                 'experiments/round10_cell_selection_repair.py','docs/round10_selection_numerical_fix.md'):
        values[name]=A.sha(ROOT/name)
    return values


def configurations():
    return {f'{m}_h{h}_t{t}':dict(method=m,h=h,temperature=t)
            for m in METHODS for h in WIDTHS for t in TEMPERATURES}


def prepare(observations,partition,name):
    p=partition[name]
    query=prepare_gaussian_strata(observations['q_to_k_xy']/8,observations['q_to_k_pixel_cov']/64,
        p['owner'],p['bounds'],p['parent'])
    query['features']=A.pool_content(observations['q_features'],p['atoms'],len(p['mean']))
    return query


def baseline_contract(baseline,data):
    B.verify_lock(baseline,data)
    selection=B.verify_selection(baseline)
    completion=json.loads((baseline/'completion.json').read_text())
    for name in BASELINE_FILES:
        if name!='completion.json' and A.sha(baseline/name)!=completion['files'][name]:
            raise ValueError('changed frozen B baseline artifact: '+name)
    extension=json.loads((baseline/'selection_repair_lock.json').read_text())
    if A.sha(baseline/'selection_repair_lock.json')!=(baseline/'selection_repair_lock.sha256').read_text().strip():
        raise ValueError('changed B numerical repair lock')
    if extension['base_lock_sha256']!=A.sha(baseline/'lock.json'):
        raise ValueError('B repair/base lock mismatch')
    for name,digest in extension['source_hashes'].items():
        if A.sha(ROOT/name)!=digest:raise ValueError('changed B numerical repair source')
    if selection['strong_quadrature']!='quad16':
        raise ValueError('this predeclared paired-timing protocol requires B strong quadrature quad16')
    summary=json.loads((baseline/'summary.json').read_text())
    if summary['source_hashes']!=B.sources() or summary['selection_sha256']!=A.sha(baseline/'selection.json'):
        raise ValueError('B summary source/selection mismatch')
    return dict(files={name:A.sha(baseline/name) for name in BASELINE_FILES},selection=selection,
                summary=summary,rows=json.loads((baseline/'gate_rows.json').read_text()))


def verify_lock(out,data,baseline):
    lock=json.loads((out/'lock.json').read_text())
    if A.sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip():raise ValueError('changed Round11 lock')
    if sources()!=lock['source_hashes']:raise ValueError('changed Round11 source after freeze')
    if str(data.resolve())!=lock['data_root'] or str(baseline.resolve())!=lock['baseline_root']:
        raise ValueError('changed data/baseline path')
    if A.sha(data/'completion.json')!=lock['data_completion_sha256'] or A.sha(data/'partitions.npz')!=lock['partition_sha256']:
        raise ValueError('changed development data')
    reference=baseline_contract(baseline,data)
    if reference['files']!=lock['baseline_files']:raise ValueError('changed frozen B comparison evidence')
    return lock,reference


def verify_selection(out):
    if A.sha(out/'selection.json')!=(out/'selection.sha256').read_text().strip():raise ValueError('changed Round11 selection')
    selection=json.loads((out/'selection.json').read_text())
    if selection['lock_sha256']!=A.sha(out/'lock.json'):raise ValueError('selection/lock mismatch')
    for name,digest in selection['calibration_files'].items():
        if A.sha(out/name)!=digest:raise ValueError('changed Round11 calibration evidence')
    return selection


def run_stage(entries,parts,partition,configs,device,out,*,predictions=False):
    rows={name:[] for name in configs};numerical=[]
    for index,entry in enumerate(entries):
        path,meta=entry['path'],entry['meta']
        if A.sha(path/'labels.npz')!=meta['label_sha256']:raise ValueError('changed supervision')
        obs=A.observation_tensors(A.load_npz(path/'inputs.npz'),device)
        labels=A.load_npz(path/'labels.npz');saved={}
        for qname in PARTITIONS:
            query=prepare(obs,partition,qname)
            for kname in PARTITIONS:
                kp=partition[kname];key=B.prepare_key(obs,kp)
                target=A.targets(labels,parts,qname,kname,device)
                content=query['features']@key['features'].T
                cache={}
                for name,config in configs.items():
                    method,h,temp=config['method'],config['h'],config['temperature']
                    identity=(method,h)
                    if identity not in cache:
                        diagnostic=dict(scene=meta['scene'],pair=meta['pair'],condition=qname+'_'+kname,method=method,h=h)
                        cache[identity]=gaussian_stratum_logits(query,key,method,h,diagnostics=diagnostic)
                        numerical.append(diagnostic)
                    probability=(cache[identity]+temp*content).softmax(-1)
                    common=A.common_prediction(probability,query,key,kp['atoms'])
                    row=A.measure(probability,common,target,key)
                    row.update(scene=meta['scene'],pair=meta['pair'],condition=qname+'_'+kname,
                        config=name,method=method,h=h,temperature=temp)
                    rows[name].append(row)
                    if predictions:
                        saved[name+'__'+qname+'_'+kname]=common.cpu().numpy()
                        if qname==kname=='g8':saved[name+'__point_argmax']=probability.argmax(-1).cpu().numpy()
        if predictions:
            p=out/'predictions'/B.scene_slug(meta['scene'])/(path.name+'.npz')
            p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**saved)
        if (index+1)%4==0:print('PAIRS',meta['scene'],index+1,'/',len(entries),flush=True)
    B.verify_metric_population(rows,configs,entries)
    return rows,numerical


def benchmark(entries,partition,candidates,reference,device):
    """Re-time candidates and B's full1/quad16 in the same process/input loop."""
    configs={**candidates,'quad16':reference['selection']['configurations']['quad16'],
             'full1':reference['selection']['configurations']['full']}
    first={}
    for entry in entries:first.setdefault(entry['meta']['scene'],entry)
    timing=[]
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize()
    for scene,entry in first.items():
        obs=A.observation_tensors(A.load_npz(entry['path']/'inputs.npz'),device)
        for qname in PARTITIONS:
            for kname in PARTITIONS:
                for role in TIMING_ROLES:
                    config=configs[role];prep=[];op=[]
                    for repetition in range(18):
                        sync();start=time.perf_counter()
                        if role in METHODS:query=prepare(obs,partition,qname)
                        else:query=A.prepare(obs,partition,qname,16 if role=='quad16' else 1)
                        key=B.prepare_key(obs,partition[kname])
                        sync();middle=time.perf_counter()
                        if role in METHODS:logits=gaussian_stratum_logits(query,key,role,config['h'])
                        else:logits=B.geometry_logits(query,key,config['method'],config['h'])
                        logits=logits+config['temperature']*(query['features']@key['features'].T)
                        A.common_prediction(logits.softmax(-1),query,key,partition[kname]['atoms'])
                        sync();end=time.perf_counter()
                        if repetition>=3:prep.append((middle-start)*1000);op.append((end-middle)*1000)
                    timing.append(dict(scene=scene,condition=qname+'_'+kname,method=role,
                        prepare_ms=float(np.median(prep)),operator_ms=float(np.median(op)),
                        total_ms=float(np.median(np.array(prep)+op)),repeats=15,warmups=3,
                        prepared_query_bytes=sum(v.numel()*v.element_size() for v in query.values() if torch.is_tensor(v)),
                        prepared_key_bytes=sum(v.numel()*v.element_size() for v in key.values() if torch.is_tensor(v)),
                        attention_score_bytes=logits.numel()*logits.element_size(),
                        memory_scope='stored support/content and score tensors; not measured allocator peak',
                        baseline_config_frozen=role not in METHODS,device=device))
        print('TIMED',scene,flush=True)
    return timing


def gates(summary,reference,timing):
    full=summary['full4'];base=reference['summary']['summary'];selected=reference['selection']
    pointname=selected['strong_point'];quadname=selected['strong_quadrature']
    point,quad=base[pointname],base[quadname]
    times={role:float(np.mean([r['total_ms'] for r in timing if r['method']==role])) for role in TIMING_ROLES}
    pointgain=1-full['kl']/point['kl'];quadgain=1-full['kl']/quad['kl'];ratio=times['full4']/times[quadname]
    condition={name:1-full['by_condition'][name]/quad['by_condition'][name] for name in full['by_condition']}
    checks=dict(point_relative_gain_at_least_2pct=pointgain>=.02,
        both_scenes_improve_over_point=all(full['by_scene'][scene]<point['by_scene'][scene] for scene in full['by_scene']),
        quadrature_accuracy_or_speed=(quadgain>=.02 and ratio<=1.25) or (quadgain>=-.01 and ratio<=.5),
        no_partition_worse_than_quadrature_by_2pct=min(condition.values())>=-.02,
        pck_loss_no_more_than_half_pp=full['pck8']>=base['point_pck']['pck8']-.005)
    return dict(passed=all(checks.values()),checks=checks,point_relative_gain=pointgain,strong_point=pointname,
        strong_quadrature=quadname,quadrature_relative_gain=quadgain,full_to_quad_time_ratio=ratio,
        partition_gains=condition,incremental_pe_ms=times,
        pck_difference_pp=100*(full['pck8']-base['point_pck']['pck8']),
        full1_relative_gain=1-full['kl']/base['full']['kl'],
        gate_quadrature_envelope=min(base[m]['kl'] for m in ('quad4','quad16')),
        timing_source='Round11 within-process paired remeasurement; no historical B timing used in gate',
        fresh_confirmation_authorized=False,general_research_goal_achieved=False)


def verify_paired_populations(rows,baseline_rows):
    """Join stored records to enforce unchanged sensor supervision populations."""
    def population(values):
        keyed={}
        for row in values:
            identity=(row['scene'],tuple(row['pair']),row['condition'])
            if identity in keyed:raise ValueError('duplicate paired metric identity')
            keyed[identity]=(row['mass'],row['n_point'])
        return keyed
    expected=population(baseline_rows['point'])
    for name,values in {**{'baseline_'+k:v for k,v in baseline_rows.items()},**rows}.items():
        if population(values)!=expected:
            raise ValueError('candidate/B scene, pair, condition or supervision population mismatch: '+name)


def freeze(out,data,baseline,device):
    if out.exists():raise ValueError('refusing overwrite')
    _,entries=A.load_data(data);reference=baseline_contract(baseline,data)
    lock=dict(source_hashes=sources(),data_root=str(data.resolve()),baseline_root=str(baseline.resolve()),
        baseline_files=reference['files'],data_completion_sha256=A.sha(data/'completion.json'),
        partition_sha256=A.sha(data/'partitions.npz'),created_unix=time.time(),torch=torch.__version__,device_requested=device,
        stage='before Round11 candidate calibration and candidate gate evaluation',
        scenes={e['meta']['scene']:e['meta']['split'] for e in entries},primary='full4',diagnostic='iso4',
        fixed_component_count=4,fixed_new_calibration_count=24,baseline_recalibrated=False,
        numerical_policy='same B GL32 cell integral plus 1e-30 native-area background density')
    out.mkdir(parents=True);A.write(out/'lock.json',lock)
    (out/'lock.sha256').write_text(A.sha(out/'lock.json')+'\n')
    print('FROZEN',A.sha(out/'lock.json'),flush=True)


def calibrate(out,data,baseline,scene,device):
    verify_lock(out,data,baseline)
    if (out/'selection.json').exists():raise ValueError('Round11 calibration closed')
    parts,entries=A.load_data(data);subset=B.scene_entries(entries,scene,'train')
    directory=out/'calibration'/B.scene_slug(scene)
    if directory.exists():raise ValueError('refusing overwrite calibration shard')
    directory.mkdir(parents=True);started=time.time()
    with torch.no_grad():rows,numerical=run_stage(subset,parts,A.partition_tensors(parts,device),configurations(),device,directory)
    A.write(directory/'rows.json',rows);A.write(directory/'numerical.json',numerical)
    A.write(directory/'summary.json',{name:A.summarize(value) for name,value in rows.items()})
    verify_lock(out,data,baseline)
    B.complete_shard(directory,out,'calibrate',scene,started,candidate_gate_labels_read=False,device=device)
    print('CALIBRATED',scene,flush=True)


def select(out,data,baseline):
    lock,reference=verify_lock(out,data,baseline)
    if (out/'selection.json').exists() or (out/'selection.sha256').exists():raise ValueError('refusing overwrite selection')
    _,entries=A.load_data(data);configs=configurations();rows={name:[] for name in configs};files={}
    for scene,split in lock['scenes'].items():
        if split!='train':continue
        directory=out/'calibration'/B.scene_slug(scene);B.verify_shard(directory,out,'calibrate',scene)
        values=json.loads((directory/'rows.json').read_text())
        B.verify_metric_population(values,configs,B.scene_entries(entries,scene,'train'))
        for name,value in values.items():rows[name].extend(value)
        for p in directory.rglob('*'):
            if p.is_file():files[str(p.relative_to(out))]=A.sha(p)
    summary={name:A.summarize(value) for name,value in rows.items()}
    selected={method:min((name for name in configs if configs[name]['method']==method),
                         key=lambda name:summary[name]['kl']) for method in METHODS}
    chosen={m:dict(configs[n]) for m,n in selected.items()}
    chosen['iso4_matched']=dict(chosen['full4'],method='iso4')
    selected['iso4_matched']=selected['full4']
    selection=dict(configurations=chosen,selected_names=selected,
        calibration_summaries=summary,calibration_files=files,lock_sha256=A.sha(out/'lock.json'),created_unix=time.time(),
        primary='full4',diagnostic='iso4',baseline_files=reference['files'],
        baseline_selection_sha256=reference['files']['selection.json'],baseline_recalibrated=False,
        candidate_gate_labels_read=False,tie_rule='configuration insertion order: smallest h then temperature')
    verify_lock(out,data,baseline);A.write(out/'selection.json',selection)
    (out/'selection.sha256').write_text(A.sha(out/'selection.json')+'\n')
    print('SELECTED',selection['configurations'],flush=True)


def evaluate(out,data,baseline,scene,device):
    lock,reference=verify_lock(out,data,baseline);selection=verify_selection(out)
    parts,entries=A.load_data(data);subset=B.scene_entries(entries,scene)
    directory=out/'evaluation'/B.scene_slug(scene)
    if directory.exists():raise ValueError('refusing overwrite evaluation shard')
    directory.mkdir(parents=True);started=time.time();partition=A.partition_tensors(parts,device)
    with torch.no_grad():
        if lock['scenes'][scene]=='val':
            rows,numerical=run_stage(subset,parts,partition,selection['configurations'],device,directory,predictions=True)
            A.write(directory/'rows.json',rows);A.write(directory/'numerical.json',numerical)
        timing=benchmark(subset,partition,selection['configurations'],reference,device)
        A.write(directory/'timing.json',timing)
    verify_lock(out,data,baseline);verify_selection(out)
    B.complete_shard(directory,out,'evaluate',scene,started,selection_sha256=A.sha(out/'selection.json'),
        split=lock['scenes'][scene],device=device,labels_read=lock['scenes'][scene]=='val',
        benchmark_population='first metadata pair; all 9 conditions; four paired roles; 3 warmups + 15 repeats')
    print('EVALUATED',scene,flush=True)


def finalize(out,data,baseline):
    lock,reference=verify_lock(out,data,baseline);selection=verify_selection(out)
    if (out/'completion.json').exists() or (out/'summary.json').exists():raise ValueError('refusing overwrite finalization')
    _,entries=A.load_data(data);rows={name:[] for name in EVAL_ROLES};timing=[];numerical=[]
    for scene,split in lock['scenes'].items():
        directory=out/'evaluation'/B.scene_slug(scene)
        B.verify_shard(directory,out,'evaluate',scene,selection)
        timing.extend(json.loads((directory/'timing.json').read_text()))
        if split=='val':
            values=json.loads((directory/'rows.json').read_text())
            B.verify_metric_population(values,selection['configurations'],B.scene_entries(entries,scene,'val'))
            for name,value in values.items():rows[name].extend(value)
            numerical.extend(json.loads((directory/'numerical.json').read_text()))
    identities=[(r['scene'],r['condition'],r['method']) for r in timing]
    expected={(scene,q+'_'+k,method) for scene in lock['scenes'] for q in PARTITIONS for k in PARTITIONS for method in TIMING_ROLES}
    if len(identities)!=len(expected) or set(identities)!=expected:raise ValueError('incomplete paired timing population')
    verify_paired_populations(rows,reference['rows'])
    summary={name:A.summarize(value) for name,value in rows.items()}
    A.write(out/'gate_rows.json',rows);A.write(out/'timing.json',timing);A.write(out/'gate_numerical.json',numerical)
    # Preserve exact frozen B metric evidence; it is not a new baseline result.
    A.write(out/'baseline_gate_rows.json',reference['rows'])
    result=dict(summary=summary,baseline_summary=reference['summary']['summary'],gates=gates(summary,reference,timing),
        selection_sha256=A.sha(out/'selection.json'),source_hashes=sources(),baseline_files=reference['files'],
        seconds=time.time()-lock['created_unix'],candidate='four fixed strata with full local moment covariance',
        calibration_and_gate_scenes_are_reused_development=True,
        historical_B_failure_retained=reference['summary']['gates'],
        numerical_scope='per geometry before floor; no temperature duplication',
        numerical_totals=dict(geometry_records=len(numerical),
            raw_zero_fov_rows=sum(r['raw_zero_fov_rows'] for r in numerical),
            raw_tiny_fov_rows=sum(r['raw_tiny_fov_rows'] for r in numerical),
            geometry_query_rows=sum(r['query_rows'] for r in numerical)))
    verify_lock(out,data,baseline);verify_selection(out);A.write(out/'summary.json',result)
    files={str(p.relative_to(out)):A.sha(p) for p in out.rglob('*') if p.is_file()}
    A.write(out/'completion.json',dict(files=files,seconds=result['seconds'],fresh_test_read=False,
        general_research_goal_achieved=False,lock_sha256=A.sha(out/'lock.json')))
    print('GATES',result['gates'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','calibrate','select','evaluate','finalize'))
    parser.add_argument('--data',type=Path,default=ROOT/'results/round10_partition_data')
    parser.add_argument('--baseline',type=Path,default=ROOT/'results/round10_partition_b')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scene')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.stage in ('calibrate','evaluate') and not args.scene:parser.error('--scene required for stage')
    if args.stage=='freeze':freeze(args.out,args.data,args.baseline,args.device)
    elif args.stage=='calibrate':calibrate(args.out,args.data,args.baseline,args.scene,args.device)
    elif args.stage=='select':select(args.out,args.data,args.baseline)
    elif args.stage=='evaluate':evaluate(args.out,args.data,args.baseline,args.scene,args.device)
    else:finalize(args.out,args.data,args.baseline)


if __name__=='__main__':main()
