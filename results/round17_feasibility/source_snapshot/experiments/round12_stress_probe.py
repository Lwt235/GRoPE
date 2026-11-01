"""Locked old-development scale and controlled-geometry sensitivity checks.

Every scene's full forward archive must complete before any evaluation stage
decodes sensor supervision. Only direct probabilities are stored; common-grid
predictions are the fixed linear reconstruction of those same arrays.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments import round11_stratum_probe as R11
from experiments import round10_partition_data as D
from experiments.round6_region_data import read_scene_selection, _project_gt_pixels
from experiments.round5_rgbd_data import load_sensor_depth
from models import round12_stress_pe as M

A=R11.A
B=R11.B
ROOT=A.ROOT
STATES=M.STATES
ROLES=M.ROLES
SCENES=('harvard_c8/hv_c8_3','brown_bm_4/brown_bm_4')
BASE_FILES=('lock.json','lock.sha256','selection.json','selection.sha256','completion.json')


def sources():
    result=R11.sources()
    for name in ('ROUND12_STRESS_PROTOCOL.md','models/round12_stress_pe.py',
                 'experiments/round12_stress_probe.py','tests/test_round12_stress_pe.py',
                 'tests/test_round12_stress_probe.py'):
        result[name]=A.sha(ROOT/name)
    return result


def inherited_contract(data,r11,baseline):
    R11.verify_lock(r11,data,baseline)
    selected=R11.verify_selection(r11)
    bselected=B.verify_selection(baseline)
    complete=json.loads((r11/'completion.json').read_text())
    for name in BASE_FILES[:-1]:
        if A.sha(r11/name)!=complete['files'][name]:raise ValueError('changed completed R11 contract')
    cfg={role:dict(selected['configurations'][role]) for role in ('full4','iso4')}
    cfg.update({role:dict(bselected['configurations']['full' if role=='full1' else role])
                for role in ROLES if role not in cfg})
    if bselected['strong_point']!='point' or bselected['strong_quadrature']!='quad16':
        raise ValueError('predeclared point/quad16 controls changed')
    return dict(configurations=cfg,files={str(root/name):A.sha(root/name)
                for root in (r11,baseline) for name in BASE_FILES})


def verify_files(files):
    for path,digest in files.items():
        if A.sha(path)!=digest:raise ValueError('changed frozen file: '+path)


def freeze(out,data,r11,baseline):
    if out.exists():raise ValueError('refusing to overwrite stress output')
    _,entries=A.load_data(data)
    entries=[e for e in entries if e['meta']['split']=='val']
    if {e['meta']['scene'] for e in entries}!=set(SCENES) or len(entries)!=32:
        raise ValueError('unexpected old validation population')
    inherited=inherited_contract(data,r11,baseline)
    metadata=[];inputs={};upstream={}
    for e in entries:
        path,meta=e['path'],e['meta']
        inputs[str(path/'inputs.npz')]=meta['input_sha256']
        upstream[str(path/'manifest.json')]=A.sha(path/'manifest.json')
        upstream[meta['source_selection']]=meta['source_selection_sha256']
        upstream.update(meta['camera_files']);upstream.update(meta['source_index_files'])
        metadata.append(dict(path=str(path),scene=meta['scene'],pair=meta['pair']))
    verify_files(upstream)
    out.mkdir(parents=True)
    parts=M.partitions()
    np.savez_compressed(out/'partitions.npz',**{name+'__'+k:v for name,p in parts.items() for k,v in p.items()})
    lock=dict(created_unix=time.time(),source_hashes=sources(),data_root=str(data.resolve()),
        r11_root=str(r11.resolve()),baseline_root=str(baseline.resolve()),inherited=inherited,
        data_completion_sha256=A.sha(data/'completion.json'),inputs=inputs,upstream_metadata=upstream,
        entries=metadata,partitions_sha256=A.sha(out/'partitions.npz'),
        stages=['freeze','forward','evaluate','finalize'],all_forward_before_any_sensor_decode=True,
        scenarios=['scale',*STATES],roles=list(ROLES),configurations=inherited['configurations'],
        pairs=32,forward_sensor_used=False,labels_before_freeze=False,
        probability_dtype='float64',torch=torch.__version__,numpy=np.__version__,
        calibration=False,fresh_confirmation=False,general_research_goal_achieved=False)
    A.write(out/'lock.json',lock);(out/'lock.sha256').write_text(A.sha(out/'lock.json')+'\n')
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for name in ('ROUND12_STRESS_PROTOCOL.md','models/round12_stress_pe.py','experiments/round12_stress_probe.py'):
        shutil.copy2(ROOT/name,snapshot/Path(name).name)
    print('FROZEN',A.sha(out/'lock.json'),flush=True)


def verify_lock(out):
    lock=json.loads((out/'lock.json').read_text())
    if A.sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip():raise ValueError('changed stress lock')
    if sources()!=lock['source_hashes']:raise ValueError('changed source after freeze')
    verify_files(lock['inputs']);verify_files(lock['upstream_metadata']);verify_files(lock['inherited']['files'])
    if A.sha(out/'partitions.npz')!=lock['partitions_sha256']:raise ValueError('changed partitions')
    if A.sha(Path(lock['data_root'])/'completion.json')!=lock['data_completion_sha256']:
        raise ValueError('changed data completion')
    current=inherited_contract(Path(lock['data_root']),Path(lock['r11_root']),Path(lock['baseline_root']))
    if current!=lock['inherited']:raise ValueError('changed inherited controls')
    return lock


def scene_entries(lock,scene):
    if scene not in SCENES:raise ValueError('only the two old validation scenes are authorized')
    selected=[e for e in lock['entries'] if e['scene']==scene]
    if len(selected)!=16:raise ValueError('incomplete scene')
    return sorted(selected,key=lambda e:Path(e['path']).name)


def read_parts(out):
    archive=A.load_npz(out/'partitions.npz')
    return {name:{key.split('__')[1]:value for key,value in archive.items()
                  if key.startswith(name+'__')} for name in M.PARTITIONS}


def complete_stage(directory,out,stage,scene,started,**metadata):
    files={str(p.relative_to(directory)):A.sha(p) for p in directory.rglob('*') if p.is_file()}
    A.write(directory/'completion.json',dict(stage=stage,scene=scene,started_unix=started,
        finished_unix=time.time(),lock_sha256=A.sha(out/'lock.json'),files=files,**metadata))


def verify_stage(directory,out,stage,scene):
    result=json.loads((directory/'completion.json').read_text())
    if result['stage']!=stage or result['scene']!=scene or result['lock_sha256']!=A.sha(out/'lock.json'):
        raise ValueError('stage identity mismatch')
    actual={str(p.relative_to(directory)) for p in directory.rglob('*')
            if p.is_file() and p.name!='completion.json'}
    if actual!=set(result['files']):raise ValueError('extra or missing stage artifacts')
    verify_files({str(directory/p):digest for p,digest in result['files'].items()})
    return result


def all_forward_complete(out):
    completions={scene:verify_stage(out/'forward'/B.scene_slug(scene),out,'forward',scene) for scene in SCENES}
    return {scene:dict(completion_sha256=A.sha(out/'forward'/B.scene_slug(scene)/'completion.json'),
                       finished_unix=value['finished_unix']) for scene,value in completions.items()}


@torch.no_grad()
def forward(out,scene,device):
    lock=verify_lock(out);entries=scene_entries(lock,scene)
    directory=out/'forward'/B.scene_slug(scene)
    if directory.exists():raise ValueError('refusing to overwrite forward shard')
    if shutil.disk_usage(out).free<8*1024**3:raise ValueError('less than 8 GiB free before forward shard')
    directory.mkdir(parents=True);started=time.time()
    partition=M.partition_tensors(read_parts(out),device)
    numerical=[];geometry=[];configs=lock['configurations'];count=0
    for index,entry in enumerate(entries):
        path=Path(entry['path']);inputs=A.load_npz(path/'inputs.npz')
        dest=directory/path.name;dest.mkdir()
        for state in STATES:
            raw,diag=M.perturbed_observations(inputs,state)
            diag.update(scene=scene,pair=entry['pair']);geometry.append(diag)
            obs=A.observation_tensors(raw,device)
            qnames=M.PARTITIONS if state=='zero' else M.PARTITIONS[:3]
            queries={q:{method:M.prepare_query(obs,partition[q],method)
                       for method in ('full4','full','quad4','quad16')} for q in qnames}
            keys={k:M.prepare_key(obs,partition[k]) for k in qnames}
            groups=[(state,M.ORIGINAL_CONDITIONS)]
            if state=='zero':groups.append(('scale',M.SCALE_CONDITIONS))
            for scenario,conditions in groups:
                saved={}
                for qname,kname in conditions:
                    key=keys[kname];content=queries[qname]['full4']['features']@key['features'].T
                    for role in ROLES:
                        cfg=configs[role];method=cfg['method']
                        prep='full4' if method in ('full4','iso4') else method if method in ('quad4','quad16') else 'full'
                        detail=dict(scene=scene,pair=entry['pair'],scenario=scenario,
                            condition=qname+'_'+kname,role=role,h=cfg['h'])
                        logits=M.geometry_logits(queries[qname][prep],key,method,cfg['h'],diagnostics=detail)
                        probability=(logits+cfg['temperature']*content).softmax(-1)
                        if not bool(torch.isfinite(probability).all()):raise ValueError('nonfinite prediction')
                        error=float((probability.sum(-1)-1).abs().max())
                        if error>1.e-12:raise ValueError('unnormalized prediction')
                        detail['max_normalization_error']=error;numerical.append(detail)
                        name=role+'__'+qname+'_'+kname
                        saved[name]=probability.cpu().numpy()
                        saved[name+'__argmax']=probability.argmax(-1).cpu().numpy().astype(np.int32)
                        count+=1
                np.savez_compressed(dest/(scenario+'.npz'),**saved)
            del queries,keys,obs
        print('FORWARD',scene,index+1,'/',len(entries),'elapsed_seconds',round(time.time()-started,1),flush=True)
    if count!=16*len(ROLES)*(5+len(STATES)*9):raise ValueError('forward record count mismatch')
    A.write(directory/'numerical.json',numerical);A.write(directory/'geometry.json',geometry)
    complete_stage(directory,out,'forward',scene,started,record_count=count,pair_count=len(entries),
        source_input_files={str(Path(e['path'])/'inputs.npz'):lock['inputs'][str(Path(e['path'])/'inputs.npz')] for e in entries},
        sensor_decoded=False,device=device,probability_dtype='float64',common_probabilities_stored=False)
    print('FORWARD_COMPLETE',scene,count,flush=True)


def independent_24_center_labels(meta,parts):
    selection_path=Path(meta['source_selection'])
    if A.sha(selection_path)!=meta['source_selection_sha256']:raise ValueError('changed selection')
    verify_files(meta['camera_files']);verify_files(meta['source_index_files'])
    selection=json.loads(selection_path.read_text())
    matches=[r for r in selection['scenes'] if r['scene']==meta['scene']]
    if len(matches)!=1:raise ValueError('ambiguous sensor scene')
    frames=read_scene_selection(selection_path.parent,matches[0]);qi,ki=meta['pair']
    q,k=frames[qi],frames[ki]
    expected=meta['sensor_sources']
    if {str(q.depth_path),str(k.depth_path)}!=set(expected):raise ValueError('sensor timestamp/pair mismatch')
    verify_files(expected)
    qdepth,kdepth=load_sensor_depth(q.depth_path),load_sensor_depth(k.depth_path)
    centers=parts['g24']['mean']*8
    xy,visible,status=_project_gt_pixels(centers,qdepth,kdepth,q.K_native,k.K_native,q.c2w,k.c2w,(144,192),.05,.05)
    return dict(g24_point_xy=xy,g24_point_visible=visible,g24_point_status=status),dict(
        source_selection=str(selection_path),source_selection_sha256=meta['source_selection_sha256'],
        query_sensor=str(q.depth_path),key_sensor=str(k.depth_path),sensor_sha256=expected,
        query_rgb_index=qi,key_rgb_index=ki,
        query_center_rule='native q24 cell center (integer lower+upper-1)/2; sample original sensor depth then project',
        projected_centroid_used=False,absolute_tolerance=.05,relative_tolerance=.05)


def target_counts(labels,parts,qname,kname):
    visible=labels['gt_visible'];xy=labels['gt_xy']
    index,inside=D.containing_partition(xy,parts[kname]['bounds'])
    if np.any(visible&~inside):raise ValueError('visible target out of native domain')
    qi=parts[qname]['owner'];nq=len(parts[qname]['mass']);nk=len(parts[kname]['mass'])
    direct=np.bincount(qi[visible]*nk+index[visible],minlength=nq*nk).reshape(nq,nk)
    atom,inside8=D.containing_partition(xy,parts['g8']['bounds'])
    if np.any(visible&~inside8):raise ValueError('visible target out of common domain')
    common=np.bincount(parts['g16']['owner'][visible]*432+atom[visible],minlength=108*432).reshape(108,432)
    return direct,common


def common_probability(probability,query,key):
    if np.any(query['parent']<0):raise ValueError('q24 has no common16 parent')
    lifted=probability[:,key['atoms']]*(64/key['mass'][key['atoms']])[None]
    common=np.zeros((108,432),np.float64)
    np.add.at(common,query['parent'],lifted*query['mass'][:,None]/256)
    return common


def kl_js(probability,counts):
    mass=counts.sum(-1);truth=counts/np.maximum(mass,1)[:,None]
    logtruth=np.log(np.maximum(truth,1.e-15));p=np.asarray(probability,np.float64)
    kl=float(np.sum(counts*(logtruth-np.log(np.maximum(p,1.e-15)))))
    midpoint=(truth+p)*.5
    js=.5*(truth*(logtruth-np.log(np.maximum(midpoint,1.e-15)))+
           p*(np.log(np.maximum(p,1.e-15))-np.log(np.maximum(midpoint,1.e-15)))).sum(-1)
    return kl,float(np.sum(js*mass)),int(mass.sum())


def measure(probability,argmax,labels,parts,qname,kname,scenario,direct,common):
    if probability.shape!=direct.shape or probability.dtype!=np.float64:
        raise ValueError('prediction dtype/shape mismatch')
    if not np.isfinite(probability).all() or np.any(probability<0):raise ValueError('invalid probability')
    np.testing.assert_allclose(probability.sum(-1),1.,rtol=0,atol=1.e-12)
    if not np.array_equal(argmax,probability.argmax(-1)):raise ValueError('saved argmax mismatch')
    direct_kl,direct_js,mass=kl_js(probability,direct)
    if scenario=='scale':kl,js=direct_kl,direct_js
    else:
        common_p=common_probability(probability,parts[qname],parts[kname])
        kl,js,common_mass=kl_js(common_p,common)
        if common_mass!=mass:raise ValueError('direct/common population mismatch')
    good=labels[qname+'_point_visible'];point=labels[qname+'_point_xy']
    pred=parts[kname]['mean'][argmax]*8
    correct=(np.linalg.norm(pred-point,axis=-1)<=8)&good
    return dict(kl_sum=kl,js_sum=js,direct_kl_sum=direct_kl,mass=mass,
                pck_count=int(correct.sum()),n_point=int(good.sum()))


def evaluate(out,scene):
    lock=verify_lock(out);entries=scene_entries(lock,scene)
    forward_chain=all_forward_complete(out)  # Both scenes complete before the first sensor decode.
    directory=out/'evaluation'/B.scene_slug(scene)
    if directory.exists():raise ValueError('refusing to overwrite evaluation shard')
    directory.mkdir(parents=True);(directory/'labels').mkdir();started=time.time()
    if any(value['finished_unix']>started for value in forward_chain.values()):raise ValueError('forward chronology')
    parts=read_parts(out);rows=[];provenance=[]
    for i,entry in enumerate(entries):
        path=Path(entry['path']);meta=json.loads((path/'manifest.json').read_text())
        if A.sha(path/'labels.npz')!=meta['label_sha256']:raise ValueError('changed old sensor labels')
        labels=A.load_npz(path/'labels.npz');extra,source=independent_24_center_labels(meta,parts)
        labels.update(extra)
        np.savez_compressed(directory/'labels'/(path.name+'.npz'),**labels)
        source.update(pair=meta['pair'],scene=scene,old_labels_sha256=meta['label_sha256']);provenance.append(source)
        for scenario in ('scale',*STATES):
            predictions=A.load_npz(out/'forward'/B.scene_slug(scene)/path.name/(scenario+'.npz'))
            conditions=M.SCALE_CONDITIONS if scenario=='scale' else M.ORIGINAL_CONDITIONS
            expected={role+'__'+q+'_'+k+suffix for role in ROLES for q,k in conditions for suffix in ('','__argmax')}
            if set(predictions)!=expected:raise ValueError('unexpected prediction identities')
            for qname,kname in conditions:
                direct,common=target_counts(labels,parts,qname,kname)
                for role in ROLES:
                    name=role+'__'+qname+'_'+kname
                    row=measure(predictions[name],predictions[name+'__argmax'],labels,parts,qname,kname,
                                scenario,direct,common)
                    row.update(scene=scene,pair=meta['pair'],condition=qname+'_'+kname,scenario=scenario,
                        role=role,config=lock['configurations'][role],metric='direct' if scenario=='scale' else 'common16_to_8')
                    rows.append(row)
        print('EVALUATED',scene,i+1,'/',len(entries),flush=True)
    verify_population(rows,entries)
    A.write(directory/'rows.json',rows);A.write(directory/'label_provenance.json',provenance)
    complete_stage(directory,out,'evaluate',scene,started,forward_chain=forward_chain,
                   pair_count=len(entries),record_count=len(rows),sensor_only_for_supervision=True)
    print('EVALUATION_COMPLETE',scene,len(rows),flush=True)


def verify_population(rows,entries):
    expected={(e['scene'],tuple(e['pair']),scenario,q+'_'+k) for e in entries
        for scenario in ('scale',*STATES)
        for q,k in (M.SCALE_CONDITIONS if scenario=='scale' else M.ORIGINAL_CONDITIONS)}
    population=None
    for role in ROLES:
        selected=[r for r in rows if r['role']==role]
        mapping={(r['scene'],tuple(r['pair']),r['scenario'],r['condition']):(r['mass'],r['n_point']) for r in selected}
        if len(mapping)!=len(selected) or set(mapping)!=expected:raise ValueError('missing or duplicate role population')
        if population is None:population=mapping
        elif population!=mapping:raise ValueError('different supervision populations across roles')
    for (scene,pair,scenario,condition),value in population.items():
        if scenario!='scale' and value!=population[(scene,pair,'zero',condition)]:
            raise ValueError('perturbation changed supervision population')
        if value[0]!=population[(scene,pair,'zero','g8_g8')][0]:
            raise ValueError('native partition changed visible pixel population')


def summarize(rows,scenario):
    conditions=[q+'_'+k for q,k in (M.SCALE_CONDITIONS if scenario=='scale' else M.ORIGINAL_CONDITIONS)]
    point_condition='g24_g8' if scenario=='scale' else 'g8_g8'
    output={}
    for role in ROLES:
        selected=[r for r in rows if r['role']==role and r['scenario']==scenario]
        def metric(subset,field='kl_sum',denominator='mass'):
            n=sum(r[denominator] for r in subset)
            if n<=0:raise ValueError('empty metric population')
            return sum(r[field] for r in subset)/n
        by_scene_condition={scene:{condition:metric([r for r in selected if r['scene']==scene and r['condition']==condition])
                              for condition in conditions} for scene in SCENES}
        by_scene={s:float(np.mean(list(v.values()))) for s,v in by_scene_condition.items()}
        by_condition={c:float(np.mean([v[c] for v in by_scene_condition.values()])) for c in conditions}
        pck={s:metric([r for r in selected if r['scene']==s and r['condition']==point_condition],'pck_count','n_point') for s in SCENES}
        output[role]=dict(kl=float(np.mean(list(by_scene.values()))),by_scene=by_scene,by_condition=by_condition,
            by_scene_condition=by_scene_condition,pck8=float(np.mean(list(pck.values()))),pck8_by_scene=pck,
            pck_condition=point_condition,metric='direct' if scenario=='scale' else 'common16_to_8')
    return output


def gates(summaries):
    scale=summaries['scale'];full=scale['full4'];point=scale['point'];quad=scale['quad16']
    gains={c:1-full['by_condition'][c]/quad['by_condition'][c] for c in full['by_condition']}
    scale_checks=dict(point_relative_gain_at_least_2pct=1-full['kl']/point['kl']>=.02,
        both_scenes_improve_over_point=all(full['by_scene'][s]<point['by_scene'][s] for s in SCENES),
        no_condition_worse_than_quad16_by_2pct=min(gains.values())>=-.02,
        q24_to_k8_pck_loss_no_more_than_half_pp=full['pck8']>=scale['point_pck']['pck8']-.005)
    scale_gate=dict(passed=all(scale_checks.values()),checks=scale_checks,
        point_relative_gain=1-full['kl']/point['kl'],quad16_relative_gain=1-full['kl']/quad['kl'],
        condition_gains_vs_quad16=gains,pck_difference_pp=100*(full['pck8']-scale['point_pck']['pck8']))
    stress={}
    for state in STATES:
        values=summaries[state];f=values['full4'];q=values['quad16'];p=values['point_pck']
        gain=1-f['kl']/q['kl'];difference=f['pck8']-p['pck8']
        checks=dict(kl_no_more_than_2pct_worse_than_quad16=gain>=-.02,
                    pck_loss_no_more_than_half_pp=difference>=-.005,all_predictions_finite=True)
        stress[state]=dict(passed=all(checks.values()),checks=checks,
            quad16_relative_gain=gain,pck_difference_pp=100*difference,
            full4_absolute_kl=f['kl'],full4_absolute_pck=f['pck8'],
            full4_kl_change_from_zero=f['kl']-summaries['zero']['full4']['kl'],
            full4_pck_change_from_zero_pp=100*(f['pck8']-summaries['zero']['full4']['pck8']))
    return dict(passed=scale_gate['passed'] and all(stress[s]['passed'] for s in STATES),
                scale=scale_gate,stress=stress,general_research_goal_achieved=False,fresh_confirmation_authorized=False)


def finalize(out):
    lock=verify_lock(out)
    if (out/'completion.json').exists():raise ValueError('refusing to overwrite final result')
    forward_chain=all_forward_complete(out);rows=[];shards={}
    for scene in SCENES:
        directory=out/'evaluation'/B.scene_slug(scene)
        complete=verify_stage(directory,out,'evaluate',scene)
        if complete['forward_chain']!=forward_chain:raise ValueError('changed forward/evaluation chronology')
        rows.extend(json.loads((directory/'rows.json').read_text()))
        shards[scene]=A.sha(directory/'completion.json')
    verify_population(rows,lock['entries'])
    summaries={scenario:summarize(rows,scenario) for scenario in ('scale',*STATES)}
    result=dict(summary=summaries,gates=gates(summaries),source_hashes=sources(),
        lock_sha256=A.sha(out/'lock.json'),configurations=lock['configurations'],
        pair_count=32,record_count=len(rows),forward_chain=forward_chain,evaluation_shards=shards,
        limitations=['old development only','controlled geometry biases, not a natural sensor noise model',
                    'key predicted depth scaled consistently but unused by this positional operator',
                    'no optimized Gaussian R/S or NVS validation','no depth-task result supplied by this branch'])
    A.write(out/'rows.json',rows);A.write(out/'summary.json',result)
    files={str(p.relative_to(out)):A.sha(p) for p in out.rglob('*') if p.is_file()}
    A.write(out/'completion.json',dict(finished_unix=time.time(),files=files,lock_sha256=A.sha(out/'lock.json'),
        fresh_test_read=False,general_research_goal_achieved=False,record_count=len(rows)))
    print(json.dumps(result['gates'],indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('freeze','forward','evaluate','finalize'))
    parser.add_argument('--out',type=Path,default=ROOT/'results/round12_stress')
    parser.add_argument('--data',type=Path,default=ROOT/'results/round10_partition_data')
    parser.add_argument('--r11',type=Path,default=ROOT/'results/round11_stratum')
    parser.add_argument('--baseline',type=Path,default=ROOT/'results/round10_partition_b')
    parser.add_argument('--scene',choices=SCENES)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.stage=='freeze':freeze(args.out,args.data,args.r11,args.baseline)
    elif args.stage=='forward':forward(args.out,args.scene,args.device)
    elif args.stage=='evaluate':evaluate(args.out,args.scene)
    else:finalize(args.out)


if __name__=='__main__':main()
