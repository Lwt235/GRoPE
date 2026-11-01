"""Round10 B: staged actual-cell integration with unchanged A effect gates.

The frozen A module supplies cohort verification, support preparation, target
counts, common-grid lifting, metrics, selection conventions and benchmark loop.
Only the native key integration changes. Benchmark dispatch is patched locally
at runtime; no frozen A source or output is modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from experiments import round10_partition_probe as A
from models.round10_cell_integral_fast import gaussian_cell_mass_fast


ROOT=A.ROOT
PARTITIONS=A.PARTITIONS
METHODS=A.METHODS
WIDTHS=A.WIDTHS
TEMPERATURES=A.TEMPERATURES
FLOOR_DENSITY=1.e-30
TINY_FOV=1.e-20


def sources():
    result=A.sources()
    for name in ('ROUND10B_PROTOCOL.md','experiments/round10_cell_probe.py',
                 'models/round10_cell_integral.py','models/round10_cell_integral_fast.py'):
        result[name]=A.sha(ROOT/name)
    return result


def configurations():
    return {f'{m}_h{h}_t{t}':dict(method=m,h=h,temperature=t)
            for m in METHODS for h in WIDTHS for t in TEMPERATURES}


def prepare_key(observations, partition):
    result=A.prepare_key(observations,partition)
    result['cell_bounds']=(partition['bounds'].double()-.5)/8
    return result


def geometry_logits(query,key,method,bandwidth,*,diagnostics=None):
    """The same query controls, integrated over the actual native key cells."""
    options=dict(bandwidth=bandwidth,quadrature_order=32,
                 query_chunk_size=256,key_chunk_size=432)
    if method in ('quad4','quad16'):
        node_mass=gaussian_cell_mass_fast(query['nodes'],query['node_cov'],key['cell_bounds'],**options)
        mass=(node_mass*query['node_weights'].double()[...,None]).sum(1)
    else:
        cov=query['cov'].double()
        eye=torch.eye(2,device=cov.device,dtype=cov.dtype)
        if method in ('point','center'):
            cov=torch.zeros_like(cov)
        elif method=='fixed':
            cov=query['mass'].double()[:,None,None]/(64*12)*eye
        elif method=='isotropic':
            cov=cov.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]*eye
        elif method=='rotated':
            rotation=eye.new_tensor([[0.,-1.],[1.,0.]])
            cov=rotation@cov@rotation.T
        elif method!='full':
            raise ValueError('unknown B control: '+str(method))
        mass=gaussian_cell_mass_fast(query['mean'],cov,key['cell_bounds'],**options)
    if diagnostics is not None:
        fov=mass.sum(-1)
        diagnostics.update(query_rows=len(mass),key_count=mass.shape[-1],
            raw_zero_cell_count=int((mass==0).sum()),raw_zero_fov_rows=int((fov==0).sum()),
            raw_tiny_fov_rows=int((fov<TINY_FOV).sum()),
            raw_fov_min=float(fov.min()),raw_fov_max=float(fov.max()),
            floor_fov_mass=float(FLOOR_DENSITY*(key['mass'].double()/64).sum()))
    # This is a constant background density integrated over actual native area.
    # It preserves geometric key subdivision, unlike an equal per-token floor.
    mass=mass+FLOOR_DENSITY*(key['mass'].double()/64)[None]
    return mass.log()


def run_stage(entries,parts,partition,configs,device,out,*,predictions=False):
    """A's metric loop with explicit per-geometry numerical diagnostics."""
    rows={name:[] for name in configs}
    numerical=[]
    for index,entry in enumerate(entries):
        path,meta=entry['path'],entry['meta']
        if A.sha(path/'labels.npz')!=meta['label_sha256']:
            raise ValueError('changed supervision')
        observations=A.observation_tensors(A.load_npz(path/'inputs.npz'),device)
        labels=A.load_npz(path/'labels.npz')
        saved={}
        for qname in PARTITIONS:
            queries={budget:A.prepare(observations,partition,qname,budget) for budget in (1,4,16)}
            for kname in PARTITIONS:
                kp=partition[kname];key=prepare_key(observations,kp)
                target=A.targets(labels,parts,qname,kname,device)
                content=queries[1]['features']@key['features'].T
                geometries={}
                for name,config in configs.items():
                    method,width,temp=config['method'],config['h'],config['temperature']
                    budget=4 if method=='quad4' else 16 if method=='quad16' else 1
                    query=queries[budget]
                    gkey=(method,width)
                    if gkey not in geometries:
                        diag=dict(scene=meta['scene'],pair=meta['pair'],condition=qname+'_'+kname,
                                  method=method,h=width)
                        geometries[gkey]=geometry_logits(query,key,method,width,diagnostics=diag)
                        numerical.append(diag)
                    probability=(geometries[gkey]+temp*content).softmax(-1)
                    common=A.common_prediction(probability,query,key,kp['atoms'])
                    row=A.measure(probability,common,target,key)
                    row.update(scene=meta['scene'],pair=meta['pair'],condition=qname+'_'+kname,
                               config=name,method=method,h=width,temperature=temp)
                    rows[name].append(row)
                    if predictions:
                        saved[name+'__'+qname+'_'+kname]=common.cpu().numpy()
                        if qname==kname=='g8':
                            saved[name+'__point_argmax']=probability.argmax(-1).cpu().numpy()
        if predictions:
            predpath=out/'predictions'/meta['scene'].replace('/','__')/(path.name+'.npz')
            predpath.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(predpath,**saved)
        if (index+1)%4==0:
            print('PAIRS',meta['scene'],index+1,'/',len(entries),flush=True)
    return rows,numerical


@contextmanager
def benchmark_dispatch():
    """Patch module bindings only for A's unchanged benchmark implementation."""
    original=A.prepare_key
    def key_fn(observations,partition):
        result=original(observations,partition)
        result['cell_bounds']=(partition['bounds'].double()-.5)/8
        return result
    with patch.object(A,'prepare_key',key_fn),patch.object(A,'geometry_logits',geometry_logits):
        yield


def scene_slug(scene):
    return scene.replace('/','__')


def scene_entries(entries,scene,split=None):
    subset=[e for e in entries if e['meta']['scene']==scene]
    if len(subset)!=16 or (split is not None and any(e['meta']['split']!=split for e in subset)):
        raise ValueError('unknown scene or wrong stage split: '+str(scene))
    return subset


def verify_metric_population(rows,configs,entries):
    expected={(e['meta']['scene'],tuple(e['meta']['pair']),q+'_'+k)
              for e in entries for q in PARTITIONS for k in PARTITIONS}
    if set(rows)!=set(configs):raise ValueError('incomplete metric roles')
    for name,values in rows.items():
        identities=[(r['scene'],tuple(r['pair']),r['condition']) for r in values]
        if len(identities)!=len(expected) or set(identities)!=expected:
            raise ValueError('missing or duplicate scene/pair/condition metric row')
        config=configs[name]
        if any((r['method'],r['h'],r['temperature'])!=(config['method'],config['h'],config['temperature'])
               or r['config']!=name for r in values):
            raise ValueError('metric configuration mismatch')


def verify_lock(out,data):
    lock=json.loads((out/'lock.json').read_text())
    if sources()!=lock['source_hashes']:
        raise ValueError('source changed after B freeze')
    if str(data.resolve())!=lock['data_root']:
        raise ValueError('changed data location')
    if A.sha(data/'completion.json')!=lock['data_completion_sha256'] or A.sha(data/'partitions.npz')!=lock['partition_sha256']:
        raise ValueError('changed data completion or partitions')
    if A.sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip():
        raise ValueError('changed lock record')
    return lock


def verify_selection(out):
    path=out/'selection.json'
    if A.sha(path)!=(out/'selection.sha256').read_text().strip():
        raise ValueError('changed selection')
    selected=json.loads(path.read_text())
    if selected['lock_sha256']!=A.sha(out/'lock.json'):
        raise ValueError('selection lock mismatch')
    for name,digest in selected['calibration_files'].items():
        if A.sha(out/name)!=digest:raise ValueError('changed calibration evidence')
    return selected


def complete_shard(directory,out,stage,scene,started,**metadata):
    files={str(p.relative_to(directory)):A.sha(p) for p in directory.rglob('*') if p.is_file()}
    A.write(directory/'completion.json',dict(stage=stage,scene=scene,files=files,
        lock_sha256=A.sha(out/'lock.json'),seconds=time.time()-started,**metadata))


def verify_shard(directory,out,stage,scene,selection=None):
    record=json.loads((directory/'completion.json').read_text())
    if record['stage']!=stage or record['scene']!=scene or record['lock_sha256']!=A.sha(out/'lock.json'):
        raise ValueError('shard identity mismatch')
    if selection is not None and record.get('selection_sha256')!=A.sha(out/'selection.json'):
        raise ValueError('shard selection mismatch')
    for name,digest in record['files'].items():
        if A.sha(directory/name)!=digest:raise ValueError('changed stage artifact')
    return record


def freeze(out,data,device):
    if out.exists():raise ValueError('refusing overwrite')
    _,entries=A.load_data(data)
    prior=ROOT/'results/round10_partition_a/lock.json'
    a_lock=json.loads(prior.read_text())
    if A.sources()!=a_lock['source_hashes']:
        raise ValueError('A sources no longer match their frozen lock')
    lock=dict(source_hashes=sources(),data_root=str(data.resolve()),
        data_completion_sha256=A.sha(data/'completion.json'),partition_sha256=A.sha(data/'partitions.npz'),
        frozen_A_lock_sha256=A.sha(prior),stage='before B calibration or gate scores',
        created_unix=time.time(),torch=torch.__version__,device_requested=device,
        scenes={e['meta']['scene']:e['meta']['split'] for e in entries},
        quadrature_order=32,floor_density=FLOOR_DENSITY,tiny_fov_threshold=TINY_FOV,
        numerical_policy='raw mass + 1e-30 * native_key_area / 64 before logarithm; never remove rows')
    out.mkdir(parents=True)
    A.write(out/'lock.json',lock)
    (out/'lock.sha256').write_text(A.sha(out/'lock.json')+'\n')
    print('FROZEN',A.sha(out/'lock.json'),flush=True)


def calibrate(out,data,scene,device):
    verify_lock(out,data)
    if (out/'selection.json').exists():raise ValueError('calibration closed after selection')
    parts,entries=A.load_data(data);subset=scene_entries(entries,scene,'train')
    directory=out/'calibration'/scene_slug(scene)
    if directory.exists():raise ValueError('refusing overwrite calibration shard')
    directory.mkdir(parents=True)
    started=time.time()
    with torch.no_grad():
        rows,numerical=run_stage(subset,parts,A.partition_tensors(parts,device),configurations(),device,directory)
    verify_metric_population(rows,configurations(),subset)
    A.write(directory/'rows.json',rows);A.write(directory/'numerical.json',numerical)
    A.write(directory/'summary.json',{name:A.summarize(value) for name,value in rows.items()})
    verify_lock(out,data)
    complete_shard(directory,out,'calibrate',scene,started,gate_labels_read=False,device=device)
    print('CALIBRATED',scene,flush=True)


def select(out,data):
    lock=verify_lock(out,data)
    if (out/'selection.json').exists() or (out/'selection.sha256').exists():
        raise ValueError('refusing overwrite selection')
    _,entries=A.load_data(data)
    configs=configurations();rows={name:[] for name in configs};files={}
    for scene,split in lock['scenes'].items():
        if split!='train':continue
        directory=out/'calibration'/scene_slug(scene)
        record=verify_shard(directory,out,'calibrate',scene)
        data_rows=json.loads((directory/'rows.json').read_text())
        verify_metric_population(data_rows,configs,scene_entries(entries,scene,'train'))
        for name,values in data_rows.items():rows[name].extend(values)
        for p in directory.rglob('*'):
            if p.is_file():files[str(p.relative_to(out))]=A.sha(p)
    summary={name:A.summarize(values) for name,values in rows.items()}
    selected={method:min((name for name in configs if configs[name]['method']==method),
                        key=lambda name:summary[name]['kl']) for method in METHODS}
    selected['point_pck']=max((name for name in configs if configs[name]['method'] in ('point','center')),
                              key=lambda name:summary[name]['pck8'])
    selected['rotated']=selected['full']
    chosen={method:dict(configs[name]) for method,name in selected.items()}
    chosen['rotated']['method']='rotated'
    strong=min(('quad4','quad16'),key=lambda name:summary[selected[name]]['kl'])
    strong_point=min(('point','center'),key=lambda name:summary[selected[name]]['kl'])
    # Native-cell point and center are identical positive implementation controls.
    for h in WIDTHS:
        for t in TEMPERATURES:
            left=summary[f'point_h{h}_t{t}'];right=summary[f'center_h{h}_t{t}']
            if left!=right:raise ValueError('point/center positive-control identity failed')
    selection=dict(configurations=chosen,selected_names=selected,strong_quadrature=strong,
        strong_point=strong_point,calibration_summaries=summary,calibration_files=files,
        lock_sha256=A.sha(out/'lock.json'),created_unix=time.time(),
        gate_labels_read_by_B=False,point_center_identity=True,
        tie_rule='method/h/T insertion order; point before center; smallest h then T')
    verify_lock(out,data)
    A.write(out/'selection.json',selection)
    (out/'selection.sha256').write_text(A.sha(out/'selection.json')+'\n')
    print('SELECTED',chosen,flush=True)


def evaluate(out,data,scene,device):
    lock=verify_lock(out,data);selection=verify_selection(out)
    parts,entries=A.load_data(data);subset=scene_entries(entries,scene)
    directory=out/'evaluation'/scene_slug(scene)
    if directory.exists():raise ValueError('refusing overwrite evaluation shard')
    directory.mkdir(parents=True);started=time.time()
    partition=A.partition_tensors(parts,device)
    with torch.no_grad():
        if lock['scenes'][scene]=='val':
            rows,numerical=run_stage(subset,parts,partition,selection['configurations'],device,directory,predictions=True)
            verify_metric_population(rows,selection['configurations'],subset)
            A.write(directory/'rows.json',rows);A.write(directory/'numerical.json',numerical)
            A.write(directory/'dense_predicted_diagnostic.json',A.dense_diagnostic(subset,parts))
        with benchmark_dispatch():
            timing=A.benchmark(subset,partition,{m:selection['configurations'][m] for m in METHODS},device)
        A.write(directory/'timing.json',timing)
    verify_lock(out,data);verify_selection(out)
    complete_shard(directory,out,'evaluate',scene,started,selection_sha256=A.sha(out/'selection.json'),
        split=lock['scenes'][scene],device=device,
        labels_read=lock['scenes'][scene]=='val',benchmark_population='first metadata pair per scene; 3 warmups + 15 repeats')
    print('EVALUATED',scene,flush=True)


def finalize(out,data):
    lock=verify_lock(out,data);selection=verify_selection(out)
    if (out/'completion.json').exists() or (out/'summary.json').exists():
        raise ValueError('refusing overwrite finalization')
    rows={name:[] for name in selection['configurations']};timing=[];dense=[];numerical=[]
    for scene,split in lock['scenes'].items():
        directory=out/'evaluation'/scene_slug(scene)
        verify_shard(directory,out,'evaluate',scene,selection)
        timing.extend(json.loads((directory/'timing.json').read_text()))
        if split=='val':
            values=json.loads((directory/'rows.json').read_text())
            if set(values)!=set(rows):raise ValueError('missing gate role')
            for name,value in values.items():rows[name].extend(value)
            dense.extend(json.loads((directory/'dense_predicted_diagnostic.json').read_text()))
            numerical.extend(json.loads((directory/'numerical.json').read_text()))
    if len(timing)!=5*9*7 or any(len(value)!=32*9 for value in rows.values()):
        raise ValueError('incomplete unchanged metric/timing population')
    summary={name:A.summarize(value) for name,value in rows.items()}
    A.write(out/'gate_rows.json',rows);A.write(out/'timing.json',timing)
    A.write(out/'dense_predicted_diagnostic.json',dense);A.write(out/'gate_numerical.json',numerical)
    result=dict(summary=summary,gates=A.gates(summary,selection,timing),
        selection_sha256=A.sha(out/'selection.json'),source_hashes=sources(),
        seconds=time.time()-lock['created_unix'],candidate='Gaussian query integrated over actual native key rectangle',
        calibration_and_gate_scenes_are_reused_development=True,
        numerical_scope='record per geometry before floor; temperature copies do not duplicate row counts',
        numerical_totals=dict(geometry_records=len(numerical),
            raw_zero_fov_rows=sum(r['raw_zero_fov_rows'] for r in numerical),
            raw_tiny_fov_rows=sum(r['raw_tiny_fov_rows'] for r in numerical),
            geometry_query_rows=sum(r['query_rows'] for r in numerical)))
    verify_lock(out,data);verify_selection(out)
    A.write(out/'summary.json',result)
    files={str(p.relative_to(out)):A.sha(p) for p in out.rglob('*') if p.is_file()}
    A.write(out/'completion.json',dict(files=files,seconds=result['seconds'],fresh_test_read=False,
        general_research_goal_achieved=False,lock_sha256=A.sha(out/'lock.json')))
    print('GATES',result['gates'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','calibrate','select','evaluate','finalize'))
    parser.add_argument('--data',type=Path,default=ROOT/'results/round10_partition_data')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scene')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.stage in ('calibrate','evaluate') and not args.scene:
        parser.error('--scene is required for this stage')
    if args.stage=='freeze':freeze(args.out,args.data,args.device)
    elif args.stage=='calibrate':calibrate(args.out,args.data,args.scene,args.device)
    elif args.stage=='select':select(args.out,args.data)
    elif args.stage=='evaluate':evaluate(args.out,args.data,args.scene,args.device)
    else:finalize(args.out,args.data)


if __name__=='__main__':main()
