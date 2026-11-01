"""Independent Round10 numeric and saved-prediction/selection verification.

The reference geometry, partition reduction, metrics and selection below do not
import the producer or experiment runner. The numeric stage imports only the
actual model being checked, and opens calibration observations without labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ('g8','g16','mixed')
METHODS = ('point','center','fixed','isotropic','full','quad4','quad16')
WIDTHS = (.25,.5,1.,2.)
TEMPERATURES = (0.,5.,10.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def archive(path):
    with np.load(path,allow_pickle=False) as data:
        return {name:data[name] for name in data.files}


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def entries(data,split=None):
    complete=read(data/'completion.json')
    output=[]
    for item in complete['manifests']:
        path=Path(item['path'])
        if sha(path)!=item['sha256']:raise ValueError('changed manifest')
        meta=read(path)
        if split is not None and meta['split']!=split:continue
        if meta['split'] not in ('train','val') or meta['forward_uses_sensor']:
            raise ValueError('nondevelopment or sensor forward input')
        if sha(path.parent/'inputs.npz')!=meta['input_sha256']:
            raise ValueError('changed forward input')
        output.append((path.parent,meta))
    return output


def reference_features(features,bounds):
    image=np.asarray(features,np.float64).reshape(18,24,-1)
    pooled=[]
    for x0,y0,x1,y1 in bounds:
        row=image[y0//8:y1//8,x0//8:x1//8].mean((0,1))
        pooled.append(row/max(np.linalg.norm(row),1e-12))
    return np.asarray(pooled)


def reference_support(inputs,parts,name,samples=1):
    xy=np.asarray(inputs['q_to_k_xy'],np.float64).reshape(144,192,2)/8
    within=np.asarray(inputs['q_to_k_pixel_cov'],np.float64).reshape(144,192,2,2)/64
    bounds=parts[name+'_bounds']
    means,covariances,nodes,node_cov,node_weights=[],[],[],[],[]
    for x0,y0,x1,y1 in bounds:
        points=xy[y0:y1,x0:x1].reshape(-1,2)
        pixel=within[y0:y1,x0:x1].reshape(-1,2,2)
        m=points.mean(0);centered=points-m
        means.append(m);covariances.append(centered.T@centered/len(points)+pixel.mean(0))
        if samples>1:
            side=int(np.sqrt(samples)); sx=(x1-x0)//side;sy=(y1-y0)//side
            nm,nc,nw=[],[],[]
            for by in range(side):
                for bx in range(side):
                    ys=slice(y0+by*sy,y0+(by+1)*sy);xs=slice(x0+bx*sx,x0+(bx+1)*sx)
                    nm.append(xy[ys,xs].mean((0,1)));nc.append(within[ys,xs].mean((0,1)))
                    nw.append(sx*sy/len(points))
            nodes.append(nm);node_cov.append(nc);node_weights.append(nw)
    output=dict(mean=np.asarray(means),cov=np.asarray(covariances),
        mass=np.asarray(parts[name+'_area'],np.float64),parent=parts[name+'_parent_g16'],
        features=reference_features(inputs['q_features'],bounds))
    if samples>1:
        output.update(nodes=np.asarray(nodes),node_cov=np.asarray(node_cov),node_weights=np.asarray(node_weights))
    return output


def reference_key(inputs,parts,name):
    return dict(mean=np.asarray(parts[name+'_centers'],np.float64)/8,
        cov=np.asarray(parts[name+'_covariance'],np.float64)/64,
        mass=np.asarray(parts[name+'_area'],np.float64),
        features=reference_features(inputs['k_features'],parts[name+'_bounds']))


def log_density(delta,covariance):
    factor=np.linalg.cholesky(covariance)
    whitened=np.linalg.solve(factor,delta[...,None])[...,0]
    logdet=2*np.log(np.diagonal(factor,axis1=-2,axis2=-1)).sum(-1)
    return -.5*(np.square(whitened).sum(-1)+logdet)-np.log(2*np.pi)


def reference_logits(query,key,method,width):
    identity=np.eye(2)
    qc,kc=query['cov'].copy(),key['cov'].copy()
    if method.startswith('quad'):
        score=log_density(query['nodes'][:,:,None]-key['mean'][None,None],
            query['node_cov'][:,:,None]+kc[None,None]+width**2*identity)
        return logsumexp(score+np.log(query['node_weights'])[:,:,None],axis=1)+np.log(key['mass'])[None]
    if method=='point':qc[:]=0;kc[:]=0
    elif method=='center':qc[:]=0
    elif method=='fixed':
        qc=query['mass'][:,None,None]*identity/(64*12)
        kc=key['mass'][:,None,None]*identity/(64*12)
    elif method=='isotropic':
        qc=np.trace(qc,axis1=1,axis2=2)[:,None,None]*identity/2
        kc=np.trace(kc,axis1=1,axis2=2)[:,None,None]*identity/2
    elif method=='rotated':
        rot=np.array([[0.,-1.],[1.,0.]])
        qc=rot@qc@rot.T
    elif method!='full':raise ValueError(method)
    return log_density(query['mean'][:,None]-key['mean'][None],qc[:,None]+kc[None]+width**2*identity)+np.log(key['mass'])[None]


def numeric(data,out,device):
    import torch
    from models.round10_partition_pe import prepare_support,geometry_logits,pool_content
    torch.set_num_threads(4)
    if out.exists():raise ValueError('refuse overwrite')
    out.mkdir(parents=True)
    checked_sources={str(Path(__file__).resolve()):sha(__file__),
        str(ROOT/'models/round10_partition_pe.py'):sha(ROOT/'models/round10_partition_pe.py')}
    started=time.time();parts=archive(data/'partitions.npz')
    first={}
    for path,meta in entries(data,'train'):
        first.setdefault(meta['scene'],(path,meta))
    if len(first)!=3:raise ValueError('expected three calibration scans')
    records=[];moment_errors=[]
    for scene,(path,meta) in first.items():
        inputs=archive(path/'inputs.npz')
        xy=torch.as_tensor(inputs['q_to_k_xy'],dtype=torch.float32,device=device)/8
        within=torch.as_tensor(inputs['q_to_k_pixel_cov'],dtype=torch.float32,device=device)/64
        qfeatures=torch.as_tensor(inputs['q_features'],dtype=torch.float32,device=device)
        kfeatures=torch.as_tensor(inputs['k_features'],dtype=torch.float32,device=device)
        for qname in PARTITIONS:
            owner=torch.as_tensor(parts[qname+'_pixel_index'].ravel(),dtype=torch.long,device=device)
            bounds=torch.as_tensor(parts[qname+'_bounds'],dtype=torch.long,device=device)
            parent=torch.as_tensor(parts[qname+'_parent_g16'],dtype=torch.long,device=device)
            qatom=torch.as_tensor(parts[qname+'_pixel_index'][4::8,4::8].ravel(),dtype=torch.long,device=device)
            qf=pool_content(qfeatures,qatom,len(bounds))
            queries={};references={}
            for method,budget in [('full',1),('quad4',4),('quad16',16)]:
                actual=prepare_support(xy,within,owner,bounds,parent,samples=budget,include_full_moments=budget==1)
                reference=reference_support(inputs,parts,qname,budget)
                for field in ('mean','cov','nodes','node_cov','node_weights'):
                    if field in actual:
                        delta=np.abs(actual[field].cpu().numpy()-reference[field])
                        moment_errors.append(dict(scene=scene,partition=qname,method=method,field=field,
                            max_absolute=float(delta.max()),max_relative_scale=float((delta/(1+np.abs(reference[field]))).max())))
                # Fixed even rows plus geometry-defined extremes; no label or
                # model-score selection. Every native key is retained.
                eigen=np.linalg.eigvalsh(reference['cov'])
                indices=np.unique(np.r_[np.linspace(0,len(bounds)-1,min(32,len(bounds))).astype(int),
                    np.argmax(eigen[:,-1]),np.argmax(eigen[:,-1]/np.maximum(eigen[:,0],1e-12))])
                queries[method]={k:v[indices] for k,v in actual.items()}
                references[method]={k:v[indices] for k,v in reference.items()}
                queries[method]['features']=qf[indices]
            for kname in PARTITIONS:
                katom=torch.as_tensor(parts[kname+'_pixel_index'][4::8,4::8].ravel(),dtype=torch.long,device=device)
                key={k:torch.as_tensor(parts[kname+'_'+field],dtype=torch.float32,device=device)/scale
                     for k,field,scale in [('mean','centers',8),('cov','covariance',64),('mass','area',1)]}
                key['features']=pool_content(kfeatures,katom,len(key['mean']))
                key_ref=reference_key(inputs,parts,kname)
                for method in queries:
                    query,ref=queries[method],references[method]
                    c32=query['features']@key['features'].T
                    c64=ref['features']@key_ref['features'].T
                    for width in WIDTHS:
                        g32=geometry_logits(query,key,method,width)
                        g64=reference_logits(ref,key_ref,method,width)
                        for temperature in (0.,10.):
                            p32=(g32+temperature*c32).softmax(-1).cpu().numpy()
                            p64=softmax(g64+temperature*c64,axis=-1)
                            error=np.abs(p32-p64)
                            records.append(dict(scene=scene,pair=meta['pair'],q=qname,k=kname,
                                method=method,h=width,temperature=temperature,query_rows=len(p32),key_rows=p32.shape[1],
                                max_probability_error=float(error.max()),mean_probability_error=float(error.mean()),
                                topkey_disagreements=int((p32.argmax(-1)!=p64.argmax(-1)).sum())))
        print('NUMERIC',scene,'finished',flush=True)
    maximum=max(r['max_probability_error'] for r in records)
    result=dict(passed=maximum<=1e-4,tolerance=1e-4,max_probability_error=maximum,
        topkey_disagreements=sum(r['topkey_disagreements'] for r in records),
        scope='first calibration pair per scan; all three partitions on each side; full/quad4/quad16; all widths; T 0/10; at most34 fixed/extreme query rows, all keys',
        opened_sensor_labels=False,opened_gate_inputs=False,opened_gate_metrics=False,
        moment_errors=moment_errors,records=records,seconds=time.time()-started,
        source_hashes=checked_sources,
        inputs={str(path/'inputs.npz'):sha(path/'inputs.npz') for path,_ in first.values()})
    if any(sha(path)!=digest for path,digest in checked_sources.items()):
        raise ValueError('numeric audit source changed during execution')
    write(out/'numeric_verification.json',result)
    print('MAXIMUM_PROBABILITY_ERROR',maximum,'PASSED',result['passed'],flush=True)
    return result


def common_reference(probability,query,key,parts,key_name):
    owners=parts[key_name+'_pixel_index'][4::8,4::8].ravel()
    result=np.zeros((108,432),np.float64)
    for row,parent in enumerate(query['parent']):
        density=probability[row,owners]/key['mass'][owners]
        result[parent]+=density*64*query['mass'][row]/256
    return result


def independent_counts(labels,parts):
    xy=np.asarray(labels['gt_xy'],np.float64)
    valid=np.asarray(labels['gt_visible'],bool)
    if np.any(valid & (~np.isfinite(xy).all(-1) | (xy[:,0]<-.5) | (xy[:,0]>=191.5)
                      | (xy[:,1]<-.5) | (xy[:,1]>=143.5))):
        raise ValueError('invalid visible projection')
    indices=np.floor((xy[valid]+.5+1e-9)/8).astype(np.int64)
    target=indices[:,1]*24+indices[:,0]
    parent=parts['g16_pixel_index'].ravel()[valid]
    counts=np.zeros((108,432),np.int64)
    np.add.at(counts,(parent,target),1)
    return counts


def independent_metrics(probability,counts):
    p=np.asarray(probability,np.float64)
    if p.shape!=counts.shape or not np.isfinite(p).all() or np.any(p<0):
        raise ValueError('invalid saved probability')
    if np.max(np.abs(p.sum(-1)-1))>3e-6:raise ValueError('non-normalized saved probability')
    mass=counts.sum(-1)
    active=mass>0
    p=p[active];c=counts[active].astype(np.float64);m=mass[active]
    truth=c/m[:,None]
    log_truth=np.log(np.maximum(truth,1e-15));log_p=np.log(np.maximum(p,1e-15))
    midpoint=.5*(truth+p)
    kl=float(np.sum(c*(log_truth-log_p)))
    js=.5*(np.sum(truth*(log_truth-np.log(np.maximum(midpoint,1e-15))),axis=-1)
            +np.sum(p*(log_p-np.log(np.maximum(midpoint,1e-15))),axis=-1))
    return dict(kl_sum=kl,js_sum=float(js@m),mass=float(m.sum()))


def row_identity(row):
    return (row['scene'],tuple(row['pair']),row['condition'])


def aggregate(rows):
    scenes=sorted({r['scene'] for r in rows})
    conditions=[q+'_'+k for q in PARTITIONS for k in PARTITIONS]
    table=np.zeros((len(scenes),len(conditions),6),np.float64)
    fields=('kl_sum','js_sum','direct_kl_sum','mass','pck_count','n_point')
    for row in rows:
        table[scenes.index(row['scene']),conditions.index(row['condition'])]+=np.array([row[k] for k in fields])
    if np.any(table[:,:,3]<=0) or np.any(table[:,0,5]<=0):raise ValueError('empty aggregate domain')
    kl=table[:,:,0]/table[:,:,3]
    return dict(kl=float(kl.mean()),js=float((table[:,:,1]/table[:,:,3]).mean()),
        direct_kl=float((table[:,:,2]/table[:,:,3]).mean()),
        pck8=float((table[:,0,4]/table[:,0,5]).mean()),
        by_scene=dict(zip(scenes,map(float,kl.mean(1)))),
        by_condition=dict(zip(conditions,map(float,kl.mean(0)))))


def compare_tree(actual,expected,path='',atol=1e-10):
    differences=[]
    if isinstance(expected,dict):
        if set(actual)!=set(expected):raise ValueError('field mismatch '+path)
        for name in expected:differences.extend(compare_tree(actual[name],expected[name],path+'/'+name,atol))
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        delta=abs(float(actual)-float(expected))
        if not np.isfinite(delta) or delta>atol:raise ValueError(f'numeric mismatch {path}: {actual} / {expected}')
        differences.append(delta)
    elif actual!=expected:raise ValueError('value mismatch '+path)
    return differences


def audit_results(data,runs,out,replay=True):
    if out.exists():raise ValueError('refuse overwrite')
    if not (runs/'completion.json').exists():raise ValueError('experiment incomplete')
    out.mkdir(parents=True)
    initial_source=sha(__file__);started=time.time()
    complete=read(runs/'completion.json');lock=read(runs/'lock.json')
    if complete['fresh_test_read']:raise ValueError('unexpected fresh data')
    for relative,digest in complete['files'].items():
        if sha(runs/relative)!=digest:raise ValueError('changed completed experiment artifact: '+relative)
    for relative,digest in lock['source_hashes'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('changed frozen experiment source: '+relative)
    if sha(data/'completion.json')!=lock['data_completion_sha256'] or sha(data/'partitions.npz')!=lock['partition_sha256']:
        raise ValueError('changed data/partition artifact')
    selection=read(runs/'selection.json');summary=read(runs/'summary.json')
    if selection['lock_sha256']!=sha(runs/'lock.json') or selection['calibration_rows_sha256']!=sha(runs/'calibration_rows.json'):
        raise ValueError('broken selection provenance')
    if summary['selection_sha256']!=sha(runs/'selection.json') or (runs/'selection.sha256').read_text().strip()!=sha(runs/'selection.json'):
        raise ValueError('changed selection')
    if summary['source_hashes']!=lock['source_hashes']:raise ValueError('completion source mismatch')
    if selection['created_unix']<lock['created_unix']:raise ValueError('selection predates lock')
    train_rows=read(runs/'calibration_rows.json');gate_rows=read(runs/'gate_rows.json')
    expected_names=[f'{m}_h{h}_t{t}' for m in METHODS for h in WIDTHS for t in TEMPERATURES]
    if list(train_rows)!=expected_names:raise ValueError('configuration grid/order mismatch')
    train_entries=entries(data,'train');val_entries=entries(data,'val')
    if len(train_entries)!=48 or len(val_entries)!=32:raise ValueError('wrong calibration/gate count')
    conditions=[q+'_'+k for q in PARTITIONS for k in PARTITIONS]
    calibration_keys={(meta['scene'],tuple(meta['pair']),condition) for _,meta in train_entries for condition in conditions}
    gate_keys={(meta['scene'],tuple(meta['pair']),condition) for _,meta in val_entries for condition in conditions}
    for table,expected in ((train_rows,calibration_keys),(gate_rows,gate_keys)):
        for name,rows in table.items():
            identities=list(map(row_identity,rows))
            if len(set(identities))!=len(identities) or set(identities)!=expected:
                raise ValueError('incomplete/duplicate row population for '+name)
    recomputed_calibration={name:aggregate(rows) for name,rows in train_rows.items()}
    numeric_differences=compare_tree(selection['calibration_summaries'],recomputed_calibration,atol=1e-10)
    selected={m:min((n for n in expected_names if train_rows[n][0]['method']==m),
        key=lambda n:recomputed_calibration[n]['kl']) for m in METHODS}
    selected['point_pck']=max((n for n in expected_names if train_rows[n][0]['method'] in ('point','center')),
        key=lambda n:recomputed_calibration[n]['pck8'])
    selected['rotated']=selected['full']
    if selected!=selection['selected_names']:raise ValueError('calibration selection mismatch')
    strong_quad=min(('quad4','quad16'),key=lambda name:recomputed_calibration[selected[name]]['kl'])
    strong_point=min(('point','center'),key=lambda name:recomputed_calibration[selected[name]]['kl'])
    if strong_quad!=selection['strong_quadrature'] or strong_point!=selection['strong_point']:
        raise ValueError('strong comparator selection mismatch')
    for name,chosen in selection['configurations'].items():
        expected=train_rows[selected[name]][0]
        if chosen!={k:('rotated' if name=='rotated' and k=='method' else expected[k]) for k in ('method','h','temperature')}:
            raise ValueError('chosen configuration differs from selected rows')
    if set(gate_rows)!=set(selection['configurations']):raise ValueError('wrong gate roles')
    gate_index={role:{row_identity(row):row for row in rows} for role,rows in gate_rows.items()}
    parts=archive(data/'partitions.npz');metric_errors=[];replays=[]
    first={}
    for path,meta in val_entries:first.setdefault(meta['scene'],path)
    probability_files=0;probability_arrays=0
    for path,meta in val_entries:
        if sha(path/'labels.npz')!=meta['label_sha256']:raise ValueError('changed gate labels')
        labels=archive(path/'labels.npz');counts=independent_counts(labels,parts)
        prediction_path=runs/'predictions'/meta['scene'].replace('/','__')/(path.name+'.npz')
        if prediction_path.stat().st_mtime < selection['created_unix']:
            raise ValueError('gate prediction artifact predates selection')
        predictions=archive(prediction_path);probability_files+=1
        expected_arrays={role+'__'+condition for role in gate_rows for condition in conditions}
        expected_arrays|={role+'__point_argmax' for role in gate_rows}
        if set(predictions)!=expected_arrays:raise ValueError('wrong prediction archive fields')
        for role in gate_rows:
            for condition in conditions:
                probability=predictions[role+'__'+condition]
                metric=independent_metrics(probability,counts);probability_arrays+=1
                row=gate_index[role][(meta['scene'],tuple(meta['pair']),condition)]
                for field,value in metric.items():
                    error=abs(value-row[field]);metric_errors.append(error)
                    if error>2e-7:raise ValueError('saved prediction metric mismatch '+field)
            argmax=predictions[role+'__point_argmax']
            centers=parts['g8_centers'][argmax]
            point_valid=labels['g8_point_visible'];point_xy=labels['g8_point_xy']
            correct=int(np.count_nonzero(point_valid & (np.linalg.norm(centers-point_xy,axis=-1)<=8)))
            row=gate_index[role][(meta['scene'],tuple(meta['pair']),'g8_g8')]
            if row['pck_count']!=correct or row['n_point']!=int(point_valid.sum()):raise ValueError('PCK mismatch')
        if replay and path==first[meta['scene']]:
            inputs=archive(path/'inputs.npz')
            for qname in PARTITIONS:
                supports={budget:reference_support(inputs,parts,qname,budget) for budget in (1,4,16)}
                for kname in PARTITIONS:
                    key=reference_key(inputs,parts,kname)
                    for role,config in selection['configurations'].items():
                        budget=int(config['method'][4:]) if config['method'].startswith('quad') else 1
                        query=supports[budget]
                        logits=reference_logits(query,key,config['method'],config['h'])
                        probability=softmax(logits+config['temperature']*(query['features']@key['features'].T),axis=-1)
                        common=common_reference(probability,query,key,parts,kname)
                        observed=predictions[role+'__'+qname+'_'+kname]
                        error=float(np.max(np.abs(common-observed)))
                        if error>1e-4:raise ValueError('float64 selected prediction replay mismatch')
                        replays.append(dict(scene=meta['scene'],pair=meta['pair'],q=qname,k=kname,
                            role=role,max_common_probability_error=error))
        print('AUDITED_GATE',meta['scene'],path.name,flush=True)
    recomputed_gate={name:aggregate(rows) for name,rows in gate_rows.items()}
    numeric_differences.extend(compare_tree(summary['summary'],recomputed_gate,atol=1e-10))
    timing=read(runs/'timing.json')
    all_scenes={meta['scene'] for _,meta in train_entries+val_entries}
    timing_keys={(r['scene'],r['condition'],r['method']) for r in timing}
    if timing_keys!={(s,c,m) for s in all_scenes for c in conditions for m in METHODS} or len(timing)!=len(timing_keys):
        raise ValueError('incomplete timing population')
    if any(r['repeats']!=15 or not np.isfinite(r['total_ms']) or r['total_ms']<=0 for r in timing):
        raise ValueError('invalid timing record')
    times={method:float(np.mean([r['total_ms'] for r in timing if r['method']==method])) for method in METHODS}
    full=recomputed_gate['full'];point=recomputed_gate[strong_point];quad=recomputed_gate[strong_quad]
    point_gain=1-full['kl']/point['kl'];quad_gain=1-full['kl']/quad['kl'];ratio=times['full']/times[strong_quad]
    partition_gains={c:1-full['by_condition'][c]/quad['by_condition'][c] for c in conditions}
    checks=dict(point_relative_gain_at_least_2pct=point_gain>=.02,
        both_scenes_improve_over_point=all(full['by_scene'][s]<point['by_scene'][s] for s in full['by_scene']),
        quadrature_accuracy_or_speed=(quad_gain>=.02 and ratio<=1.25) or (quad_gain>=-.01 and ratio<=.5),
        no_partition_worse_than_quadrature_by_2pct=min(partition_gains.values())>=-.02,
        pck_loss_no_more_than_half_pp=full['pck8']>=recomputed_gate['point_pck']['pck8']-.005)
    expected_gates=dict(passed=all(checks.values()),checks=checks,point_relative_gain=point_gain,
        strong_point=strong_point,strong_quadrature=strong_quad,quadrature_relative_gain=quad_gain,
        full_to_quad_time_ratio=ratio,partition_gains=partition_gains,incremental_pe_ms=times,
        pck_difference_pp=100*(full['pck8']-recomputed_gate['point_pck']['pck8']),
        gate_quadrature_envelope=min(recomputed_gate[m]['kl'] for m in ('quad4','quad16')),
        fresh_confirmation_authorized=False,general_research_goal_achieved=False)
    numeric_differences.extend(compare_tree(summary['gates'],expected_gates,atol=1e-10))
    if sha(__file__)!=initial_source:raise ValueError('audit source changed during execution')
    result=dict(passed=True,scientific_gate_passed=expected_gates['passed'],checks=checks,
        saved_prediction_files=probability_files,saved_common_prediction_arrays=probability_arrays,
        gate_frame_condition_method_rows=sum(map(len,gate_rows.values())),
        calibration_frame_condition_configuration_rows=sum(map(len,train_rows.values())),
        max_saved_metric_error=max(metric_errors),max_aggregation_selection_gate_error=max(numeric_differences),
        independent_float64_replay=replays,max_replay_error=max((r['max_common_probability_error'] for r in replays),default=None),
        replay_scope='first metadata-listed gate pair per old scan; all 9 partitions and selected methods, every query/key',
        timing_scope='independently recomputed aggregates and gates from recorded synchronized medians; clocks not independently remeasured',
        direct_partition_metric_scope='aggregated from complete frozen rows; not independently replayed for every saved row',
        source_sha256=initial_source,run_lock_sha256=sha(runs/'lock.json'),run_completion_sha256=sha(runs/'completion.json'),
        seconds=time.time()-started,opened_fresh_confirmation_data=False)
    write(out/'verification.json',result)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('AUDIT_PASSED',result['passed'],'SCIENTIFIC_GATE_PASSED',result['scientific_gate_passed'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['numeric','results'],required=True)
    parser.add_argument('--data',type=Path,default=ROOT/'results/round10_partition_data')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--runs',type=Path,default=ROOT/'results/round10_partition_a')
    parser.add_argument('--skip-replay',action='store_true')
    args=parser.parse_args()
    if args.stage=='numeric':numeric(args.data,args.out,args.device)
    else:audit_results(args.data,args.runs,args.out,replay=not args.skip_replay)


if __name__=='__main__':main()
