"""Independent SciPy metric and artifact audit of the forward-only diagnostic."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import entropy


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def audit(runs,out):
    if out.exists():raise ValueError('refuse overwrite')
    protocol=read(runs/'protocol.json');completion=read(runs/'completion.json')
    if completion['protocol_sha256']!=sha(runs/'protocol.json') or (runs/'protocol.sha256').read_text().strip()!=sha(runs/'protocol.json'):
        raise ValueError('protocol hash mismatch')
    for path,digest in completion['source_hashes'].items():
        if sha(path)!=digest or sha(runs/'source_snapshots'/Path(path).name)!=digest:
            raise ValueError('source/snapshot mismatch')
    for path,digest in completion['files'].items():
        if sha(runs/path)!=digest:raise ValueError('changed metric/timing file')
    for item in protocol['cohort']:
        if sha(item['inputs'])!=item['input_sha256'] or sha(item['manifest'])!=item['manifest_sha256']:
            raise ValueError('changed calibration source')
    rows=read(runs/'rows.json');summary=read(runs/'summary.json')
    def identity(row):return row['scene'],row['partition'],row['h'],row['query_index'],row['method']
    table={identity(row):row for row in rows}
    if len(table)!=len(rows) or len(rows)!=3*3*4*16*5:raise ValueError('row identity/count mismatch')
    differences=[];metrics=[]
    for item in completion['distribution_files']:
        path=Path(item['path'])
        if sha(path)!=item['sha256']:raise ValueError('changed saved distribution')
        scene=next(i['scene'] for i in protocol['cohort'] if i['scene'].replace('/','__')==path.parent.name)
        partition,width=path.stem.split('_h');width=float(width)
        with np.load(path,allow_pickle=False) as archive:
            p=archive['dense_probability'].astype(np.float64);indices=archive['query_indices']
            if list(indices)!=protocol['query_indices'][partition]:raise ValueError('changed query selection')
            for method in protocol['methods']:
                q=archive[method+'_probability'].astype(np.float64);mid=.5*(p+q)
                values=dict(kl=entropy(p,q,axis=-1),js=.5*(entropy(p,mid,axis=-1)+entropy(q,mid,axis=-1)),
                            l1=np.abs(p-q).sum(-1))
                if max(np.abs(p.sum(-1)-1).max(),np.abs(q.sum(-1)-1).max())>1e-10:
                    raise ValueError('non-normalized saved distribution')
                for i,index in enumerate(indices):
                    row=table[(scene,partition,width,int(index),method)]
                    copied=dict(row)
                    for metric,array in values.items():
                        delta=abs(row[metric]-array[i]);differences.append(delta)
                        if not np.isfinite(delta) or delta>1e-10:raise ValueError('independent SciPy metric mismatch')
                        copied[metric]=float(array[i])
                    if row['reference_raw_fov']!=float(archive['dense_raw_fov'][i]) or row['method_raw_fov']!=float(archive[method+'_raw_fov'][i]):
                        raise ValueError('raw FOV mismatch')
                    metrics.append(copied)
    if len(metrics)!=len(rows):raise ValueError('archive coverage mismatch')
    independent={}
    by_scene={};fov_sensitivity={}
    for method in protocol['methods']:
        subset=[row for row in metrics if row['method']==method]
        independent[method]={metric:float(np.mean([row[metric] for row in subset])) for metric in ('kl','js','l1')}
        for metric in independent[method]:
            if abs(independent[method][metric]-summary['methods'][method][metric])>1e-12:
                raise ValueError('macro mismatch')
        by_scene[method]={item['scene']:{metric:float(np.mean([row[metric] for row in subset if row['scene']==item['scene']]))
                        for metric in ('kl','js','l1')} for item in protocol['cohort']}
        fov_sensitivity[method]={}
        for label,tiny in [('reference_fov_at_least_1e-20',False),('reference_fov_below_1e-20',True)]:
            group=[row for row in subset if (row['reference_raw_fov']<protocol['tiny_fov_threshold'])==tiny]
            fov_sensitivity[method][label]=dict(rows=len(group),
                **{metric:float(np.mean([row[metric] for row in group])) if group else None for metric in ('kl','js','l1')})
    out.mkdir(parents=True)
    result=dict(passed=True,metric_rows=len(metrics),checked_probability_archives=len(completion['distribution_files']),
        max_scipy_metric_error=max(differences),independent_macro=independent,by_scene=by_scene,
        fixed_numerical_threshold_sensitivity=fov_sensitivity,
        sensitivity_scope='Descriptive audit split at the predeclared numerical FOV threshold; does not replace the complete primary population.',
        no_sensor_labels_opened=True,no_validation_inputs_opened=True,reference_is_predicted_geometry_not_truth=True,
        independent_integrator_replay=False,scientific_effectiveness_passed=False,
        source_sha256=sha(__file__),protocol_sha256=sha(runs/'protocol.json'),completion_sha256=sha(runs/'completion.json'))
    (out/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:result[k] for k in ('passed','metric_rows','max_scipy_metric_error')},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('results/round11_compression_diagnostic'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();audit(args.runs,args.out)
