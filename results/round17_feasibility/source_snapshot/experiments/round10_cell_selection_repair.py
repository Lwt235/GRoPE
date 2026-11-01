"""Audited extension for B's overly strict positive-control equality guard.

The frozen B runner is unchanged. This helper substitutes absolute tolerance
1e-12 ONLY for the recursive equality check of point/center calibration
summaries. Scores, selections, exact min/max ordering and effect gates are
unchanged. An extension lock must exist before select can write anything.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from experiments import round10_cell_probe as B


A=B.A
TOLERANCE=1.e-12
HELPER='experiments/round10_cell_selection_repair.py'
PROTOCOL='docs/round10_selection_numerical_fix.md'


def recursive_difference(left,right,path=''):
    """Return max numeric discrepancy; reject structure/nonfinite changes."""
    if isinstance(left,dict):
        if not isinstance(right,dict) or set(left)!=set(right):
            raise ValueError('positive-control key mismatch at '+path)
        return max((recursive_difference(left[k],right[k],path+'/'+str(k)) for k in left),default=0.)
    if isinstance(left,(list,tuple)):
        if not isinstance(right,(list,tuple)) or len(left)!=len(right):
            raise ValueError('positive-control sequence mismatch at '+path)
        return max((recursive_difference(a,b,path+'/'+str(i)) for i,(a,b) in enumerate(zip(left,right))),default=0.)
    if isinstance(left,(int,float)) and not isinstance(left,bool):
        if not isinstance(right,(int,float)) or isinstance(right,bool) or not math.isfinite(left) or not math.isfinite(right):
            raise ValueError('nonfinite/type-changed positive-control value at '+path)
        return abs(float(left)-float(right))
    if type(left) is not type(right) or left!=right:
        raise ValueError('positive-control nonnumeric mismatch at '+path)
    return 0.


def guard(summary):
    details=[]
    for h in B.WIDTHS:
        for t in B.TEMPERATURES:
            delta=recursive_difference(summary[f'point_h{h}_t{t}'],summary[f'center_h{h}_t{t}'])
            details.append(dict(h=h,temperature=t,max_absolute_difference=delta))
    maximum=max(row['max_absolute_difference'] for row in details)
    if maximum>TOLERANCE:
        raise ValueError('point/center discrepancy exceeds the frozen repair tolerance')
    return dict(tolerance=TOLERANCE,max_absolute_difference=maximum,comparisons=details)


def calibration(out,data):
    """Verify original locks/artifacts, then reproduce the unchanged summaries."""
    lock=B.verify_lock(out,data)
    _,entries=A.load_data(data)
    configs=B.configurations();rows={name:[] for name in configs};files={}
    for scene,split in lock['scenes'].items():
        if split!='train':continue
        directory=out/'calibration'/B.scene_slug(scene)
        B.verify_shard(directory,out,'calibrate',scene)
        data_rows=json.loads((directory/'rows.json').read_text())
        B.verify_metric_population(data_rows,configs,B.scene_entries(entries,scene,'train'))
        for name,value in data_rows.items():rows[name].extend(value)
        for p in directory.rglob('*'):
            if p.is_file():files[str(p.relative_to(out))]=A.sha(p)
    summary={name:A.summarize(value) for name,value in rows.items()}
    return lock,configs,summary,files,guard(summary)


def extension_sources():
    return {name:A.sha(B.ROOT/name) for name in (HELPER,PROTOCOL)}


def assert_no_selection(out):
    if (out/'selection.json').exists() or (out/'selection.sha256').exists():
        raise ValueError('refusing to replace or repair an existing selection')
    if (out/'evaluation').exists():
        raise ValueError('repair must be frozen before any B evaluation shard')


def freeze(out,data):
    assert_no_selection(out)
    path=out/'selection_repair_lock.json'
    if path.exists() or (out/'selection_repair_lock.sha256').exists():
        raise ValueError('refusing overwrite repair lock')
    _,_,_,files,comparison=calibration(out,data)
    record=dict(source_hashes=extension_sources(),base_lock_sha256=A.sha(out/'lock.json'),
        calibration_files=files,created_unix=time.time(),
        stage='before repaired selection and before any B evaluation',
        modification='point/center summary equality guard only: absolute tolerance 1e-12',
        candidate_scores_changed=False,selection_order_changed=False,effect_gates_changed=False,
        guard_diagnostic=comparison)
    A.write(path,record)
    (out/'selection_repair_lock.sha256').write_text(A.sha(path)+'\n')
    print('REPAIR_FROZEN',A.sha(path),flush=True)


def verify_extension(out):
    path=out/'selection_repair_lock.json'
    record=json.loads(path.read_text())
    if A.sha(path)!=(out/'selection_repair_lock.sha256').read_text().strip():
        raise ValueError('changed repair lock')
    if record['source_hashes']!=extension_sources() or record['base_lock_sha256']!=A.sha(out/'lock.json'):
        raise ValueError('changed repair source/protocol or original lock')
    for name,digest in record['calibration_files'].items():
        if A.sha(out/name)!=digest:raise ValueError('changed original calibration artifact')
    return record


def select(out,data):
    assert_no_selection(out)
    extension=verify_extension(out)
    _,configs,summary,files,comparison=calibration(out,data)
    if files!=extension['calibration_files']:
        raise ValueError('calibration artifact population changed after repair freeze')
    selected={method:min((name for name in configs if configs[name]['method']==method),
                        key=lambda name:summary[name]['kl']) for method in B.METHODS}
    selected['point_pck']=max((name for name in configs if configs[name]['method'] in ('point','center')),
                              key=lambda name:summary[name]['pck8'])
    selected['rotated']=selected['full']
    chosen={method:dict(configs[name]) for method,name in selected.items()}
    chosen['rotated']['method']='rotated'
    strong=min(('quad4','quad16'),key=lambda name:summary[selected[name]]['kl'])
    strong_point=min(('point','center'),key=lambda name:summary[selected[name]]['kl'])
    selection=dict(configurations=chosen,selected_names=selected,strong_quadrature=strong,
        strong_point=strong_point,calibration_summaries=summary,calibration_files=files,
        lock_sha256=A.sha(out/'lock.json'),created_unix=time.time(),gate_labels_read_by_B=False,
        point_center_identity='within1e-12',point_center_max_absolute_difference=comparison['max_absolute_difference'],
        tie_rule='method/h/T insertion order; point before center; smallest h then T',
        selection_repair=dict(lock_file='selection_repair_lock.json',
            lock_sha256=A.sha(out/'selection_repair_lock.json'),source_hashes=extension_sources(),
            reason='GPU numerical reductions differed at floating-point roundoff; exact-score identity guard was too strict',
            tolerance=TOLERANCE))
    B.verify_lock(out,data);verify_extension(out)
    A.write(out/'selection.json',selection)
    (out/'selection.sha256').write_text(A.sha(out/'selection.json')+'\n')
    print('SELECTED',chosen,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('audit','freeze','select'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=B.ROOT/'results/round10_partition_data')
    args=parser.parse_args()
    if args.stage=='audit':
        _,_,_,_,comparison=calibration(args.out,args.data)
        print(json.dumps(comparison),flush=True)
    elif args.stage=='freeze':freeze(args.out,args.data)
    else:select(args.out,args.data)


if __name__=='__main__':main()
