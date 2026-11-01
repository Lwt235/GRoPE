"""Bind the new numerical tiling wrapper to the unchanged R15 audit and base.

The full 120-rectangle/240-split/12-FOV matrix and all tolerances come verbatim
from the frozen independent Cartesian adaptive auditor. This adds provenance
for both wrapper and frozen dependency, and verifies no child floor or token
identity leaks through the wrapper. No real observations are opened.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from experiments import round15_quartic_numerical_audit as A

ROOT=A.ROOT
PARENT=ROOT/'results/round15_quartic_numerical_audit_frozen'
PARENT_HASH='45cc4e513710d3c257f3f36d31138312ecbcd193b2a63ef4f3427899365787e6'
SOURCES=('experiments/round15_quartic_tiled_numerical_audit.py','tests/test_round15_quartic_tiled_numerical_audit.py',
    'experiments/round15_quartic_numerical_audit.py','tests/test_round15_quartic_numerical_audit.py',
    'models/round15_quartic_pe.py','models/round15_quartic_tiled.py')


def verify_parent(directory=PARENT,expected=PARENT_HASH):
    if A.sha(directory/'lock.json')!=expected or (directory/'lock.sha256').read_text().strip()!=expected:
        raise ValueError('frozen numerical parent lock changed')
    lock=json.loads((directory/'lock.json').read_text())
    for source,digest in lock['source_hashes'].items():
        if A.sha(source)!=digest or A.sha(directory/Path(source).name)!=digest:
            raise ValueError('frozen numerical parent source changed')
    if A.sha(Path(lock['report'])/'verification.json')!=lock['report_sha256'] or A.sha(Path(lock['report'])/'completion.json')!=lock['completion_sha256']:
        raise ValueError('frozen numerical parent report changed')
    return lock


def audit(out,order,device):
    if out.exists():raise ValueError('refuse overwrite numerical tiling run')
    verify_parent();out.mkdir(parents=True);started=time.time()
    sources={p:A.sha(ROOT/p) for p in SOURCES};snapshot=out/'sources';snapshot.mkdir()
    for path in SOURCES:
        dest=snapshot/path;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as file:file.write((ROOT/path).read_bytes())
    A.write(out/'started.json',dict(created_unix=started,source_hashes=sources,parent_lock_sha256=PARENT_HASH,
        quadrature_order=order,maximum_tile_side=1.,real_observations_read=False,labels_read=False,
        inherited_thresholds=dict(raw_absolute=A.RAW_ABSOLUTE_TOLERANCE,split_absolute=A.SPLIT_ABSOLUTE_TOLERANCE,
                                  normalized_probability=A.PROBABILITY_TOLERANCE,gradient=A.GRADIENT_TOLERANCE)))
    # Same frozen independent reference, same original native rectangle matrix.
    A.audit(out/'full_audit','models.round15_quartic_tiled',order,device)
    full=json.loads((out/'full_audit/verification.json').read_text())
    import torch
    from models.round15_quartic_tiled import quartic_cell_mass,integration_tiles
    bounds=np.array([[-.0625,-.0625,.9375,.9375],[-.0625,-.0625,1.9375,1.9375],
                     [-.0625,-.0625,2.9375,2.9375],[-.0625,-.0625,1.9375,.9375]])
    tensor=torch.tensor(bounds,device=device,dtype=torch.float64)
    tiles,owners=integration_tiles(tensor)
    children=tiles.detach().cpu().numpy();parent_ids=owners.detach().cpu().numpy()
    child_area=np.prod(children[:,2:]-children[:,:2],axis=1);original_area=np.prod(bounds[:,2:]-bounds[:,:2],axis=1)
    aggregate=np.bincount(parent_ids,weights=child_area,minlength=len(bounds))
    if not np.all(children[:,2:]-children[:,:2]<=1+1e-12):raise ValueError('tile too large')
    np.testing.assert_allclose(aggregate,original_area,atol=1e-12,rtol=0)
    # In deep underflow the raw API must be exactly zero. Any internal floor
    # would otherwise be hidden when the external native-area floor is added.
    mean=torch.tensor([[-128.,8.]],device=device,dtype=torch.float64)
    covariance=torch.zeros((1,2,2),device=device,dtype=torch.float64)
    with torch.no_grad():mass=quartic_cell_mass(mean,covariance,tensor,bandwidth=.25,quadrature_order=order)
    raw=mass.detach().cpu().numpy()
    if raw.shape!=(1,len(bounds)) or np.any(raw!=0):raise ValueError('child tokens/floor leaked through raw API')
    if any(A.sha(ROOT/p)!=digest for p,digest in sources.items()):raise ValueError('source changed during numerical audit')
    verify_parent()
    result=dict(passed=full['passed'],parent_lock_sha256=PARENT_HASH,source_hashes=sources,
        full_audit_verification_sha256=A.sha(out/'full_audit/verification.json'),
        full_audit_completion_sha256=A.sha(out/'full_audit/completion.json'),
        quadrature_order=order,maximum_tile_side=1.,original_key_count=len(bounds),
        numerical_tile_count=len(children),internal_floor_used=False,all_native_original_cases_retained=True,
        individual_rectangles=full['individual_rectangles'],split_checks=full['split_checks'],grid_conditions=full['grid_conditions'],
        maximum_raw_error=full['maximum_raw_error'],maximum_grid_probability_error=full['maximum_grid_probability_error'],
        maximum_split_absolute_defect=full['maximum_split_absolute_defect'],failures=full['failures'],
        real_observations_read=False,labels_read=False,validation_or_test_read=False,scientific_gate_passed=False,
        seconds=time.time()-started,scope=full['scope'])
    A.write(out/'verification.json',result)
    A.write(out/'completion.json',dict(files={str(p.relative_to(out)):A.sha(p) for p in out.rglob('*') if p.is_file()},
                                     passed=result['passed'],finished_unix=time.time()))
    print(json.dumps(result,indent=2),flush=True)


def freeze(out,report):
    if out.exists():raise ValueError('refuse overwrite final tiling audit freeze')
    verify_parent();result=json.loads((report/'verification.json').read_text())
    if not result['passed']:raise ValueError('failed numerical preflight cannot be frozen as passed')
    for path,digest in result['source_hashes'].items():
        if A.sha(ROOT/path)!=digest:raise ValueError('source changed since preflight')
    complete=json.loads((report/'completion.json').read_text())
    for path,digest in complete['files'].items():
        if A.sha(report/path)!=digest:raise ValueError('preflight artifact changed')
    out.mkdir(parents=True)
    for path in SOURCES:
        dest=out/'sources'/path;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as file:file.write((ROOT/path).read_bytes())
    lock=dict(created_unix=time.time(),parent_lock_sha256=PARENT_HASH,source_hashes=result['source_hashes'],
        report=str(report.resolve()),report_sha256=A.sha(report/'verification.json'),
        completion_sha256=A.sha(report/'completion.json'),quadrature_order=result['quadrature_order'],maximum_tile_side=1.,
        real_observations_read=False,labels_read=False,scientific_gate_passed=False)
    A.write(out/'lock.json',lock);(out/'lock.sha256').write_text(A.sha(out/'lock.json')+'\n')
    print('FINAL_TILED_NUMERICAL_FREEZE',A.sha(out/'lock.json'),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('audit','freeze'))
    parser.add_argument('--out',required=True,type=Path);parser.add_argument('--order',default=16,type=int)
    parser.add_argument('--device',default='cpu');parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.stage=='audit':audit(args.out,args.order,args.device)
    else:freeze(args.out,args.report)


if __name__=='__main__':main()
