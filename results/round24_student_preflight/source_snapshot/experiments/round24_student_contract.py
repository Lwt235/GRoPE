"""R24 fixed synthetic-only t4 numerical contract; no empirical inputs."""
from __future__ import annotations
from copy import deepcopy
import numpy as np

POLICY = dict(h=.25, floor_density=1e-30, degrees_of_freedom=4, covariance_admission_relative_tolerance=1e-12,
              quadrature_order=24, maximum_tile_side_in_h=4., max_cell_node_evaluations=4000000)
REFERENCE_POLICY = dict(orders=[64,96], maximum_panel_side_in_h=4.,
                        max_total_cell_node_evaluations=2000000, max_phase_wall_seconds=120.)
GATES = dict(probability_max_absolute=1e-4, raw_fov_relative=1e-6,
             raw_l1_over_reference_fov=1e-6, reference_probability_max_absolute=1e-8,
             reference_raw_relative=1e-8, normalization_absolute=2e-12,
             direct_fov_additivity_absolute=1e-8, integrated_identity_absolute=1e-8,
             production_max_phase_wall_seconds=120.)
BINDINGS = ('results/round23_ray_root_acceptance_20261031.json',
            'results/round23_ray_post_result_decision_20261031.json',
            'docs/round15_student_t_reserve.md',
            'docs/student_t4_route_comparison_unselected_draft.md')
SOURCE_FILES = ('ROUND24_STUDENT_PREFLIGHT_PROTOCOL.md',
                'experiments/round24_student_contract.py',
                'models/round24_student_t_pe.py',
                'experiments/round24_student_preflight.py',
                'experiments/round24_student_reference.py',
                'experiments/round24_student_audit.py',
                'tests/test_round24_student_pe.py',
                'tests/test_round24_student_reference.py')

def grid(nx=4,ny=4,x0=-2.,y0=-2.):
    return np.asarray([[x0+x,y0+y,x0+x+1,y0+y+1] for y in range(ny) for x in range(nx)],np.float64)

def fixtures():
    R=np.asarray([[np.cos(.6),-np.sin(.6)],[np.sin(.6),np.cos(.6)]])
    v=np.asarray([np.cos(.73),np.sin(.73)])
    data=[
      ('point_center',[.5,.5],np.zeros((2,2)),grid()),
      ('nominal8',[.2,-.3],np.eye(2)/12,grid()),
      ('nominal16',[.2,-.3],np.eye(2)/3,grid()),
      ('nominal24',[.2,-.3],np.eye(2)*.75,grid()),
      ('rotated',[.2,-.3],R@np.diag([2.,.04])@R.T,grid()),
      ('rankone',[.2,-.3],np.outer([1.,.5],[1.,.5]),grid()),
      ('outside',[3.,0.],np.eye(2)*.1,grid()),
      ('boundary',[2.,0.],np.eye(2)*.3,grid()),
      ('tiny_far',[1e6,0.],np.asarray([[1.,.3],[.3,.5]]),grid()),
      ('thin_wide',[.2,-.3],100*np.outer(v,v),grid()),
      ('broad',[.2,-.3],np.asarray([[100.,70.],[70.,100.]]),grid()),
      ('near_point',[.23,-.17],np.asarray([[.001,.0009],[.0009,.002]]),grid()),
      ('anchor_second_moment',[.2,-.3],np.asarray([[.3,.1],[.1,.2]])+np.outer([.8,-.6],[.8,-.6]),grid()),
      ('negative_correlation',[-.4,.7],np.asarray([[2.,-1.7],[-1.7,2.]]),grid()),
      ('native432',[12.,9.],np.asarray([[.75,.3],[.3,.8]]),grid(24,18,0.,0.)),
      ('far_native432',[40.,0.],np.zeros((2,2)),grid(24,18,0.,0.)),
      ('corner_point',[.0,.0],np.zeros((2,2)),grid()),
      ('rankone_diagonal',[.2,-.3],np.asarray([[2.,2.],[2.,2.]]),grid()),
    ]
    rows=[]
    for name,mu,cov,bounds in data:
        for content in (False,True):
            centers=(bounds[:,:2]+bounds[:,2:])/2
            similarity=np.sin(.71*centers[:,0]+.23)*np.cos(.37*centers[:,1]-.19)
            tau=10*POLICY['h']**2/(POLICY['h']**2+float(np.trace(cov)))
            bias=tau*similarity if content else np.zeros(len(bounds))
            rows.append(dict(id=name+('_t10' if content else '_t0'),kind='base',mu=np.asarray(mu,dtype=np.float64).tolist(),
                covariance=((np.asarray(cov,dtype=np.float64)+np.asarray(cov,dtype=np.float64).T)/2).tolist(),bounds=bounds.tolist(),content_bias=bias.tolist(),
                content_temperature=tau if content else 0.,policy_overrides={}))
    parents={r['id']:r for r in rows}
    parent=parents['rotated_t0']
    def identity(name,base,relation,**extras):
        row=deepcopy(base);row.update(id=name,kind='identity',identity=dict(parent=base['id'],relation=relation,**extras));return row
    t=identity('translation',parent,'equal');shift=np.asarray([1.25,-.75]);t['mu']=(np.asarray(t['mu'])+shift).tolist();t['bounds']=(np.asarray(t['bounds'])+np.tile(shift,2)).tolist();rows.append(t)
    t=identity('screen_units',parent,'equal');scale=3.7;t['mu']=(np.asarray(t['mu'])*scale).tolist();t['covariance']=(np.asarray(t['covariance'])*scale**2).tolist();t['bounds']=(np.asarray(t['bounds'])*scale).tolist();t['policy_overrides']=dict(h=POLICY['h']*scale,floor_density=POLICY['floor_density']/scale**2);rows.append(t)
    perm=list(range(15,-1,-1));t=identity('key_permutation',parent,'permutation',permutation=perm);t['bounds']=[t['bounds'][i] for i in perm];t['content_bias']=[t['content_bias'][i] for i in perm];rows.append(t)
    # Unequal rectangles that tile the original FOV, with exact cell unions.
    groups=[[0,1],[2,3,6,7,10,11],[4,5,8,9,12,13],[14,15]]
    t=identity('unequal_merge',parent,'merge',groups=groups);b=np.asarray(parent['bounds']);t['bounds']=[np.r_[b[g,:2].min(0),b[g,2:].max(0)].tolist() for g in groups];t['content_bias']=[0.]*4;rows.append(t)
    t=identity('axis_swap',parent,'equal');P=np.asarray([[0.,1.],[1.,0.]]);t['mu']=(P@np.asarray(t['mu'])).tolist();t['covariance']=(P@np.asarray(t['covariance'])@P.T).tolist();t['bounds']=np.asarray(t['bounds'])[:,[1,0,3,2]].tolist();rows.append(t)
    t=identity('zero_covariance_repeat',parents['point_center_t0'],'equal');rows.append(t)
    assert len(rows)==42 and len({r['id'] for r in rows})==42
    return rows
