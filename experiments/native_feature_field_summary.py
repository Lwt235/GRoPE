"""Fixed two-factor interaction and formed-state statistics. No model execution."""
import csv,time
import numpy as np
import torch
from experiments import native_feature_field_data as d
from experiments.native_render_parameter_prepare import rotation

def shape_statistics():
    rows=[]
    for meta in d.metas():
        rid=meta['id'];base=d.load(d.original_scene(rid));valid=base['has_content'];R=rotation(base['rotations']);A=(R*base['scales'].double().square()[:,None,:])@R.transpose(-1,-2);old_e=torch.linalg.eigvalsh(A)
        geometries=[]
        for row in ('C','J'):
            d.context(row);f=d.load(d.old.scene_path(rid));g=d.load(d.fields_path(rid));geometries.append(g)
            R=rotation(f['rotations']);B=(R*f['scales'].double().square()[:,None,:])@R.transpose(-1,-2);e=torch.linalg.eigvalsh(B);change=(B-A).norm(dim=(-2,-1))/A.norm(dim=(-2,-1)).clamp_min(1e-30);norms=f['features'][valid].float().norm(dim=-1)
            stats=d.read(d.OUT/'formation'/f'{rid}.json');init=np.mean([x['raw_MSE'] for x in stats['initial']]);fin=np.mean([x['raw_MSE'] for x in stats['final']])
            rows.append(dict(row=row,id=rid,place=meta['place'],source_initial_raw_MSE=init,source_final_raw_MSE=fin,source_relative_reduction_percent=100*(1-fin/init),feature_rms=float(f['features'][valid].float().square().mean().sqrt()),feature_norm_median=float(norms.median()),feature_norm_p90=float(norms.quantile(.9)),cov_relative_change_median=float(change[valid].median()),cov_relative_change_p90=float(change[valid].quantile(.9)),trace_ratio_median=float((e.sum(-1)/old_e.sum(-1))[valid].median()),thinnest_axis_ratio_median=float((e[:,0]/old_e[:,0])[valid].median()),anisotropy_median=float((e[:,-1]/e[:,0])[valid].median()),first_gradients=stats['first_gradients'],seconds=stats['seconds']))
        assert all(torch.equal(geometries[0][key],geometries[1][key]) for key in ['means','B_world','density4','valid','source_camera','unit'])
    d.write(d.BASEOUT/'formation_statistics.json',dict(complete=True,rows=rows,centers_B_density_validity_identical_C_J=True,source_loss_not_O6_success=True))

def factorial():
    ss={row:d.read(d.BASEOUT/row/'scoring/summary.json') for row in ('C','J')};prefix=ss['J']['primary'];arr={row:{k:np.array(v) for k,v in s['primary_arrays'].items()} for row,s in ss.items()};a=arr['J']['P']-arr['J']['G'];b=arr['C']['P']-arr['C']['G'];inter=a-b
    rng=np.random.default_rng(20261113);si=rng.integers(0,3,(5000,3,1));pi=rng.integers(0,12,(5000,1,12))
    boot=inter[si,pi].mean((1,2));jb=arr['J']['P'][si,pi].mean((1,2));cb=arr['C']['P'][si,pi].mean((1,2));relative=100*(a[si,pi].mean((1,2))/jb-b[si,pi].mean((1,2))/cb)
    interaction=dict(definition='(E_JP-E_JG)-(E_CP-E_CG), positive favors formation-conditioned G increment',absolute=float(inter.mean()),absolute_CI95=np.quantile(boot,[.025,.975]).tolist(),per_seed_absolute=inter.mean(1).tolist(),per_place_absolute=inter.mean(0).tolist(),relative_percentage_point=float(100*(a.mean()/arr['J']['P'].mean()-b.mean()/arr['C']['P'].mean())),relative_percentage_point_CI95=np.quantile(relative,[.025,.975]).tolist(),bootstrap_replicates=5000,bootstrap_seed=20261113,paired_both_rows=True,interpretation='Difference of actual paired increments; never infer interaction from difference in significance flags')
    # C/J effects with a fixed PE across rows, and within-row contrasts retained in their full tables.
    contrasts={}
    from experiments.native_feature_field_score import contrast
    for k in ('P','S','G','RasterFill'):
        contrasts['J-'+k+'_vs_C-'+k]=contrast(arr['J'][k],arr['C'][k],(si,pi))
    flags={row:d.read(d.BASEOUT/row/'method_decision.json') for row in ('C','J')}
    d.write(d.BASEOUT/'factorial_summary.json',dict(complete=True,primary_J_G_vs_P=ss['J']['comparisons'][prefix+'/G-P'],Delta_J=dict(definition='E_JP-E_JG; positive is beneficial',absolute=float(a.mean()),absolute_CI95=np.quantile(a[si,pi].mean((1,2)),[.025,.975]).tolist(),relative_percent=float(100*a.mean()/arr['J']['P'].mean())),interaction=interaction,cross_row_main=contrasts,all_within_row_files={row:str(d.BASEOUT/row/'scoring/summary.json') for row in ('C','J')},fresh=False))
    d.write(d.BASEOUT/'method_decision.json',dict(complete=True,status='complete_pending_original_Astra_scientific_assessment',within_row_flags=flags,interaction=interaction,joint_formation_interaction_positive_interval=interaction['absolute_CI95'][0]>0,scientific_candidate_selected=False,at_most_one_candidate_by_Astra=True,mechanism_not_established_by_source_reconstruction=True,overall_goal_achieved=False,fresh=False,no_successor_selected=True))
    print(dict(J_G_P=ss['J']['comparisons'][prefix+'/G-P'],interaction=interaction),flush=True)
if __name__=='__main__':
    torch.set_num_threads(4);shape_statistics();factorial()
