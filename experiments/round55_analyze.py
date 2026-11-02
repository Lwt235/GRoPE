"""Paired seed/scene summaries; samples are not treated as independent trials."""
import numpy as np
from experiments.round55_data import OUT, read, write, sha
from experiments.round55_train import ROLES, SEEDS, METRICS, macro, ORIGINS


def main():
    records={(role,seed):read(OUT/'evaluation'/role/f'{seed}_baseline.json') for role in ROLES for seed in SEEDS}
    scenes=sorted({r['scene'] for r in records[ROLES[0],SEEDS[0]]['rows']})
    scene_metrics={}
    seed_stats={}
    for role in ROLES:
        seed_stats[role]={k:dict(mean=float(np.mean([records[role,s]['metrics'][k] for s in SEEDS])),
            seed_std=float(np.std([records[role,s]['metrics'][k] for s in SEEDS],ddof=1))) for k in METRICS}
        scene_metrics[role]={}
        for scene in scenes:
            values=[macro([r for r in records[role,s]['rows'] if r['scene']==scene]) for s in SEEDS]
            scene_metrics[role][scene]={k:float(np.mean([r[k] for r in values])) for k in METRICS}
            for key in ('epe_p95','soft_p95'):
                scene_metrics[role][scene][key]=float(np.mean([records[role,s]['scene_tails'][scene][key] for s in SEEDS]))
    contrasts={}
    for role in ('ray_gaussian','camera_plane','camera_gs'):
        contrasts[role]={}
        for control in ROLES:
            if control==role:continue
            rows={}
            for metric in METRICS:
                per_scene={scene:scene_metrics[role][scene][metric]-scene_metrics[control][scene][metric] for scene in scenes}
                lower=metric in ('ce','epe_argmax','epe_soft','tail16','soft_tail16')
                rows[metric]=dict(per_scene=per_scene,scene_wins=sum(v<0 if lower else v>0 for v in per_scene.values()))
            contrasts[role][control]=rows
    runs=read(OUT/'selection.json')['records']
    cost={role:dict(mean_training_selection_seconds=float(np.mean([r['training_selection_seconds'] for r in runs if r['role']==role])),
                   peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in runs if r['role']==role),
                   reused=all(r.get('reused',False) for r in runs if r['role']==role),
                   parameters=next(r['parameters'] for r in runs if r['role']==role),
                   pe_parameters=next(r['pe_parameters'] for r in runs if r['role']==role)) for role in ROLES}
    original={(role,seed):read(ORIGINS[role]/'evaluation'/role/f'{seed}_baseline.json') for role in ROLES for seed in SEEDS}
    original_identities={f'{r}_{s}':sha(ORIGINS[r]/'evaluation'/r/f'{s}_baseline.json') for r in ROLES for s in SEEDS}
    budget_effect={}
    for role in ROLES:
        budget_effect[role]={}
        for metric in METRICS:
            before=[original[role,s]['metrics'][metric] for s in SEEDS]
            after=[records[role,s]['metrics'][metric] for s in SEEDS]
            budget_effect[role][metric]=dict(original_mean=float(np.mean(before)),extended_mean=float(np.mean(after)),
                per_seed_differences=[b-a for a,b in zip(before,after)],mean_difference=float(np.mean(np.array(after)-before)))
    interaction={}
    for candidate,control in (('ray_gaussian','ray_point'),('ray_gaussian','ray_iso'),('ray_gaussian','ray_affine'),
                              ('camera_plane','camera_only'),('camera_plane','camera_offset'),('camera_gs','camera_only')):
        name=candidate+'__vs__'+control;interaction[name]={}
        for metric in METRICS:
            before=[original[candidate,s]['metrics'][metric]-original[control,s]['metrics'][metric] for s in SEEDS]
            after=[records[candidate,s]['metrics'][metric]-records[control,s]['metrics'][metric] for s in SEEDS]
            interaction[name][metric]=dict(original_per_seed=before,extended_per_seed=after,
                original_mean=float(np.mean(before)),extended_mean=float(np.mean(after)),
                difference_of_differences=float(np.mean(np.array(after)-before)))
    prefix=[read(OUT/'runs'/r/str(s)/'prefix_check.json') for r in ROLES for s in SEEDS]
    write(OUT/'budget_comparison.json',dict(budget_effect=budget_effect,interaction=interaction,
        original_evaluation_identities=original_identities,source_sha256=sha(__file__),
        prefix_max_function_error=max(x['selected_function_max_difference'] for x in prefix),
        prefix_max_history_ce_error=max(x['history_ce_max_difference'] for x in prefix),
        prefix_warnings=sum(x['warning'] for x in prefix),
        caveat='paired development reuse; original1000 and extended5000 are correlated,not independent trials',confirmation_access=False))
    write(OUT/'analysis.json',dict(seed_stats=seed_stats,scene_metrics=scene_metrics,contrasts=contrasts,cost=cost,
        sources=dict(script_sha256=sha(__file__),summary_sha256=sha(OUT/'summary.json')),
        new_training_selection_seconds=sum(r['training_selection_seconds'] for r in runs if not r.get('reused')),
        reused_reference_count=sum(bool(r.get('reused')) for r in runs),
        selection_at_final_update=sum(r['best_step']==5000 for r in runs),
        field_tensor_changes={f"{r['role']}_{r['seed']}":r['field_changes'] for r in runs},
        caveat='3 paired seeds and 5 reused scenes; no per-query significance claim; fixed-budget not proof of convergence.',
        confirmation_access=False))
    for role in ROLES:
        m=seed_stats[role]
        print(role, 'soft EPE',round(m['epe_soft']['mean'],4),'PCK8',round(m['pck8']['mean']*100,3),
              'PCK4',round(m['pck4']['mean']*100,3),'AUC8',round(m['auc8']['mean']*100,3),flush=True)


if __name__=='__main__':main()
