"""Seal completed round036 artifacts; no predictions, metrics or control edits."""
import os, subprocess, time
from pathlib import Path
from experiments import stream_factorization_data as d

def compact(v):
 return {k:v[k] for k in ('mean','interval95','defined_places') if k in v}

def main():
 start=time.monotonic();s=d.read(d.OUT/'scoring/summary.json');c=d.read(d.OUT/'costs.json')
 assert s['complete'] and c['actual_all_waited'] and len(list((d.OUT/'completed').glob('*.json')))==12
 assert d.read(d.OUT/'interface_validation.json')['complete'] and d.read(d.OUT/'figure_review.json')['all_four_actually_viewed']
 assert (d.OUT/'report.md').is_file() and not any((d.DATA/'resume').glob('**/*.bin'))
 receipts=d.read(d.OUT/'owner_exit_receipts.json');assert all(v['actual_exit_code']==0 for v in receipts['completed_owners'])
 runtimes={p.stem:d.read(p) for p in d.OUT.glob('CPU_*_runtime.json')}
 runtimes['CPU_figures_first_runtime']=d.read(d.OUT/'engineering_revisions/figure007.json')['first_figures_CPU_runtime']
 runtimes['CPU_report_first_runtime']=d.read(d.OUT/'engineering_revisions/report008.json')['first_report_CPU_runtime']
 assert all(v['complete'] and not v['CUDA_initialized'] for v in runtimes.values())
 pools={p.stem:d.read(p) for p in d.OUT.glob('pool_*_complete.json')}
 assert len(pools)==2 and all(v['actual_all_children_waited'] and not v['failures'] for v in pools.values())
 pids=sorted({v['PID'] for v in c['workers']}|{v['manager_PID'] for v in pools.values()}|{v['PID'] for v in runtimes.values()})
 check=subprocess.run(['ps','-o','pid=,args=','-p',','.join(map(str,pids))],capture_output=True,text=True)
 active=[line for line in check.stdout.splitlines() if 'experiments.stream_factorization_' in line]
 assert not active,active
 gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
 assert gpu.returncode==0
 gpu_pids={int(line.split(',')[0]) for line in gpu.stdout.splitlines() if line.strip()}
 assert not gpu_pids.intersection({w['PID'] for w in c['workers']})
 identity=d.read(d.OUT/'execution_identity.json')
 identity['source_files']={str(p.relative_to(d.ROOT)):d.sha(p) for p in sorted((d.ROOT/'experiments').glob('stream_factorization_*.py'))}
 identity.update(status='AllGPUandscoring/validation/figures/reportprocessesactualexit0;metadata sealingonly.',source_identity_snapshot_utc=d.now())
 d.write(d.OUT/'execution_identity.json',identity)
 terminal=dict(complete=True,registered_utc=d.now(),GPU_workers_all_actual_waited=True,GPU_actual_exit_codes=[dict(PID=w['PID'],GPU=w['GPU_index'],exit_code=w['exit_code'],actual_wait=w['actual_wait_returned']) for w in c['workers']],GPU_managers=pools,owner_exit_receipts=receipts,CPU_phase_runtimes=runtimes,own_phase_process_check=check.stdout,active_owned_phase_processes=active,own_GPU_processes_present=False,host_NVML_snapshot=gpu.stdout,GPU_worker_hours=c['all_GPU_worker_hours'],preflight_cost_hours=c['preflight_cost_seconds']/3600,DATA_final_bytes=d.size(d.DATA),DATA_peak_sampled_bytes=c['DATA_peak_sampled_bytes'],DATA_conservative_bound_bytes=c['DATA_conservative_bound_bytes'],project_final_before_seal_bytes=d.size(d.OUT),project_run_sampled_peak_bytes=max(v['project_peak_sampled_bytes'] for v in pools.values()),new_resume_bank_files_remaining=0,metadata_writer_PID=os.getpid(),metadata_writer_scope='ThisshortCPUartifactsealerreturnsafterwriting;itsactualtoolreturnisreportedto mainatFINAL. Itdoesnotloadamodelorcomputeanewmetric.',no_training_or_download=True,main_control_files_modified=False)
 assert terminal['project_final_before_seal_bytes']<2**30
 d.write(d.OUT/'terminal_record.json',terminal)
 artifacts={str(p.relative_to(d.OUT)):dict(bytes=p.stat().st_size,sha256=d.sha(p)) for p in sorted(d.OUT.rglob('*')) if p.is_file() and p.name not in ('manifest.json','implementation_handoff.json')}
 manifest=dict(complete=True,utc=d.now(),scope='AllnewprojectartifactsbelowOUTexceptmanifestandhandofftoavoidcircularhashes. NewpredictionarchiveSHAinheritedfromone-timeworkercommit; oldweightsnotrehashed.',artifacts=artifacts,predictions=[d.read(p) for p in sorted((d.OUT/'completed').glob('*.json'))],source_identity=identity,science_plan=dict(path=str(d.SCIENCE),sha256=d.SCIENCE_SHA),canonical_goal=d.read(d.OUT/'goal_pointer_identity.json'),read_only_inherited_dependencies=dict(configuration=d.plan()['old_configuration'],configuration_sha256=d.plan()['old_configuration_sha256'],fixed_label_markers=d.plan()['fixed_label_markers'],WW_A_metric_source=str(d.OLDOUT/'scoring'),model_path=d.plan()['model_path'],model_sha256_inherited=d.plan()['model_sha256_inherited']),document_links=[str(d.ROOT/'docs'/x) for x in ('research_current.md','research_outline_alignment.md','gaussian_memory_evidence_scope.md')])
 d.write(d.OUT/'manifest.json',manifest)
 handoff=dict(complete=True,round=36,experiment='stream_history_factorization1',role='Originalgpt-6.1-sol/high;Fastunsupported/effectivebackendunknown',science_plan=dict(path=str(d.SCIENCE),sha256=d.SCIENCE_SHA),report=dict(path=str(d.OUT/'report.md'),sha256=d.sha(d.OUT/'report.md')),manifest=dict(path=str(d.OUT/'manifest.json'),sha256=d.sha(d.OUT/'manifest.json')),canonical_goal=d.read(d.OUT/'goal_pointer_identity.json'),results=dict(primary=compact(s['primary']),revisit_effects={k:compact(v['relative_rotation_deg']) for k,v in s['comparisons']['revisit_coverage_ge5pct'].items()},interaction=compact(s['interaction']['revisit_coverage_ge5pct']['relative_rotation_deg']),all96_AbsRel_effects={k:compact(v['absrel']) for k,v in s['comparisons']['all96frames'].items()},flags={k:{a:v[a] for a in ('practical_rotation','obvious_AbsRel_cost')} for k,v in s['predefined_flags'].items()}),artifacts=dict(scores=str(d.OUT/'scoring/summary.json'),costs=str(d.OUT/'costs.json'),bytes=str(d.OUT/'history_bytes.json'),calls=str(d.OUT/'call_accounting.json'),validation=str(d.OUT/'interface_validation.json'),figures_review=str(d.OUT/'figure_review.json'),terminal=str(d.OUT/'terminal_record.json')),budget=dict(GPU_worker_hours=c['all_GPU_worker_hours'],preflight_cost_seconds=c['preflight_cost_seconds'],DATA_bound_GiB=c['DATA_conservative_bound_bytes']/2**30,worker_peak_VRAM_GiB=c['peak_NVML_worker_MiB']/1024),implementation_material_issues=[],interpretation_limits=['Alluseddevelopment; nofreshoroverallgoalclaim.','Fourcellsareconditionalper-framecounterfactuals,notindependentonlinetrajectories.','FullFteacher/Wshadowandhardwarecopiesarecharged;notanequalbyteboundedmethod.','MainI_readflagfalse;jointpositivewithrawdepthcost;interactionanddirectA_FFuncertain.NoautomaticP2orself-selectedsuccessor.'],actual_completion='AllGPUworkers/managersandCPUsciencephasesactualwaited;sealeractualreturnreportedto mainbeforeFINAL.',scientific_disposition_owner='mainreturnsto originalAstra;currentimplementationleaseendsat actualFINAL')
 d.write(d.OUT/'implementation_handoff.json',handoff)
 print(dict(sealed=True,PID=os.getpid(),seconds=time.monotonic()-start,report_sha256=d.sha(d.OUT/'report.md'),manifest_sha256=d.sha(d.OUT/'manifest.json'),handoff_sha256=d.sha(d.OUT/'implementation_handoff.json'),project_bytes=d.size(d.OUT)),flush=True)

if __name__=='__main__':main()
