"""CPU-only evidence assembly and fixed scientific figures; no new model queries."""
import os,time,resource,collections,json
from pathlib import Path
import numpy as np
import torch
from experiments import stream_history_data as d

def main():
 start=time.monotonic();torch.set_num_threads(2)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
 plan=d.plan();summary=d.read(d.OUT/'scoring/summary.json');by=d.read(d.OUT/'scoring/by_place.json');sel=d.read(d.OUT/'scoring/selection.json');pairs=d.read(d.OUT/'scoring/pairs.json');counter=d.read(d.OUT/'scoring/all_counterexamples.json');frames=d.read(d.OUT/'scoring/frames.json')
 workers=[d.read(p) for p in sorted((d.OUT/'workers').glob('*.json'))];assert all(w['actual_wait_returned'] for w in workers)
 events=[]
 for p in (d.OUT/'calls').glob('*.jsonl'):
  events.extend(dict(worker=p.stem,**json.loads(line)) for line in p.read_text().splitlines())
 form=[e for e in events if e['kind']=='formal_frame'];identity=[(e['id'],e['frame']) for e in form];assert len(identity)==len(set(identity))==1152
 assert set(identity)=={(m['id'],t) for m in plan['catalog'] for t in range(96)}
 modules=collections.Counter(e['module'] for e in events if e['kind']=='native_module_complete');begins=collections.Counter(e['module'] for e in events if e['kind']=='native_module_begin');assert modules==begins
 labelres=[d.read(p) for p in (d.OUT/'worker_resources').glob('*.json') if d.read(p).get('phase')=='labels']
 expected={'aggregator':2313,'camera_head':3465,'depth_head':2313,'point_head':3};assert dict(modules)==expected,(modules,expected)
 counts=dict(unique_formal_streams=12,unique_formal_frames=1152,duplicate_formal_queries=0,formal_aggregator_calls=2304,formal_camera_head_calls=3456,formal_camera_refinement_iterations=13824,formal_depth_head_calls=2304,preflight_native_frames=9,preflight_direct_attention_calls=2,all_native_module_begins=dict(begins),all_native_module_completions=dict(modules),GT_GPU_projection_calls=sum(r['GT_projection_calls'] for r in labelres),GT_candidates=12*2016,GT_selected_pairs=756,GT_independent_NumPy_reference_projections=12,training_updates=0,downloads=0,finite_formal_frames=sum(e['finite_pose'] and e['finite_depth'] for e in form),model_loads=sum(e['kind']=='model_loaded' for e in events))
 assert counts['GT_GPU_projection_calls']==12*2583
 stage=collections.defaultdict(list);per=[];allcost=[]
 for m in plan['catalog']:
  pred=d.load(d.DATA/'predictions'/f"{m['id']}.pt");cost=pred['frame_costs'];assert len(cost)==96
  for c in cost:
   allcost.append(dict(id=m['id'],**c))
   for k,v in c.items():
    if isinstance(v,(int,float)) and k not in ('frame','peak_allocated_bytes','peak_reserved_bytes'):stage[k].append(v)
  per.append(dict(id=m['id'],total_frame_seconds=sum(c['frame_wall_before_checkpoint'] for c in cost),F_stage_seconds=sum(c['full_aggregator']+c['camera_F']+c['full_depth'] for c in cost),A_stage_seconds=sum(c['full_aggregator']+c['camera_A']+c['full_depth'] for c in cost),W_stage_seconds=sum(c['window_aggregator']+c['camera_W']+c['window_depth'] for c in cost),final_state=cost[-1]['state']))
 stage_summary={k:dict(total_seconds=float(sum(v)),mean_seconds=float(np.mean(v)),p50_seconds=float(np.median(v)),p95_seconds=float(np.quantile(v,.95)),records=len(v)) for k,v in stage.items()}
 checkpoints=[e for e in events if e['kind']=='checkpoint_IO'];loads=[e for e in events if e['kind']=='model_loaded'];restores=[e for e in events if e['kind']=='restore']
 pools=[d.read(p) for p in (d.OUT).glob('pool_*_complete.json')]+[d.read(p) for p in (d.OUT/'engineering_revisions').glob('pool_*interrupted*.json')]
 worker_seconds=sum(w['elapsed_seconds'] for w in workers);profile_seconds=sum(w['elapsed_seconds'] for w in workers if w['phase']=='profile');resources=[d.read(p) for p in (d.OUT/'worker_resources').glob('*.json')]
 costreport=dict(all_worker_seconds=worker_seconds,all_worker_hours=worker_seconds/3600,profiling_seconds=profile_seconds,phases={phase:sum(w['elapsed_seconds'] for w in workers if w['phase']==phase) for phase in ('profile','formal','labels')},worker_processes=len(workers),worker_model_loads=len(loads),checkpoint_IO_seconds=sum(e['seconds'] for e in checkpoints),checkpoints=len(checkpoints),restores=restores,model_load_seconds=sum(e['model_seconds'] for e in loads),stage_component_times=stage_summary,per_stream=per,peak_NVML_worker_MiB=max(p.get('peak_NVML_worker_MiB',0) for p in pools),peak_GPU_allocated_bytes=max(r.get('peak_GPU_allocated_bytes',0) for r in resources),peak_GPU_reserved_bytes=max(r.get('peak_GPU_reserved_bytes',0) for r in resources),peak_RSS_single_worker_bytes=max(r.get('peak_RSS_bytes',0) for r in resources),DATA_peak_sampled_bytes=max(p.get('DATA_peak_sampled_bytes',0) for p in pools),DATA_final_bytes=d.size(d.DATA),project_current_bytes=d.size(d.OUT),worker_failure_exits=[w for w in workers if w['exit_code'] not in (0,75)],yield_exits=[w for w in workers if w['exit_code']==75],scope='Actual Popen-to-wait worker wall includes imports/model loads, original RGB reads, CUDA work, transfers, all recovery/loading, checkpoint/atomic output I/O, failures and GT labels. CPU manager idle wait is separate. Stage times include explicit synchronization and some CPU output copies; not attention-only measurements or isolated end-to-end benchmarks. F/A share actual aggregator and depth computation exactly once; listed stand-alone stage sums each charge that common dependency and cannot be summed to obtain actual joint wall. No uncached-OS cold-start or steady-state claim.')
 assert worker_seconds<4*3600 and profile_seconds<.5*3600 and costreport['peak_NVML_worker_MiB']<=40*1024
 assert costreport['DATA_peak_sampled_bytes']<16*2**30
 # Conservative analytic bound includes every permanent prediction together
 # with the largest live recovery set, a whole duplicate bounded save, and
 # one full prediction atomic temporary. GT labels run only after inference.
 cpall=[d.read(p) for p in (d.OUT/'checkpoint_records').glob('*.json')]
 predbytes=sum(d.read(p)['bytes'] for p in (d.OUT/'completed').glob('*.json'))
 recovery_peak=max(sum(z['bytes'] for z in cp['chunks'])+sum(cp['bytes_by_file'].values())+max(cp['bytes_by_file'].values()) for cp in cpall)
 prediction_atomic=max(d.read(p)['bytes'] for p in (d.OUT/'completed').glob('*.json'))
 costreport['DATA_conservative_engineering_bound_bytes']=max(predbytes+recovery_peak+prediction_atomic,d.size(d.DATA)+256*2**20)
 costreport['DATA_bound_scope']='All12permanent predictions + largest complete checkpoint set + entire duplicate largest bounded/output save + one entire final-prediction atomic temporary; deliberately overlaps some mutually exclusive operations. Labels execute only after all inference and add <=256MiB atomic allowance to actual final phase. No external/old data counted as new; actual history backing separately charged.'
 assert costreport['DATA_conservative_engineering_bound_bytes']<16*2**30

 memory=[]
 for p in sorted((d.OUT/'memory_accounting').glob('*.json')):
  rec=d.read(p);rid=p.name.split('_formal')[0];rid=rid.split('_profile')[0]
  resourcepath=d.OUT/'worker_resources'/p.name.removeprefix(rid+'_')
  if not resourcepath.exists():continue
  res=d.read(resourcepath);last=res['last_complete_frame'];cp=d.read(d.OUT/'checkpoint_records'/f'{rid}_{last:03d}.json');fb=cp['bytes_by_file'];chunks=sum(c['bytes'] for c in cp['chunks']);manifest_bytes=len(json.dumps(cp).encode())
  arms={}
  for arm in ('F','A','W'):
   recovery=(chunks if arm!='W' else fb['window'])+fb['cam'+arm]
   source_RGB_bytes=sum(f['rgb_bytes'] for f in next(m for m in plan['catalog'] if m['id']==rid)['frames'][:last+1])
   arms[arm]=dict(source_RGB_history_bytes_if_readable=source_RGB_bytes,conservative_total_with_readable_RGB_bytes=rec[arm]['total_bytes']+recovery+manifest_bytes+source_RGB_bytes,conservative_history_peak_with_CPU_staging_and_atomic_duplicate_bytes=rec[arm]['total_bytes']+3*recovery+2*manifest_bytes,forward_history_measured_bytes=rec[arm]['total_bytes'],forward_tensor_storage_bytes=rec[arm]['tensor_unique_storage_bytes'],forward_Python_metadata_bytes=rec[arm]['Python_object_header_and_container_bytes'],own_recovery_copy_file_bytes=recovery,conservative_entire_shared_manifest_bytes=manifest_bytes,total_operational_history_including_recovery_bytes=rec[arm]['total_bytes']+recovery+manifest_bytes)
  memory.append(dict(id=rid,last_frame=last,complete=res['complete'],arms=arms,shared_F_A_full_chunks_bytes=chunks,scoring_output_archive_bytes_excluded_from_forward_history=fb['prediction_archive'],source=str(p)))
 assert any(r['complete'] and r['last_frame']==95 for r in memory)
 memory_report=dict(records=memory,scope='Distinct physical tensor backing plus actual Python/cache metadata. Each per-path operational total additionally charges its independent model-accessible exact recovery files and, conservatively, the entire shared manifest. Full aggregator is a real alias between F/A, counted once in joint physical runtime; per-arm ledgers each charge the required dependency. Output depth/pose archives and offline GT labels are never forward inputs and are separate. Checkpoints are temporary but are NOT a free history side channel.',source_RGB_archive_bytes=sum(f['rgb_bytes'] for m in plan['catalog'] for f in m['frames']),source_depth_archive_bytes=sum(f['depth_bytes'] for m in plan['catalog'] for f in m['frames']),source_archive_scope='Existing immutable selected RGB-D inventory; one monotonically advancing current RGB is read per formal frame, never historical RGB replay. GT depth/pose only offline labels/scoring. Source files remain physically present; a future method with historical RGB re-read capability must add its backing, regardless of disk location.',logical_state_by_frame=[dict(frame=c['frame'],state=c['state']) for c in allcost if c['id']=='fidelity_00'],B_convention='B_forward8 is actual W forward cache+necessary metadata. This implementation also has model-accessible recovery copies: B_operational adds them explicitly. Neither the shared F/A dependency nor temporary disk backing is free. This stage makes no equal-byte method superiority claim.',VRAM_scope='Weights, temporary activations, CUDA context and allocator reserve reported separately; no requirement that state ledger equal NVML within5percent.')
 d.write(d.OUT/'call_accounting.json',counts);d.write(d.OUT/'costs.json',costreport);d.write(d.OUT/'history_bytes.json',memory_report)
 figs=d.OUT/'figures';figs.mkdir(exist_ok=True)
 comparisons=summary['comparisons']['revisit_coverage_ge5pct'];labels=['A vs W (primary)','F vs W','F vs A'];keys=['A_vs_W','F_vs_W','F_vs_A'];fig,ax=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[1,1.8]})
 for i,k in enumerate(keys):
  r=comparisons[k]['relative_rotation_deg'];v=r['mean']
  if v is not None and r['interval95'] is not None:
   lo,hi=r['interval95'];ax[0].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4,label=labels[i])
  else:ax[0].text(0,i,'NA: no defined revisit places')
 ax[0].axvline(0,color='.5',lw=1);ax[0].set(yticks=range(3),yticklabels=labels,xlabel='Rotation improvement (degrees)\npositive favors first arm',title='Revisit pairs: paired-place 95% intervals');ax[0].invert_yaxis()
 vals=summary['primary']['per_place'];ax[1].bar(range(12),[np.nan if v is None else v for v in vals],color=['#277a86' if v is not None and v>=0 else '#ae5045' for v in vals]);ax[1].axhline(0,color='.5',lw=1);ax[1].set(xticks=range(12),xticklabels=[m['place'] for m in plan['catalog']],ylabel='E_W − E_A (degrees)',title='Primary effect at every place');ax[1].tick_params(axis='x',rotation=65)
 for i,v in enumerate(vals):
  if v is None:ax[1].text(i,.02,'NA',transform=ax[1].get_xaxis_transform(),ha='center',va='bottom',fontsize=8,color='#7c4e94')
 fig.tight_layout();fig.savefig(figs/'primary.png');plt.close(fig)
 fig,ax=plt.subplots(2,1,figsize=(12,6.7),sharex=True)
 for a,key,title in zip(ax,['coverage','selected_pair_exclusive_fraction'],['Best eligible far-source depth-consistent coverage','Selected far-source support absent from frame0/recent7']):
  arr=np.array([[next(r[key] for r in sel if r['id']==m['id'] and r['t']==t) for t in range(33,96)] for m in plan['catalog']],float);im=a.imshow(arr,aspect='auto',vmin=0,vmax=1,cmap='viridis');a.set(yticks=range(12),yticklabels=[m['place'] for m in plan['catalog']],title=title);fig.colorbar(im,ax=a,label='fraction of valid target pixels')
 ax[-1].set(xticks=[0,15,31,47,62],xticklabels=[33,48,64,80,95],xlabel='Sampled target index (actual seconds retained in selection.json)');fig.tight_layout();fig.savefig(figs/'coverage.png');plt.close(fig)
 fixed=[c for c in allcost if c['id']=='fidelity_00'];fig,ax=plt.subplots(1,2,figsize=(12,4.3))
 for arm,color in [('F','#4971a1'),('A','#c79834'),('W','#278b73')]:ax[0].plot(range(1,97),[c['state'][arm]['inference_history_bytes']/2**30 for c in fixed],label=arm,color=color)
 ax[0].set(xlabel='Completed frame',ylabel='GiB',title='Forward history: tensor + canonical metadata');ax[0].legend()
 for k in ('full_aggregator','window_aggregator','camera_F','camera_A','camera_W','full_depth','window_depth'):ax[1].plot(range(1,97),[c[k]*1000 for c in fixed],label=k.replace('_',' '),lw=1)
 ax[1].set(xlabel='Completed frame',ylabel='ms / module call (synchronized)',title='Fixed P0 stream: actual component times');ax[1].legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(figs/'cost_state.png');plt.close(fig)
 aw=[r for r in counter if r['candidate']=='A' and r['baseline']=='W'];worst=next((r for r in aw if r['revisit']),aw[0]);rid=worst['id'];fig,ax=plt.subplots(1,2,figsize=(12,4.3))
 for arm,color in [('F','#4971a1'),('A','#c79834'),('W','#278b73')]:
  rr=[r for r in pairs if r['id']==rid and r['arm']==arm];ax[0].plot([r['t'] for r in rr],[r['relative_rotation_deg'] for r in rr],label=arm,color=color)
  rr=[r for r in frames if r['id']==rid and r['arm']==arm];ax[1].plot([r['t'] for r in rr],[r['absolute_worldpoint_GT_error_raw_m'] for r in rr],label=arm,color=color)
 ax[0].axvline(worst['t'],color='.5',ls='--');ax[0].set(xlabel='Target index',ylabel='Remote rotation error (degrees)',title=f"Greatest A/W revisit harm\n{rid}, t={worst['t']}");ax[0].legend();ax[1].set(xlabel='Frame index',ylabel='Absolute GT point error (raw m)',title='Absolute GT point error\nfirst-frame rigid alignment; no fitted scale');ax[1].legend();fig.tight_layout();fig.savefig(figs/'counterexample.png');plt.close(fig)
 d.write(d.OUT/'CPU_delivery_runtime.json',dict(complete=True,PID=os.getpid(),PID_namespace='Tool namespace',wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized(),new_model_calls=0,figures=[str(p) for p in sorted(figs.glob('*.png'))]))
 print(json.dumps(dict(counts=counts,GPU_worker_hours=worker_seconds/3600,primary=summary['primary'],figures=str(figs)),indent=2))
if __name__=='__main__':main()
