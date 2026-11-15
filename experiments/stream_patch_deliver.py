"""Round038 exact costs/calls before efficacy, then exports and figures."""
import os,time,json,csv,collections,resource,argparse
import numpy as np
import torch
from experiments import stream_patch_data as d

def events():return [dict(worker=p.stem,**json.loads(line)) for p in (d.OUT/'calls').glob('*.jsonl') for line in p.read_text().splitlines()]
def preflight():
 begin=time.monotonic();checks={};old=d.load(d.OLDDATA/'predictions/fidelity_00.pt')
 for arm in ('W','U','C'):
  p=d.load(d.DATA/'predictions'/f'{arm}_fidelity_00.pt');n=96 if arm=='W' else 7;checks[arm]=dict(frames=n,depth_max_abs=float((p['depth'][:n]-old['depth_W'][:n]).abs().max()),pose_max_abs=float((p['poses'][:n]-old['poses']['W'][:n]).abs().max()),original_arrays_only=True,old_pixel_metrics_recomputed=0)
  if arm!='W':
   for t,ids in enumerate(p['retained_IDs']):
    ids=ids.long();assert len(ids)==len(ids.unique()) and bool((ids.sort().values==ids).all());f=ids//1041;patch=ids%1041>=5;near=torch.isin(f,torch.tensor([0]+list(range(max(1,t-2),t+1))));special=torch.isin(f,torch.tensor([0]+list(range(max(1,t-6),t+1))));assert int((patch&near).sum())==min(t+1,4)*1036;assert int((~patch&special).sum())==min(t+1,8)*5;assert int((patch&~near).sum())<=d.plan()['M'];assert int((~patch&~special).sum())==0
 es=events();restore=[e for e in es if e['kind']=='exact_restore'];assert len(restore)>=1 and restore[0]['next_frame']==48
 d.write(d.OUT/'preflight.json',dict(complete=True,registered_utc=d.now(),W_once96_U_C_first7=checks,exact_restore=restore,raw24layer_sourceID_field_common_selection=True,retained_sourceIDs_all_frames_checked=True,FP64_GPU_FPS_toy_check=[e for e in es if e['kind']=='deterministic_FPS_interface'],CPU_seconds=time.monotonic()-begin,new_model_calls=0,old_pixel_metrics_recomputed=0,first_U_C_outputs_reused=True,scope='Finalcompletedarchivedprefixarraysused;splitsegment-localdraftreplaysummariesarenotthesourceoftruth. Nofloatingtolerancegateornewmodelreplay.'))
 print(dict(preflight=checks),flush=True)

def costs():
 es=events();p=d.plan();ws=[d.read(v) for v in (d.OUT/'workers').glob('*.json')];assert all(w['actual_wait_returned'] for w in ws);nf=[e for e in es if e['kind']=='native_frame'];ids=[(e['policy'],e['id'],e['frame']) for e in nf];assert len(ids)==len(set(ids))==2400
 per=[]
 for arm,rid in [('W','fidelity_00')]+[(a,m['id']) for m in p['catalog'] for a in ('U','C')]:
  rr=[e for e in nf if e['policy']==arm and e['id']==rid];assert sorted(r['frame'] for r in rr)==list(range(96));per.append(dict(policy=arm,id=rid,components_seconds={k:sum(r[k] for r in rr) for k in ('aggregator_seconds','camera_depth_seconds','center_selection_eviction_seconds','RGB_read_seconds','frame_wall_seconds')},max_measured_retained_history_bytes=max(r['memory']['total_bytes'] for r in rr),actual_cold_process_worker_seconds=sum(w['elapsed_seconds'] for w in ws if w['policy']==arm and w['id']==rid),invalid_candidates=sum(r['selection'].get('invalid_candidates',0) for r in rr),invalid_fill=sum(r['selection'].get('invalid_fill',0) for r in rr)))
 pools=[d.read(v) for v in d.OUT.glob('pool_*_complete.json')];res=[d.read(v) for v in (d.OUT/'worker_resources').glob('*.json')];seconds=sum(w['elapsed_seconds'] for w in ws);profiles=sum(w['elapsed_seconds'] for w in ws if w['phase']=='profile');predbytes=sum(d.read(v)['bytes'] for v in (d.OUT/'completed').glob('*.json'));maxpred=max(d.read(v)['bytes'] for v in (d.OUT/'completed').glob('*.json'));extension=d.size(d.DATA/'extensions');# Conservativelyreserveall8workerssafeexit/finalatomics,includingalreadycompletedoutputarchives.
 bound=predbytes+8*(p['state_copy_bound_bytes']+3*maxpred+2**20)+extension
 counts=dict(formal_U_C_native_frames=2304,W_fixed_complete_cost_replay=96,actual_aggregator_calls=2400,actual_camera_head_calls=2400,actual_depth_head_calls=2400,camera_refinement_rounds=9600,model_loads=sum(e['kind']=='model_loaded' for e in es),repeated_completed_policy_stream_frames=0,FPS_interface_arithmetic_calls=sum(e['kind']=='deterministic_FPS_interface' for e in es),training=0,downloads=0,new_label_projection_calls=0,old_W_pixel_metrics_recomputed=0)
 c=dict(complete=True,actual_all_waited=True,all_GPU_worker_seconds=seconds,all_GPU_worker_hours=seconds/3600,preflight_cost_seconds=profiles,workers=ws,per_stream=per,actual_worker_processes=len(ws),model_load_body_seconds=sum(e['seconds'] for e in es if e['kind']=='model_loaded'),checkpoint_IO_seconds=sum(e['seconds'] for e in es if e['kind']=='complete_exit_checkpoint'),restores=[e for e in es if e['kind']=='exact_restore'],peak_NVML_worker_MiB=max(v['peak_NVML_worker_MiB'] for v in pools),peak_GPU_allocated_bytes=max(v['peak_GPU_allocated_bytes'] for v in res),peak_GPU_reserved_bytes=max(v['peak_GPU_reserved_bytes'] for v in res),peak_RSS_worker_bytes=max(v['peak_RSS_bytes'] for v in res),DATA_peak_sampled_bytes=max(v['DATA_peak_sampled_bytes'] for v in pools),DATA_conservative_bound_bytes=bound,DATA_final_bytes=d.size(d.DATA),permanent_output_bytes=predbytes,extension_build_backing_bytes=extension,project_current_bytes=d.size(d.OUT),ordinary_GPU_failures=[w for w in ws if w['exit_code'] not in (0,75)],cooperative_exits=[w for w in ws if w['exit_code']==75],fixed_first_case=[r for r in per if r['id']=='fidelity_00'],kernel_IO_counters=res,scope='AllPopen→actualwaitincludesimports/compilation/model-load/FPS/allnativequeries/writes/restore/I-O/failures. Nativefullqueryincludesencoder24layers/camera4round/depth;notcachedattention-onlylatency. Managerwaitingseparate. Currentactivations/weights/CUDAreservearenotretainedhistory;allrawhistorycopiesandRGBcountedB. CPUmetricsnoGPUmodelcalls.')
 assert seconds<14400 and profiles<1800 and bound<12*2**30 and c['peak_NVML_worker_MiB']<=40960
 formal_begin=collections.Counter(e['module'] for e in es if e['kind']=='module_begin');formal_end=collections.Counter(e['module'] for e in es if e['kind']=='module_complete');assert formal_begin==formal_end==dict(aggregator=2112,camera_head=2112,depth_head=2112)
 counts.update(formal_module_begin_receipts=dict(formal_begin),formal_module_complete_receipts=dict(formal_end),profile_complete_native_frames_without_separate_hooks=288,actual_policy_FPS_calls=sum(e['policy']=='C' and e['selection'].get('candidates',0)>p['M'] for e in nf),actual_policy_hash_calls=sum(e['policy']=='U' and e['selection'].get('candidates',0)>p['M'] for e in nf),finite_native_frames=sum(e['depth_finite'] and e['pose_finite'] for e in nf))
 Uhash=[e for e in nf if e['policy']=='U' and e['selection']['candidates']>p['M']];Cfill=[e for e in nf if e['policy']=='C' and e['selection'].get('invalid_fill',0)>0];provenance=sum(e['selection']['retained_tokens'] for e in nf if e['policy']!='W')*8
 c['transfer_ledger']=dict(current_full_FP32_RGB_H2D_bytes=2400*3*392*518*4,current_native_FP32_depth_pose_D2H_bytes=2400*(392*518+9)*4,retained_ID_provenance_GPU_int64_to_CPU_bytes=provenance,U_hash_eligible_ID_D2H_bytes=sum(e['selection']['candidates']*8 for e in Uhash),U_hash_selected_index_H2D_bytes=len(Uhash)*p['M']*8,C_invalid_fill_ID_D2H_bytes=sum((e['selection']['candidates']-e['selection']['valid_candidates'])*8 for e in Cfill),C_invalid_fill_selected_index_H2D_bytes=sum(e['selection']['invalid_fill']*8 for e in Cfill),checkpoint_state_file_bytes=[e['state_file_bytes'] for e in es if e['kind']=='complete_exit_checkpoint'],scope='Logicalmandatoryactualtensorcopiesfromnativeframecounts/retainedshapesandcode,directlyreconstructable;notPCIebandwidthprofiling. CompleteGPU↔CPUstatecopyandrestoreareboundedbylistedstatefilepayload+8MiBmetadataallowance;RNG/scalar/shortindexcopiesnotindividuallyinstrumented.Alltransfertimeincludedinworkerwall/completeframeenvelopes. NativeD2Dcatandindex_selectarechargedintheaggregator/updateenvelopes;notzero-costorextraunmeasuredhistorybacking.')
 d.write(d.OUT/'call_accounting.json',counts);d.write(d.OUT/'costs.json',c);d.write(d.OUT/'history_bytes.json',dict(capacity=d.read(d.OUT/'byte_capacity.json'),per_stream_measured=per,checkpoint_records=[d.read(v) for v in (d.OUT/'checkpoint_records').glob('*.json')],policy_scope='SameU/Chistorypayload/copies/allowance;selectedIDs/centers/RNG/metadataretained. NoFullteacher/Wshadow/othergrouporthrowawayKVbank. Evictedrawstoragefreed;archiveoutputs/selectionprovenancenotforwardinputs. Source2DpositionsderivedlosslesslyfromretainedID;all24samepatchselected,unchangedrawKVmass1.',cost_scope='Read=currentRGB+originalencoder24layers/camera4round/depth. Formation=currentnativeK/V+ownpredictedcenters;update=ownsourcepreservingselection/eviction. AllCPU-GPU/DATAcopiesinworkerwall;explicitstatecheckpointbytes, measuredGPU/RSS. Nozero-costcache/teacherclaim.'))
 print(dict(GPU_worker_hours=seconds/3600,DATA_bound_GiB=bound/2**30,calls=counts),flush=True)

def figures():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150});p=d.plan();s=d.read(d.OUT/'scoring/summary.json');counter=d.read(d.OUT/'scoring/all_counterexamples.json');pairs=d.read(d.OUT/'scoring/pairs.json');frames=d.read(d.OUT/'scoring/frames.json');figs=d.OUT/'figures';figs.mkdir(exist_ok=True);keys=['C_vs_U','C_vs_W','U_vs_W'];colors=['#207c80','#6970a5','#c1902b']
 for name in ('frames','pairs','selection','by_place','support_strata','all_counterexamples'):
  rows=d.read(d.OUT/'scoring'/f'{name}.json');fields=list(dict.fromkeys(k for r in rows for k in r))
  with (d.OUT/'scoring'/f'{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 fig,ax=plt.subplots(1,2,figsize=(12,4.8))
 for i,k in enumerate(keys):
  z=s['comparisons']['revisit_coverage_ge5pct'][k]['relative_rotation_deg'];v=z['mean'];lo,hi=z['interval95'];ax[0].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4,color=colors[i]);vals=z['per_place'];ax[1].bar(np.arange(12)+(i-1)*.23,[np.nan if v is None else v for v in vals],width=.22,label=k,color=colors[i])
  if i==0:
   for j,v in enumerate(vals):
    if v is None:ax[1].text(j,0,'NA',ha='center',fontsize=7)
 ax[0].axvline(0,color='.5');ax[0].set(yticks=range(3),yticklabels=['C/U (primary)','C/W (net utility)','U/W (net utility)'],xlabel='Rotation improvement (degrees)',title='Same actual byte ceiling / 189 fixed revisits');ax[0].invert_yaxis();ax[1].axhline(0,color='.5');ax[1].set(xticks=range(12),xticklabels=[m['place'] for m in p['catalog']],ylabel='Paired error improvement (degrees)',title='Own causal online feedback, all12places');ax[1].tick_params(axis='x',rotation=65);ax[1].legend(fontsize=8);fig.tight_layout();fig.savefig(figs/'primary.png');plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(11,4.4))
 for i,k in enumerate(keys):
  z=s['comparisons']['all96frames'][k]['absrel'];v=z['mean'];lo,hi=z['interval95'];ax[0].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4,color=colors[i])
  for j,metric in enumerate(['log_mse','centered_log_mse','bias_squared']):
   z=s['comparisons']['all96frames'][k][metric];v=z['mean'];lo,hi=z['interval95'];ax[1].errorbar(v,j+(i-1)*.15,xerr=[[v-lo],[hi-v]],fmt='o',capsize=3,label=k if j==0 else None,color=colors[i])
 ax[0].axvline(0,color='.5');ax[0].set(yticks=range(3),yticklabels=keys,xlabel='AbsRel improvement',title='All96frames, raw depth cost');ax[0].invert_yaxis();ax[1].axvline(0,color='.5');ax[1].set(yticks=range(3),yticklabels=['raw log-MSE','centered log-MSE','bias squared'],xlabel='Error improvement',title='No fitted GT scale');ax[1].invert_yaxis();ax[1].legend(fontsize=8);fig.tight_layout();fig.savefig(figs/'depth.png');plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(11,4.4))
 for a,arm in zip(ax,('U','C')):
  v=d.load(d.DATA/'predictions'/f'{arm}_fidelity_00.pt');arr=np.zeros((96,96))
  for t,ids in enumerate(v['retained_IDs']):
   src=(ids.long()//1041)[ids.long()%1041>=5];arr[t]=np.bincount(src.numpy(),minlength=96)/1036
  im=a.imshow(arr.T,origin='lower',aspect='auto',vmin=0,vmax=1,cmap='viridis');a.set(xlabel='Completed current frame',ylabel='Retained patch source frame',title=f'{arm}: exact own-source patch fraction, M={p["M"]}')
 fig.subplots_adjust(left=.07,right=.83,wspace=.25);cax=fig.add_axes([.87,.15,.02,.7]);fig.colorbar(im,cax=cax,label='Fraction of source patches retained');fig.savefig(figs/'retention.png');plt.close(fig)
 worst=next(r for r in counter if r['candidate']=='C' and r['baseline']=='U' and r['revisit']);rid=worst['id'];fig,ax=plt.subplots(1,2,figsize=(12,4.3))
 for arm,col in [('U','#c1902b'),('C','#207c80'),('W','#6970a5')]:
  rr=[r for r in pairs if r['id']==rid and r['arm']==arm];ff=[r for r in frames if r['id']==rid and r['arm']==arm];ax[0].plot([r['t'] for r in rr],[r['relative_rotation_deg'] for r in rr],label=arm,color=col);ax[1].plot([r['t'] for r in ff],[r['absolute_worldpoint_GT_error_raw_m'] for r in ff],label=arm,color=col)
 ax[0].axvline(worst['t'],ls='--',color='.5');ax[0].set(xlabel='Target frame',ylabel='Far-pair rotation error (degrees)',title=f'Largest C/U revisit harm: {rid}, t={worst["t"]}');ax[1].set(xlabel='Current frame',ylabel='Absolute GT point error (raw m)',title='Online geometry, no fitted scale');ax[0].legend();ax[1].legend();fig.tight_layout();fig.savefig(figs/'counterexample.png');plt.close(fig)
 print(dict(figures_complete=True),flush=True)

def main():
 a=argparse.ArgumentParser();a.add_argument('phase',choices=['preflight','costs','figures']);v=a.parse_args();begin=time.monotonic();torch.set_num_threads(2);globals()[v.phase]();d.write(d.OUT/f'CPU_{v.phase}_runtime.json',dict(complete=True,PID=os.getpid(),seconds=time.monotonic()-begin,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized(),new_model_calls=0))
if __name__=='__main__':main()
