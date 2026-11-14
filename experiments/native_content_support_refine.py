"""CPU figure de-duplication/coverage terminology and fixed I/O-only calibration."""
from experiments import native_content_support_score as s
from experiments import native_content_support as d
import time,os,hashlib,resource
start=time.monotonic();cfg=d.config();original=s.original_coverage(cfg);tokens=d.read(d.OUT/'token_measurements.json');selection=d.read(d.OUT/'figure_selection.json');selected=selection['selected'][:3]
zero={x['key'] for x in original['images'] if x['J_whole_grid_zero_threshold_reference']}
candidates=[r for r in tokens if r['cohort']=='query' and not r['read_defined'] and r['key'] in zero and r['key'] not in {x['key'] for x in selected}];assert candidates
selected.append(dict(candidates[0],label='Original zero-grid reference / undefined read'));selection['selected']=selected;selection['rule']='Primary largest negative/positive, fixedfirstquerytoken, first distinct priorJwhole-grid-zero reference with current undefined selected read. All outcomes/thresholds unchanged.';d.write(d.OUT/'figure_selection.json',selection)
s.figures(d.read(d.OUT/'summary.json')['cohorts'],d.read(d.OUT/'place_measurements.json'),selected)
records=[]
for cohort in ['query','source']:
    case=next(c for c in cfg['images'] if c['cohort']==cohort);p=d.OUT/'images'/case['key']/'energy.npy';t=time.monotonic();data=p.read_bytes();read_s=time.monotonic()-t
    for label,buf in [('full_energy_file',data),('one_probe_8192_bytes',data[-8192:])]:
        tmp=d.OUT/f'io_calibration_{cohort}.tmp';t=time.monotonic()
        with tmp.open('wb') as f:f.write(buf);f.flush();os.fsync(f.fileno())
        write_s=time.monotonic()-t;t=time.monotonic();back=tmp.read_bytes();reload_s=time.monotonic()-t;assert hashlib.sha256(back).digest()==hashlib.sha256(buf).digest();tmp.unlink();records.append(dict(cohort=cohort,case=case['key'],payload=label,bytes=len(buf),actual_write_fsync_seconds=write_s,logical_reload_seconds=reload_s,initial_full_energy_read_seconds=read_s))
d.write(d.OUT/'io_cost_calibration.json',dict(records=records,scope='One fixedfirstquery and firstsource, each fullnewenergyfile and one8192byte probe write/fsync/logicalreload; CPU-only separate calibration, not original concurrentGPUwrite latency. No hardware-cold cache or exclusivity claim; no repeatedmodel/gradient/render. Original VJP timer includes GPUbackward,FP64reduction,CPUtransfer,energyflush andfsync and is never presented as pureGPUbackward time.'))
d.write(d.OUT/'CPU_refinement_runtime.json',dict(complete=True,body_wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=d.torch.cuda.is_initialized(),model_VJP_render_calls=0,statistics_recomputed=False))
print({'figures_rerendered_without_statistical_recomputation':True,'original_query_zero_images':[x['key'] for x in original['images'] if x['cohort']=='query' and x['J_whole_grid_zero_threshold_reference']],'original_source_zero_images':[x['key'] for x in original['images'] if x['cohort']=='source' and x['J_whole_grid_zero_threshold_reference']]},flush=True)
