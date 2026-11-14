"""Read-only host occupancy snapshot and current-round observation journal."""
import json,datetime,subprocess
from pathlib import Path
from experiments.native_source_resolved_pool import gpus
OUT=Path('/home/liwt/GRoPE/results/gaussian_memory/native_source_resolved_pe1')
def main():
    q=json.loads((OUT/'queue_status.json').read_text());rows,apps=gpus();entry=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPUs=[dict(g,PIDs=sorted(apps.get(g['uuid'],set()))) for g in rows],active=[dict(PID=w['pid'],GPU=w['GPU_index'],job=w['job']) for w in q['active']],GPU_worker_hours=q['GPU_worker_seconds']/3600,DATA_bytes=q['DATA_bytes'],RAM_available_bytes=q['RAM_available_bytes'],manager=q['manager_pid'])
    with (OUT/'resource_observations.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
    progress={}
    for p in sorted((OUT/'calls').glob('fits*')):
        e=[json.loads(l) for l in p.read_text().splitlines()];u=[x for x in e if x['kind']=='formal_update'];v=[x for x in e if x['kind']=='evaluation_query'];last=u[-1] if u else v[-1] if v else None
        if last:
            key=f"{last['arm']}/{last['seed']}";r=progress.setdefault(key,dict(step=0,evaluation_calls=0));r['step']=max(r['step'],u[-1]['step'] if u else 0);r['evaluation_calls']+=len(v)
    compact=[dict(index=g['index'],used_MiB=g['used_MiB'],free_MiB=g['free_MiB'],util=g['util'],PIDs=g['PIDs']) for g in entry['GPUs']]
    print(json.dumps(dict(utc=entry['utc'],manager=q['manager_pid'],active=entry['active'],worker_h=round(entry['GPU_worker_hours'],4),failed=len(q['failed']),GPUs=compact,progress=progress)),flush=True)
if __name__=='__main__':main()
