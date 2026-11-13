"""Read-only compact monitor; no model imports or state mutation."""
import json
import time
from pathlib import Path
OUT=Path('/home/liwt/GRoPE/results/gaussian_memory/native_memory_context_pe1')
def read(p):return json.loads(p.read_text())
def main():
    q=read(OUT/'queue_status.json');b=read(OUT/'budget.json');jobs={}
    for seed in [20261110,20261111,20261112]:
        for kind in ['P','S','G']:
            name=f'{kind}_{seed}';folder=OUT/name
            if (folder/'status.json').exists():jobs[name]=2000
            elif (folder/'progress.json').exists():jobs[name]=read(folder/'progress.json')['step']
    records=[read(p) for p in (OUT/'runtime').glob('*.json')];hours=sum(r.get('seconds',time.time()-r['start_epoch']) for r in records if r.get('runtime_session'))/3600;res=read(OUT/'resources/current.json')
    print(json.dumps(dict(state=q['state'],complete=len(q['complete']),steps=jobs,failed=q['failed'],budget=b,worker_hours=round(hours,6),H_GiB=round(res['H']/2**30,3),data_GiB=round(res['data_bytes']/2**30,3),postqueue=read(OUT/'postqueue/status.json')['state'])))
if __name__=='__main__':main()
