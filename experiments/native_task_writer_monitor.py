"""Read-only natural-progress estimate; never launches additional numerical work."""
import json,time
from pathlib import Path
from experiments import native_task_writer_data as d

def main():
    queue=d.read(d.OUT/'queue_status.json');active={(x['arm'],x['seed']) for x in queue['active']};updates=0;rows=[];rates=[]
    for folder in sorted((d.OUT/'fits').iterdir()):
        if not folder.is_dir():continue
        logs=[]
        for line in (folder/'training.jsonl').read_text().splitlines() if (folder/'training.jsonl').exists() else []:
            try:logs.append(json.loads(line))
            except json.JSONDecodeError:pass
        if not logs:continue
        steps=logs[-1]['step'];updates+=steps;segment=[]
        for r in logs:
            if segment and r['elapsed_seconds']<segment[-1]['elapsed_seconds']:segment=[]
            segment.append(r)
        recent=segment[-101:];slope=(recent[-1]['elapsed_seconds']-recent[0]['elapsed_seconds'])/(recent[-1]['step']-recent[0]['step']) if len(recent)>10 else None
        arm,seed=folder.name.rsplit('_',1);isactive=(arm,int(seed)) in active
        rows.append(dict(fit=folder.name,step=steps,current_segment_samples=len(segment),recent_natural_seconds_per_update=slope,active=isactive))
        if isactive and slope:rates.append(slope)
    seconds=sum(d.read(p).get('elapsed_seconds',0) for p in (d.OUT/'workers').glob('*.json'));slope=sum(rates)/len(rates) if rates else None
    forecast=seconds+(36000-updates)*slope+1200 if slope else None
    row=dict(utc=d.now(),retained_updates=updates,remaining_updates=36000-updates,actual_worker_hours=seconds/3600,current_active=len(active),active_recent_average_seconds_per_update=slope,projected_worker_hours_with1200s_loading_evaluation_cost_reserve=forecast/3600 if forecast else None,forecast_scope='Natural recent log slopes include checkpoints and shared-host contention; early mixed tasks, no benchmark or controlled speedup claim',fits=rows)
    d.write(d.OUT/'natural_progress.json',row);d.journal(d.OUT/'natural_progress.jsonl',row);print(json.dumps(row),flush=True)
if __name__=='__main__':main()
