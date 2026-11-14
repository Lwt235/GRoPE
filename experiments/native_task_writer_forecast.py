"""Resource-aware natural forecast; no benchmark, model call or admission override."""
import datetime,json,os,time
from experiments import native_task_writer_data as d
from experiments.native_task_writer_manage import compute_pids
from experiments.native_task_writer_monitor import main as monitor

def main():
    monitor();r=d.read(d.OUT/'natural_progress.json');q=d.read(d.OUT/'queue_status.json');apps=compute_pids();own={x['pid'] for x in q['active']};foreign={uuid:sorted(pids-own) for uuid,pids in apps.items() if pids-own};lookup={x['fit']:x for x in r['fits']};active=[];safe=[]
    for worker in q['active']:
        name=f"{worker['arm']}_{worker['seed']}";row=lookup.get(name,dict(step=0,recent_natural_seconds_per_update=None));rate=row['recent_natural_seconds_per_update'];isforeign=worker['gpu_uuid'] in foreign
        if rate is not None and not isforeign:safe.append(rate)
        active.append(dict(fit=name,remaining_updates=2000-row['step'],rate=rate,foreign_shared=isforeign,gpu=worker['GPU_index']))
    pending_updates=r['remaining_updates']-sum(x['remaining_updates'] for x in active);safe_rate=sum(safe)/len(safe) if safe else None;post=[]
    for p in (d.OUT/'workers').glob('*.json'):
        w=d.read(p)
        if w.get('exit_code')==0 and w.get('arm') in d.ARMS:
            f=d.OUT/'fits'/f"{w['arm']}_{w['seed']}"/'complete.json';post.append(datetime.datetime.fromisoformat(w['ended_utc']).timestamp()-f.stat().st_mtime)
    remaining_fits=18-len(q['completed']);post_mean=sum(post)/len(post) if post else 180.;reserve=remaining_fits*max(180.,post_mean)+max(0,len(q['pending']))*30+180
    forecast=None
    if safe_rate is not None:
        active_seconds=sum(x['remaining_updates']*(x['rate'] or safe_rate) for x in active);forecast=r['actual_worker_hours']+(active_seconds+pending_updates*safe_rate+reserve)/3600
    row=dict(utc=d.now(),retained_updates=r['retained_updates'],actual_worker_hours=r['actual_worker_hours'],active=len(active),uniform_extrapolation_hours=r['projected_worker_hours_with1200s_loading_evaluation_cost_reserve'],resource_aware_forecast_hours=forecast,pending_updates=pending_updates,admissible_current_cards_mean_seconds_per_update=safe_rate,remaining_loading_evaluation_cost_reserve_seconds=reserve,observed_postfinal_worker_seconds=post,active_remaining=active,foreign_PID_observations=foreign,CPU_cores=os.cpu_count(),CPU_loadavg=list(os.getloadavg()),scope='Existing foreign-shared active work uses its actual slower rate; no future fit is admitted on a card still containing foreign PIDs. Future pending work uses currently admissible-card rates. Prospective estimate, not hard guarantee or extra budget; original18h runtime protection unchanged.')
    d.write(d.OUT/'resource_aware_forecast.json',row);d.journal(d.OUT/'resource_aware_forecast.jsonl',row);print(json.dumps(row),flush=True)
if __name__=='__main__':main()
