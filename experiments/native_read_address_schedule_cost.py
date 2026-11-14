"""Wait for a spare safe card after all nine fixed fits have been admitted."""
import fcntl,os,sys,time
from experiments import native_read_address_data as d
from experiments.native_task_writer_manage import gpu_rows,compute_pids
def main():
    assert not (d.OUT/'cost.json').exists() and not (d.OUT/'calls/cost.jsonl').exists();last_notice=0.
    while True:
        q=d.read(d.OUT/'queue_status.json');assert not q['failed'] and not d.resource(True)['stop_reasons'];active_cards={r['GPU_index'] for r in q['active']};rows=gpu_rows();apps=compute_pids();selected=None
        if not q['pending'] and all(d.final(a,d.SEEDS[0]).exists() for a in d.ARMS):
            for g in rows:
                if g['index'] in active_cards or apps.get(g['uuid']) or g['used']>=256 or g['util']>2 or g['free']<42*1024:continue
                completed=[d.read(p) for p in (d.OUT/'workers').glob('*.json') if d.read(p).get('gpu_uuid')==g['uuid']]
                completed.sort(key=lambda x:x['started_utc']);previous=completed[-1] if completed else None
                own_clear=bool(previous and previous.get('actual_process_wait_returned') and previous['exit_code']==0 and not previous.get('foreign_seen_during_worker'))
                stable=q['stable_idle_seconds'].get(g['uuid'],0)>=60
                if own_clear or stable:selected=(g,previous,own_clear,stable);break
        if selected:
            g,previous,own_clear,stable=selected;d.write(d.OUT/'cost_dispatch.json',dict(utc=d.now(),GPU=g,prior_own_worker=previous,own_clean_release_live_verified=own_clear,stable_idle60=stable,formal_active=[dict(arm=r['arm'],seed=r['seed'],GPU=r['GPU_index'],pid=r['pid']) for r in q['active']],all_formal_fits_already_admitted=True,fixed_checkpoint_seed=d.SEEDS[0],prescribed_only=dict(full_native=70,query_encoder=70,actual_raster=60,source_Cmu=2),science_sha256=d.SCIENCE_SHA));print(dict(cost_dispatch_GPU=g['index'],prior_own_worker=previous['name'] if previous else None,formal_active=len(q['active'])),flush=True);os.execv(sys.executable,[sys.executable,'-u','-m','experiments.native_read_address_auxlaunch',str(g['index'])])
        if time.monotonic()-last_notice>=60:print(dict(cost_waiting_for_spare_safe_card=True,formal_pending=q['pending'],formal_active=len(q['active']),no_model_calls_yet=True),flush=True);last_notice=time.monotonic()
        time.sleep(10)
if __name__=='__main__':
    with (d.OUT/'claims/cost_dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
