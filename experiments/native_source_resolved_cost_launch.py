"""Dispatch the sole prescribed cost case on a spare card, never a second fit queue."""
import fcntl,os,sys,time,subprocess,signal
from pathlib import Path
from experiments import native_source_resolved_data as d
from experiments.native_source_resolved_pool import gpus

def main():
    signal.signal(signal.SIGTERM,lambda *args:sys.exit(75))
    with (d.OUT/'cost_dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (d.OUT/'actual_cost.json').exists()
        assert not any((d.OUT/'calls').glob('cost_*')),'A cost attempt already exists: inspect before any recovery'
        notice=0;external_last_seen={}
        while True:
            q=d.read(d.OUT/'queue_status.json');assert not q['failed']
            rows,apps=gpus();active={w['GPU_index'] for w in q['active']};ownpids={w['pid'] for w in q['active']};chosen=None
            for g in rows:
                if apps.get(g['uuid'],set())-ownpids:external_last_seen[g['uuid']]=time.time()
            if not q['pending'] and all((d.CACHE/'fits'/f'{k}_{d.SEEDS[0]}'/'final.pt').exists() for k in d.KINDS):
                for g in rows:
                    guard=d.read(d.OUT/'admission_guard_policy.json') if (d.OUT/'admission_guard_policy.json').exists() else {};last=max(guard.get('by_uuid',{}).get(g['uuid'],{}).get('last_foreign_or_busy_seen_unix',0),external_last_seen.get(g['uuid'],0));stable=g['index'] not in guard.get('guarded_GPU_indices',[]) or not guard.get('enabled') or time.time()-last>=guard.get('stable_idle_seconds',120)
                    if stable and g['index'] not in active and not apps.get(g['uuid']) and g['used_MiB']<=256 and g['free_MiB']>=24000 and g['util']<=2:chosen=g;break
            if chosen:break
            if time.monotonic()-notice>=60:print(dict(cost_waiting_only_for_spare_safe_card=True,formal_active=len(q['active']),extra_model_calls=0),flush=True);notice=time.monotonic()
            time.sleep(5)
        assert q['GPU_worker_seconds']+180<18*3600
        assert int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024>=20*2**30
        d.write(d.OUT/'cost_dispatch.json',dict(utc=d.now(),GPU=chosen,all_fixed_fits_already_admitted=True,fixed_checkpoint_seed=d.SEEDS[0],formal_active_on_other_cards=q['active'],science_sha256=d.SCIENCE_SHA,one_prescribed_case_only=True,same_fit_queue_unchanged=True))
        name=f'cost_{time.time_ns()}';logpath=d.OUT/'logs'/f'{name}.log';env=dict(os.environ,CUDA_VISIBLE_DEVICES=chosen['uuid'],MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PATH='/home/liwt/miniColmap/.venv/bin:/usr/local/cuda-12.4/bin:'+os.environ['PATH'],CUDA_HOME='/usr/local/cuda-12.4',TORCH_CUDA_ARCH_LIST='8.6',TORCH_EXTENSIONS_DIR=str(d.old.DATA/'torch_extensions'),MAX_JOBS='2')
        with logpath.open('a') as log:
            start=time.monotonic();child=subprocess.Popen([sys.executable,'-u','-m','experiments.native_source_resolved_cost','--name',name],cwd=d.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            record=dict(name=name,phase='cost',job=0,pid=child.pid,GPU_index=chosen['index'],gpu_uuid=chosen['uuid'],started_utc=d.now(),elapsed_seconds=0.,exit_code=None,admission=chosen,log=str(logpath),one_prescribed_case_only=True)
            print(dict(prescribed_cost_started=True,PID=child.pid,GPU=chosen['index'],fixed_case='fit_00/a0/q2',formal_active=len(q['active'])),flush=True)
            try:
                while child.poll() is None:
                    record.update(elapsed_seconds=time.monotonic()-start);d.write(d.OUT/'workers'/f'{name}.json',record)
                    rows,apps=gpus();g=next(x for x in rows if x['uuid']==chosen['uuid']);foreign=apps.get(chosen['uuid'],set())-{child.pid}
                    if (foreign or g['free_MiB']<6144) and not record.get('stop_reason'):record['stop_reason']='external_PID_or_memory_pressure';child.send_signal(signal.SIGTERM)
                    time.sleep(1)
            finally:
                if child.poll() is None:child.send_signal(signal.SIGTERM)
                code=child.wait();record.update(elapsed_seconds=time.monotonic()-start,exit_code=code,actual_wait_returned=True,ended_utc=d.now());d.write(d.OUT/'workers'/f'{name}.json',record)
            assert code==0 and (d.OUT/'actual_cost.json').exists(),record
            print(dict(cost_child_actual_exit=code,worker_seconds=record['elapsed_seconds']),flush=True)
if __name__=='__main__':main()
