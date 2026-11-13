"""Resource-only allocated-output ledger and mutually watched fail-closed guard."""
import argparse
from collections import deque
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/home/liwt/GRoPE')
OUT=ROOT/'results/gaussian_memory/native_render_parameter_pe1'
ADD=ROOT/'results/gaussian_memory/autonomous_research_loop/round_002/resource_addendum_v1.json'
CACHE=Path('/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/gaussian_memory/native_render_parameter_pe_v1')
RAM=Path('/dev/shm/liwt_native_render_parameter_pe1_1020')
FOLDER=OUT/'resource_sidecar'
GIB=2**30
ADD_SHA='909268727e0139b6726860142218e6d5f494605b4569c613e5ab9b7198f13d20'
STOP=False


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def free(p):
    s=os.statvfs(p);return s.f_bavail*s.f_frsize

def identity(pid):
    p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text();tail=stat[stat.rfind(')')+2:].split();cmd=(p/'cmdline').read_bytes()
        return dict(pid=pid,uid=p.stat().st_uid,starttime=int(tail[19]),state=tail[0],cmdline=cmd.decode().replace('\0',' '),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError,ProcessLookupError):return None

def same(saved,required=None):
    current=identity(saved['pid'])
    if current is None or current['state']=='Z':return None
    assert all(current[k]==saved[k] for k in ['uid','starttime','boot_id']),('PID identity changed',saved,current)
    assert current['uid']==os.getuid()
    # Linux can expose an empty cmdline briefly after Python has torn down its
    # signal handling but before the final zombie state becomes visible.
    if not current['cmdline']:return None
    assert current['cmdline']==saved['cmdline'],('Live command changed',saved,current)
    if required:assert required in current['cmdline']
    return current

def process_remaining(saved):
    """Wait for real exit even when cmdline is already empty; never signal here."""
    current=identity(saved['pid'])
    return current is not None and current['state']!='Z' and all(current[k]==saved[k] for k in ['uid','starttime','boot_id'])

def queue_current(saved):
    current=same(saved)
    if current is not None:
        assert any(module in current['cmdline'] for module in ['-m experiments.native_render_parameter_queue','-m experiments.native_render_parameter_resume_controller'])
    return current

def runtime():
    records=[]
    for p in (OUT/'runtime').glob('*.json'):
        r=read(p)
        if r.get('runtime_session'):records.append((p,r))
    return records

def active_workers():
    found=[]
    for p,r in runtime():
        if r.get('process_exited'):continue
        if 'pid' not in r:continue  # launch's atomic first record precedes Popen
        ident=identity(r['pid'])
        if ident is None or ident['state']=='Z' or not ident['cmdline']:continue
        assert ident['uid']==os.getuid() and r['module'] in ident['cmdline']
        if 'parent_pid' in r:
            try:parent=Path('/proc',str(r['pid']),'stat').read_text().split()[3]
            except FileNotFoundError:continue
            assert parent==str(r['parent_pid'])
        found.append(dict(identity=ident,runtime_path=str(p),record=r))
    return found

def validate_original(q):
    assert sha(ADD)==ADD_SHA
    a=read(ADD)
    for name in ['parent_science_plan','parent_implementation_plan','parent_configuration_lock']:
        r=a[name];assert sha(r['path'])==r['sha256']
    lock=read(OUT/'configuration_lock.json')
    assert all(sha(ROOT/n)==h for n,h in lock['source_sha256'].items())
    assert queue_current(q)['state']=='T'
    with (OUT/'queue.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise RuntimeError('Original queue no longer holds queue.lock')
    assert not active_workers()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            if proc.stat().st_uid==os.getuid() and b'experiments.native_render_parameter_train' in (proc/'cmdline').read_bytes().split(b'\0'):
                assert identity(int(proc.name))['state']=='Z',('Unaccounted existing trainer',proc.name)
        except (FileNotFoundError,ProcessLookupError):pass
    return a

def file_stat(p):
    try:
        s=Path(p).stat();return dict(path=str(p),allocated=s.st_blocks*512,logical=s.st_size,inode=[s.st_dev,s.st_ino])
    except FileNotFoundError:return None

def marker_for(item):
    if item['kind']=='final_weight':return OUT/f"{item['arm']}_{item['seed']}"/'status.json'
    return OUT/'evaluation'/f"{item['cohort']}_{item['arm']}_{item['seed']}.json"

def ledger(a,needs):
    rows=[];remaining=0;transient=0;known=set();transient_seen=set()
    for item in a['baseline_artifact_accounting']['remaining_artifacts']:
        target=Path(item['path']);tmp=target.with_suffix('.tmp');known.update([str(target),str(tmp)])
        stats=[s for s in [file_stat(target),file_stat(tmp)] if s]
        unique={tuple(s['inode']):s for s in stats}
        assert len(unique)<=1,('Duplicate output publication',str(target),stats)
        expected=max(needs.get(str(target),item['expected_bytes']),*(s['logical'] for s in stats),item['expected_bytes']);needs[str(target)]=expected
        marker=marker_for(item);verified=False;hash_value=None
        if marker.exists():
            m=read(marker);verified=m.get('complete',False)
            if verified:
                s=file_stat(target);assert s is not None
                hash_value=m['checkpoint_sha256' if item['kind']=='final_weight' else 'prediction_sha256'];assert len(hash_value)==64
                assert m['checkpoint_path' if item['kind']=='final_weight' else 'prediction_path']==str(target)
                if item['kind']=='final_weight':assert m['bytes']==s['logical']
                expected=max(expected,s['logical']);needs[str(target)]=expected
        credit=0 if verified else min(expected,max([s['allocated'] for s in unique.values()]+[0]));need=0 if verified else max(0,expected-credit);remaining+=need
        if item['kind']=='predictions':
            partial=target.with_name(target.stem+'_partial.pt')
            for p in [partial,partial.with_suffix('.tmp')]:
                known.add(str(p));s=file_stat(p)
                if s and tuple(s['inode']) not in transient_seen:transient+=s['allocated'];transient_seen.add(tuple(s['inode']))
        rows.append(dict(path=str(target),kind=item['kind'],expected_bytes=expected,verified=verified,verified_sha256=hash_value,publication=stats,permanent_allocated_credit=credit,unallocated_bytes=need))
    owned=[];seen=set()
    for p in CACHE.rglob('*'):
        if not p.is_file():continue
        s=file_stat(p)
        if s is None:continue
        if tuple(s['inode']) not in seen:owned.append(s);seen.add(tuple(s['inode']))
        if p.suffix=='.tmp' or 'partial' in p.name:assert str(p) in known,('Unrecognized temporary file',str(p))
    own=sum(s['allocated'] for s in owned);logical=sum(s['logical'] for s in owned);project=sum(p.stat().st_blocks*512 for p in OUT.rglob('*') if p.is_file())
    t=max(0,.5*GIB-transient);f=free(CACHE);h=f-remaining-t
    worker_seconds=sum(r.get('seconds',time.time()-r['start_epoch']) for _,r in runtime())
    return dict(epoch=time.time(),F=f,R=remaining,T_remaining=t,H=h,transient_allocated=transient,own_allocated=own,own_logical=logical,own_peak_forecast=own+remaining+t,project_allocated=project,project_free=free(OUT),project_completion_headroom=free(OUT)-max(0,2*GIB-project),tmpfs_free=free(RAM),GPU_worker_seconds=worker_seconds,budget=read(OUT/'budget.json'),artifacts=rows,space_gap_admission=max(0,remaining+t+6.5*GIB-f))

def reasons(r,admission=False):
    errors=[]
    if r['H']<(6.5 if admission else 6)*GIB:errors.append('forecast_headroom')
    if r['F']<5*GIB:errors.append('actual_data_free')
    if r['transient_allocated']>.5*GIB:errors.append('transient_allowance')
    if max(r['own_allocated'],r['own_peak_forecast'])>26*GIB:errors.append('data_hard_cap')
    if r['project_allocated']>2*GIB or r['project_completion_headroom']<8*GIB:errors.append('project_cap_or_reserve')
    if r['tmpfs_free']<(20 if admission else 10)*GIB:errors.append('tmpfs_reserve')
    if r['GPU_worker_seconds']>=8*3600:errors.append('GPU_worker_hard_cap')
    if r['budget']['formal_updates']>18000 or r['budget']['nontraining_forwards']>6600:errors.append('execution_hard_budget')
    return errors

def signal_once(saved):
    """Caller holds stop_once.lock; never send repeated TERM during teardown."""
    path=FOLDER/'stop_signals.json';history=read(path) if path.exists() else {}
    key=f"{saved['boot_id']}:{saved['pid']}:{saved['starttime']}"
    if key in history or same(saved) is None:return False
    history[key]=dict(identity=saved,signal='SIGTERM',epoch=time.time())
    write(path,history)
    try:os.kill(saved['pid'],signal.SIGTERM)
    except ProcessLookupError:return False
    return True

def stop_owned(q,reason):
    # Main guard, watchdog and proc-only backstop share this serialization.
    with (FOLDER/'stop_once.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        return stop_owned_serial(q,reason)

def stop_owned_serial(q,reason):
    current=queue_current(q)
    if current:os.kill(q['pid'],signal.SIGSTOP)
    workers=active_workers();sent=[]
    for w in workers:
        if same(w['identity'],w['record']['module']) and signal_once(w['identity']):sent.append(w['identity'])
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        left=[w for w in workers if process_remaining(w['identity'])]
        unsealed=[w['runtime_path'] for w in workers if not read(w['runtime_path']).get('process_exited')]
        if not left and not unsealed:break
        time.sleep(1)
    record=dict(epoch=time.time(),reason=reason,queue=q,queue_stopped=bool(current),SIGTERM_workers=sent,still_active=left,runtime_not_closed=unsealed,complete_checkpoint_yield_observed=not left and not unsealed,budget=read(OUT/'budget.json'))
    write(FOLDER/'last_safe_stop.json',record)
    if left or unsealed:raise RuntimeError('Full-state stop did not complete within60seconds; no SIGKILL or retry allowed')
    return record

def request_stop(*_):
    global STOP
    STOP=True

def log(r):
    write(FOLDER/'current.json',r)
    with (FOLDER/'ledger.jsonl').open('a',buffering=1) as f:f.write(json.dumps(r)+'\n')

def watchdog(config_path,main_ident,terminal):
    config=read(config_path);started=time.monotonic()
    try:
        while not terminal.exists():
            time.sleep(1)
            if STOP:raise RuntimeError('watchdog_stop_signal')
            if same(main_ident) is None:
                stop_owned(config['queue'],'sidecar_main_died');return
            heartbeat=FOLDER/'current.json'
            if not heartbeat.exists() and time.monotonic()-started>8:
                stop_owned(config['queue'],'sidecar_initial_heartbeat_missing');return
            if heartbeat.exists() and time.time()-heartbeat.stat().st_mtime>8:
                stop_owned(config['queue'],'sidecar_heartbeat_stale');return
    except BaseException:
        stop_owned(config['queue'],'watchdog_exception');raise

def run(config_path):
    config=read(config_path);q=config['queue'];a=validate_original(q);needs=read(FOLDER/'expected_sizes.json') if (FOLDER/'expected_sizes.json').exists() else {};history=deque();terminal=FOLDER/f'terminal_{time.time_ns()}.json';child=None;completed=False;resumed=False
    with (FOLDER/'monitor.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            # A different process independently checks this guard's life/heartbeat.
            own=identity(os.getpid());child=subprocess.Popen([sys.executable,'-u','-m',__spec__.name,'--watchdog',str(config_path),'--main-identity',json.dumps(own),'--terminal',str(terminal)],cwd=ROOT)
            first=None
            while True:
                if STOP:raise RuntimeError('sidecar_stop_signal')
                if child.poll() is not None:raise RuntimeError('watchdog_died')
                r=ledger(a,needs);r.update(state='running' if resumed else 'admission',monitor_pid=os.getpid(),watchdog_pid=child.pid,queue=q,addendum_sha256=ADD_SHA);log(r);write(FOLDER/'expected_sizes.json',needs)
                errors=reasons(r,not resumed)
                history.append((r['epoch'],r['H']))
                while history and r['epoch']-history[0][0]>30:history.popleft()
                if history and max(v for _,v in history)-r['H']>=.5*GIB:errors.append('unexplained_30second_headroom_drop')
                if errors:raise RuntimeError(','.join(errors))
                if not resumed:
                    assert not active_workers()
                    if first is None:first=r
                    if r['epoch']-first['epoch']>=30:
                        assert r['H']>=6.5*GIB and first['H']>=6.5*GIB
                        assert first['H']-r['H']<.5*GIB
                        assert same(q)['state']=='T'
                        os.kill(q['pid'],signal.SIGCONT);resumed=True;write(FOLDER/'admission.json',dict(first=first,second=r,resumed_original_queue=True,epoch=time.time()))
                else:
                    current=same(q)
                    state=read(OUT/'queue_status.json')
                    if current and current['state']=='T':raise RuntimeError('queue_unexpectedly_stopped')
                    # Queue may exit after setting complete; keep guard for cost until release file.
                    if current is None and state['state']!='complete':raise RuntimeError('queue_died_incomplete')
                    if state['state']=='complete' and not active_workers() and (FOLDER/'release_after_cost.json').exists():completed=True;break
                time.sleep(2)
        except BaseException as exc:
            stop_owned(q,repr(exc));write(FOLDER/'failure.json',dict(epoch=time.time(),error=repr(exc),resumed=resumed));raise
        finally:
            if not completed and not (FOLDER/'last_safe_stop.json').exists():stop_owned(q,'guard_finally')
            write(terminal,dict(complete=completed,epoch=time.time()))
            if child:
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:stop_owned(q,'watchdog_failed_to_exit')
        write(FOLDER/'complete.json',dict(complete=True,epoch=time.time(),all_GPU_workers_exited=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--snapshot',action='store_true');p.add_argument('--watchdog');p.add_argument('--main-identity');p.add_argument('--terminal');args=p.parse_args();FOLDER.mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    if args.watchdog:return watchdog(args.watchdog,json.loads(args.main_identity),Path(args.terminal))
    if args.snapshot:
        assert sha(ADD)==ADD_SHA;print(json.dumps(ledger(read(ADD),{}),indent=2));return
    run(Path(args.run))


if __name__=='__main__':main()
