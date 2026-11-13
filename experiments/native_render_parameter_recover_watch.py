"""Adopt existing own trainers after a metadata-only controller failure."""
import os
import signal
import subprocess
import time
from experiments.native_render_parameter_data import *
from experiments.native_render_parameter_launch import active_seconds
from experiments.native_render_parameter_queue import done
from experiments import gaussian_scene_orchestrate as resource


def main():
    watched=[]
    for path in (OUT/'runtime').glob('*.json'):
        r=read(path)
        if not isinstance(r,dict) or not r.get('runtime_session') or r.get('process_exited'):continue
        assert r['module']=='experiments.native_render_parameter_train'
        proc=Path('/proc')/str(r['pid']);assert proc.exists() and proc.stat().st_uid==os.getuid()
        assert b'experiments.native_render_parameter_train' in proc.joinpath('cmdline').read_bytes()
        kind=r['args'][1];seed=int(r['args'][3]);watched.append(dict(path=path,record=r,kind=kind,seed=seed,job=dict(name=f'{kind}_{seed}',kind=kind,seed=seed),signal_sent=False))
    assert len(watched)==2
    while watched:
        for item in list(watched):
            r=item['record'];pid=r['pid'];alive=resource.process_state(pid) not in (None,'Z')
            if not alive:
                complete=done(item['job']);r.update(process_exited=True,seconds=time.time()-r['start_epoch'],finished_utc=now(),returncode=0 if complete else 75 if item['signal_sent'] else 1,controller_recovered=True,recovery_controller_pid=os.getpid());write(item['path'],r);watched.remove(item);print(dict(adopted_trainer_exited=pid,job=item['job']['name'],complete=complete),flush=True);continue
            free=int(subprocess.check_output(['nvidia-smi','-i',str(r['gpu']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if (free<6144 or active_seconds()>=8*3600) and not item['signal_sent']:
                os.kill(pid,signal.SIGTERM);item['signal_sent']=True;item['signal_time']=time.time()
            if item['signal_sent'] and time.time()-item['signal_time']>60:raise RuntimeError('Owned trainer did not save and exit within60s; preserve and return for diagnosis')
        if watched:time.sleep(3)
    print('Existing trainers drained without duplicate dispatch.',flush=True)


if __name__=='__main__':main()
