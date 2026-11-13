"""Finite tail resharing on released owned cards, never concurrent fit owners."""
import os,subprocess,time
from experiments import native_scene_exchange_data as d

def main():
    old=d.read(d.OUT/'workers/formal_gpu0_v3.json');assert old.get('actual_process_wait_returned') and old['exit_code']==75
    state=d.load(d.final('L',20261111).parent/'latest.pt');step=state['step'];assert step==1270;del state
    assert not (d.OUT/'fits/L_20261112_initial.json').exists()
    reg=d.read(d.OUT/'parallel_execution.json');names=['formal_tail_L12_v4','formal_tail_L11_v4'];reg['active_workers']=[x for x in reg['active_workers'] if x!='formal_gpu0_v3']+names
    reg['previous_valid_saved_segments'].append('formal_gpu0_v3');reg['tail_reassignment']=dict(utc=d.now(),retired_worker='formal_gpu0_v3',reason='Release shared card0 and split independent pending L fits across first released own idle cards.',L11_exact_step=step,L12_step=0,expected_workers=names,max_concurrent_GPU_workers=7)
    d.write(d.OUT/'parallel_execution.json',reg)
    d.journal(d.OUT/'engineering_versions.jsonl',dict(utc=d.now(),version='v4_tail_resharing',change=reg['tail_reassignment'],valid_steps_discarded=0,old_segment_actual_exit75=True,scientific_conditions_changed=False))
    jobs=[dict(name=names[0],job='L:20261112',candidates=['formal_gpu6_v3']),dict(name=names[1],job='L:20261111',candidates=['formal_gpu4_v3','formal_gpu7_v3'])];started={};children=[]
    while jobs or any(c.poll() is None for c in children):
        for job in list(jobs):
            ready=[]
            for name in job['candidates']:
                w=d.read(d.OUT/'workers'/f'{name}.json')
                if w.get('actual_process_wait_returned') and w['exit_code']==0:ready.append(w)
            if not ready:continue
            w=ready[0];gpu=w['gpu_uuid']
            args=['/home/liwt/miniColmap/.venv/bin/python','-u','-m','experiments.native_scene_exchange_launch','fits','--gpu',gpu,'--name',job['name'],'--arms',job['job']]
            log=(d.OUT/'logs'/f"{job['name']}_dispatcher.log").open('a',buffering=1)
            child=subprocess.Popen(args,cwd=d.ROOT,stdout=log,stderr=subprocess.STDOUT);children.append(child);started[job['name']]=dict(gpu_uuid=gpu,job=job['job'],launcher_pid=child.pid,started_utc=d.now(),prior_worker=w['name']);jobs.remove(job)
            d.write(d.OUT/'tail_dispatch.json',dict(pid=os.getpid(),started=started,pending=jobs,updated_utc=d.now()))
        if jobs or any(c.poll() is None for c in children):time.sleep(10)
    codes=[c.wait() for c in children];d.write(d.OUT/'tail_dispatch.json',dict(pid=os.getpid(),started=started,pending=[],actual_launcher_exit_codes=codes,complete=all(c==0 for c in codes),updated_utc=d.now()));assert all(c==0 for c in codes),codes

if __name__=='__main__':main()
