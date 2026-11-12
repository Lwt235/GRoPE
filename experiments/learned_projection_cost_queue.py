"""Run finite query timing when all fits are dispatched and a worker slot is free."""
import fcntl
import subprocess
import sys
import time

from experiments.learned_projection_data import *
from experiments import gaussian_scene_orchestrate as resource


def main():
    with (OUT/'cost_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            q=read(OUT/'queue_status.json')
            if q.get('failed') and not q.get('running'):raise RuntimeError(q['failed'])
            first_seed_ready=all((OUT/f'{kind}_{SEEDS[0]}'/'status.json').exists() for kind in KINDS)
            if first_seed_ready and not q.get('pending') and len(q.get('running',[]))<=3:break
            time.sleep(30)
        while not (OUT/'cost.json').exists():
            q=read(OUT/'queue_status.json');states=resource.occupancy();apps=resource.device_processes()
            own=resource.descendants({j['pid'] for j in q.get('running',[])})
            occupied={j['gpu'] for j in q.get('running',[])}
            shared_cards={j['gpu'] for j in q.get('running',[]) if apps[j['gpu']]-own}
            idle=[g for g,u,f,v in states if g not in occupied and u<=512 and f>=40000 and v<=5]
            shared=[g for g,u,f,v in sorted(states,key=lambda x:x[3]) if g not in occupied and f>=22528] if len(shared_cards)<2 else []
            if not idle and not shared:time.sleep(30);continue
            gpu=(idle or shared)[0];command=[sys.executable,'-u','-m','experiments.learned_projection_launch','--gpu',str(gpu)]
            if idle:command+=['--require-idle']
            command+=['experiments.learned_projection_cost']
            with (OUT/'cost.log').open('a',buffering=1) as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                tail=(OUT/'cost.log').read_text()[-4000:]
                if 'no job launched' not in tail:raise RuntimeError(tail)
                time.sleep(20)
        assert read(OUT/'cost.json')['complete']
        print('Finite full-query cost complete.',flush=True)


if __name__=='__main__':main()
