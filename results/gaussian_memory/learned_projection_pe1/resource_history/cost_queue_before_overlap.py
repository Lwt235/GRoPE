"""Run the finite query-cost measurement once the nine GPU fits have exited."""
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
            if q.get('state')=='all_nine_fits_and_predictions_complete':break
            time.sleep(30)
        while not (OUT/'cost.json').exists():
            states=resource.occupancy()
            idle=[g for g,u,f,v in states if u<=512 and f>=40000 and v<=5]
            shared=[g for g,u,f,v in sorted(states,key=lambda x:x[3]) if f>=22528]
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
