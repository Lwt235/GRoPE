"""R55 host queue: launch only on devices observed idle twice."""
import os
import subprocess
import sys
import time
from experiments.round55_data import OUT, ROOT, write
from experiments import round55_train as P


def occupancy():
    value=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {int(row.split(',')[0]):tuple(float(v) for v in row.split(',')[1:]) for row in value.strip().splitlines()}


def main():
    P.check();destination=OUT/'queue_status.json'
    if destination.exists():raise FileExistsError('retain prior queue state')
    state=dict(state='running',children=[],active=[],observations=[])
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for stage in ('smoke','train','eval'):
        if stage=='eval':P.seal()
        pending=[(None,None)] if stage=='smoke' else [(s,g) for g in range(len(P.GROUPS)) for s in P.SEEDS] if stage=='train' else [(s,None) for s in P.SEEDS]
        running=[]
        while pending or running:
            active=[]
            for proc,log,rec in running:
                code=proc.poll()
                if code is None:active.append((proc,log,rec))
                else:
                    log.close();rec.update(exit_code=code,finished=time.time());state['children'].append(rec)
                    print('EXIT',rec['tag'],code,flush=True)
            running=active
            if any(r['exit_code']!=0 for r in state['children']):pending=[]
            if pending:
                first=occupancy();time.sleep(5);second=occupancy()
                used={r['gpu'] for _,_,r in running}
                idle=[g for g in second if g not in used and second[g][0]<128 and second[g][1]<5 and first.get(g,(1e9,100))[0]<128 and first[g][1]<5]
                state['observations'].append(dict(time=time.time(),first=first,second=second))
                while idle and pending and len(running)<6:
                    gpu=idle.pop(0)
                    if occupancy()[gpu][0]>=128:continue
                    seed,group=pending.pop(0);tag=stage+(f'_{seed}' if seed is not None else '')+(f'_g{group}' if group is not None else '')
                    command=[sys.executable,'-m','experiments.round55_train',stage,'--gpu',str(gpu)]
                    if seed is not None:command+=['--seed',str(seed)]
                    if group is not None:command+=['--group',str(group)]
                    logfile=OUT/(tag+'.log');log=open(logfile,'x')
                    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    rec=dict(stage=stage,tag=tag,gpu=gpu,pid=process.pid,seed=seed,group=group,command=command,log=str(logfile),started=time.time())
                    running.append((process,log,rec));print('START',tag,'GPU',gpu,'PID',process.pid,flush=True)
            state.update(stage=stage,active=[r for _,_,r in running],pending=pending);write(destination,state)
            if running or pending:time.sleep(10)
        if any(r['exit_code']!=0 for r in state['children']):
            state.update(state='failed',active=[]);write(destination,state);return 1
    P.report();P.check();state.update(state='complete',active=[],finished=time.time());write(destination,state)
    return 0


if __name__=='__main__':sys.exit(main())
