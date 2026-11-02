"""Close the user-interrupted R53 without changing its frozen experiment."""
import os
import time
from pathlib import Path
import numpy as np
import torch
from experiments import round53_train as P
from experiments import round53_postcheck as C
from experiments.round53_data import OUT, ROOT, read, write, sha, load
from experiments.round53_queue import occupancy


def main():
    P.check();P.setup();destination=OUT/'resume_status.json'
    if destination.exists():raise FileExistsError('preserve previous resumption')
    pause=read(OUT/'user_pause.json')
    for r in pause['processes']:
        path=Path('/proc')/str(r['pid'])/'cmdline'
        if path.exists() and path.read_bytes().split(b'\0')[:3]==[s.encode() for s in r['argv'][:3]]:
            raise RuntimeError('old R53 process still live')
    missing=[(role,seed) for role in P.ROLES for seed in P.SEEDS if not (OUT/'runs'/role/str(seed)/'run.json').exists()]
    if missing!=[('ray_gaussian',33),('ray_affine',33)]:raise ValueError('unexpected incomplete fits')
    for role in P.ROLES:
        for seed in P.SEEDS:
            folder=OUT/'runs'/role/str(seed)
            if (folder/'run.json').exists() and sha(folder/'model.pt')!=read(folder/'run.json')['model_sha256']:
                raise ValueError('existing selected checkpoint changed')
    write(OUT/'resume_plan.json',dict(source_sha256=sha(__file__),user_resumed=True,
        original_lock_sha256=sha(OUT/'lock.json'),missing_fits=missing,
        policy='Preserve16 completed new fits and12 references. Archive only interrupted history. Restart the one interrupted fit from the identical seed/batches because no optimizer checkpoint exists; no choice among completed trials.',
        confirmation_access=False))
    (OUT/'resume_source.py').write_bytes(Path(__file__).read_bytes())
    interrupted=OUT/'runs/ray_gaussian/33'
    archive=OUT/'interrupted/user_pause_20261102/ray_gaussian/33'
    if (interrupted/'model.pt').exists() or (interrupted/'run.json').exists():raise ValueError('do not replace complete fit')
    archive.parent.mkdir(parents=True,exist_ok=True);interrupted.rename(archive)
    observations=[]
    while True:
        first=occupancy();time.sleep(5);second=occupancy();observations.append(dict(first=first,second=second,time=time.time()))
        idle=[g for g in second if first[g][0]<128 and first[g][1]<5 and second[g][0]<128 and second[g][1]<5]
        if idle:break
        write(destination,dict(state='waiting_for_idle_gpu',pid=os.getpid(),observations=observations));time.sleep(10)
    gpu=idle[0];device=P.setup(gpu)
    state=dict(state='running',pid=os.getpid(),gpu=gpu,device=torch.cuda.get_device_name(gpu),observations=observations,finished_fits=[])
    write(destination,state);print('RESUME GPU',gpu,'PID',os.getpid(),flush=True)
    fit,_=load(OUT,'fit',device);val,_=load(OUT,'val',device)
    for role,seed in missing:
        P.fit_one(role,seed,fit,val,device);state['finished_fits'].append([role,seed]);write(destination,state)
    del fit,val;torch.cuda.empty_cache();P.seal()
    for seed in P.SEEDS:P.evaluate(gpu,seed)
    P.report();P.check()
    # Reuse the28 checks made before user pause, preserving their source/run identities.
    plan=read(OUT/'postcheck/plan.json')
    for path,digest in plan['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('postcheck source changed')
    data,_=load(OUT,'val',torch.device('cpu'));records=[]
    for role in P.ROLES:
        for seed in P.SEEDS:
            path=OUT/'postcheck/records'/f'{role}_{seed}.json';run=OUT/'runs'/role/str(seed)/'run.json'
            if path.exists():
                r=read(path)
                if sha(run)!=r['run_record_sha256']:raise ValueError('checked selection changed')
            else:
                r=C.operator(role,seed,data);r['run_record_sha256']=sha(run);write(path,r)
                print('CHECK',role,seed,r,flush=True)
            records.append(r)
    data,ids=load(OUT,'old_eval',torch.device('cpu'));metrics=C.replay_metrics(data,ids)
    write(OUT/'postcheck/result.json',dict(records=records,metrics=metrics,
        warnings=sum(r['warning'] for r in records),confirmation_access=False,
        scope='28 unchanged pre-pause records plus2 new independent NumPy/CPU checks; all60 metric groups replayed; same author, not a second lab'))
    P.check();state.update(state='complete',finished=time.time());write(destination,state)
    print('R53 RESUMPTION COMPLETE',len(records),len(metrics),flush=True)


if __name__=='__main__':main()
