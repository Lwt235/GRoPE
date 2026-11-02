"""Overlap the prespecified CPU checks with training; check each model once."""
import time
import torch
from experiments import round54_train as P
from experiments import round54_postcheck as C
from experiments.round54_data import OUT, ROOT, read, write, sha, load


def main():
    source_digest=sha(__file__);plan=read(OUT/'postcheck/plan.json')
    for path,digest in plan['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('postcheck source changed')
    destination=OUT/'postcheck/execution.json'
    if destination.exists():raise FileExistsError('retain checker execution')
    write(destination,dict(source_sha256=source_digest,
        scheduling='Check each already-selected model on CPU once while GPU fits continue; old_eval replay waits for all selections and exports.',
        sources_science_changed=False,started=time.time()))
    (OUT/'postcheck/execution_source.py').write_bytes(open(__file__,'rb').read())
    P.setup();val,_=load(OUT,'val',torch.device('cpu'));done={}
    while len(done)<len(P.ROLES)*len(P.SEEDS):
        state=read(OUT/'queue_status.json')
        if state['state']=='failed':raise RuntimeError('GPU queue failed; retain completed CPU records')
        for role in P.ROLES:
            for seed in P.SEEDS:
                key=f'{role}_{seed}';runpath=OUT/'runs'/role/str(seed)/'run.json'
                if key in done or not runpath.exists():continue
                record=C.operator(role,seed,val);record['run_record_sha256']=sha(runpath)
                done[key]=record;write(OUT/'postcheck/records'/f'{key}.json',record)
                print('CHECK',key,'ce_error',record['validation_ce_error'],'p_error',record['probability_max_error'],'warnings',record['warning'],flush=True)
        if len(done)<27:time.sleep(10)
    while not (OUT/'summary.json').exists():
        if read(OUT/'queue_status.json')['state']=='failed':raise RuntimeError('GPU evaluation failed')
        time.sleep(10)
    selections=read(OUT/'selection.json')
    if len(selections['records'])!=27:raise ValueError('all selections required before old_eval replay')
    for r in selections['records']:
        key=f"{r['role']}_{r['seed']}";runpath=OUT/'runs'/r['role']/str(r['seed'])/'run.json'
        if read(runpath)!=r or sha(runpath)!=done[key]['run_record_sha256']:raise ValueError('selected model changed after check')
    data,ids=load(OUT,'old_eval',torch.device('cpu'));metrics=C.replay_metrics(data,ids)
    P.check()
    for path,digest in plan['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('postcheck source changed')
    if sha(__file__)!=source_digest:raise ValueError('execution source changed')
    write(OUT/'postcheck/result.json',dict(records=list(done.values()),metrics=metrics,
        warnings=sum(r['warning'] for r in done.values()),confirmation_access=False,
        scope='same-author independent NumPy operator; CPU validation and saved metric replay; not a second complete network or lab'))
    print('POSTCHECK COMPLETE',len(done),len(metrics),flush=True)


if __name__=='__main__':main()
