"""Check explicit checkpoint reuse and the newly exported reference predictions."""
import numpy as np
from experiments import round53_train as P
from experiments.round53_data import OUT, PREVIOUS, read, write, sha


def main():
    P.check(); records=[]
    for entry in read(OUT/'reuse_manifest.json')['entries']:
        role, origin, seed=entry['role'],entry['origin_role'],entry['seed']
        folder=OUT/'runs'/role/str(seed); old=PREVIOUS/'runs'/origin/str(seed)
        run=read(folder/'run.json'); original=read(old/'run.json')
        if sha(folder/'model.pt')!=sha(old/'model.pt'):raise ValueError('checkpoint changed')
        if read(folder/'origin_run.json')!=original:raise ValueError('origin record changed')
        if run['lock_sha256']!=original['lock_sha256']:raise ValueError('origin lock relabeled')
        for variant in ('baseline','rgb_permuted'):
            before=PREVIOUS/'evaluation'/origin/f'{seed}_{variant}.json'
            after=OUT/'evaluation'/role/f'{seed}_{variant}.json'
            a,b=read(before),read(after)
            for path,record in ((before,a),(after,b)):
                if sha(path.with_suffix('.npz'))!=record['prediction_sha256']:raise ValueError('prediction changed')
            with np.load(before.with_suffix('.npz')) as x, np.load(after.with_suffix('.npz')) as y:
                differences={k:float(np.max(np.abs(x[k].astype(float)-y[k].astype(float)))) for k in x.files}
            records.append(dict(role=role,origin_role=origin,seed=seed,variant=variant,
                prediction_max_differences=differences,
                metric_differences={k:b['metrics'][k]-a['metrics'][k] for k in P.METRICS}))
    write(OUT/'reference_replay.json',dict(records=records,source_sha256=sha(__file__),
        reuse_manifest_sha256=sha(OUT/'reuse_manifest.json'),
        scope='12 reused R52 checkpoints, 24 prediction exports; no new training or independent confirmation.',
        confirmation_access=False))
    print('REFERENCE REPLAY',len(records),'exports; maximum prediction difference',
          max(v for r in records for v in r['prediction_max_differences'].values()),flush=True)


if __name__=='__main__':main()
