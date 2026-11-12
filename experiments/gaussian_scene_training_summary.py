"""Summarize retained optimization and preserved discarded work across resumptions."""
import json
import math
import re
from pathlib import Path

import numpy as np

from experiments.gaussian_scene_data import OUT,KINDS,SEEDS,read
from experiments.gaussian_memory_incremental_data import write_json


def lines(path):
    return [json.loads(row) for row in path.read_text().splitlines() if row.strip()]


def elapsed_segments(rows):
    total=0.;last=0.;segments=0
    for row in rows:
        elapsed=row['elapsed_seconds']
        if elapsed<last:total+=last;segments+=1
        last=elapsed
    return total+last,segments+bool(rows)


def main():
    completed=[];pending=[]
    for seed in SEEDS:
        for kind in KINDS:
            name=f'{kind}_{seed}';folder=OUT/name;marker=folder/'status.json'
            if not marker.exists():pending.append(name);continue
            status=read(marker);assert status['complete'] and status['steps']==2000
            rows=lines(folder/'training.jsonl')
            assert [row['step'] for row in rows]==list(range(1,2001))
            assert all(math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']) for row in rows)
            retained,segments=elapsed_segments(rows);by_step={row['step']:row for row in rows};discarded=[]
            for path in sorted(folder.glob('interrupted_after_*.jsonl')):
                old=lines(path)
                if not old:continue
                checkpoint=int(re.match(r'interrupted_after_(\d+)',path.name).group(1))
                base=by_step.get(checkpoint,{}).get('elapsed_seconds',0.)
                # Original-session tail continues its retained prefix; a restarted tail resets its elapsed timer.
                same_clock=old[0]['elapsed_seconds']>=base
                seconds=max(0.,old[-1]['elapsed_seconds']-(base if same_clock else 0.))
                discarded.append(dict(path=str(path),checkpoint_step=checkpoint,discarded_updates=len(old),
                    elapsed_seconds=seconds,clock_inference='continued retained clock' if same_clock else 'reset after checkpoint'))
            completed.append(dict(name=name,kind=kind,seed=seed,updates=2000,all_logged_losses_and_gradient_norms_finite=True,
                initial_mean20_loss=float(np.mean([row['loss'] for row in rows[:20]])),
                final_mean20_loss=float(np.mean([row['loss'] for row in rows[-20:]])),
                retained_loop_elapsed_seconds=retained,observed_clock_segments=segments,
                discarded_loop_elapsed_seconds=sum(row['elapsed_seconds'] for row in discarded),discarded=discarded,
                peak_allocated_GiB=status['peak_allocated_GiB']))
    write_json(OUT/'training_summary.json',dict(complete=not pending,completed=completed,pending=pending,
        retained_loop_worker_hours=sum(row['retained_loop_elapsed_seconds'] for row in completed)/3600,
        discarded_loop_worker_hours=sum(row['discarded_loop_elapsed_seconds'] for row in completed)/3600,
        scope='Logged optimization-loop elapsed, reconstructed at timer resets; excludes model/data loading and paused wall time. Discarded uncheckpointed work separately retained. Not exclusive GPU performance.'))
    print(dict(completed_fits=len(completed),pending_fits=len(pending)),flush=True)


if __name__=='__main__':main()
