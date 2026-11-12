"""Finite postprocessing; scientific decision and actual visual review remain explicit."""
import json
import os
import subprocess
import sys
import time
from experiments.native_input_pe_data import *
from experiments.native_input_pe_queue import jobs,done
from experiments import gaussian_scene_orchestrate as resource


def run(module,args=()):
    env=os.environ.copy();env['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib_native_input_pe')
    subprocess.run([sys.executable,'-u','-m',module,*args],cwd=ROOT,env=env,check=True)


def cost_tables():
    p=plan();s=read(OUT/'scoring/summary.json');fits=[]
    for seed in SEEDS:
        for kind in KINDS:
            folder=OUT/f'{kind}_{seed}';rr=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
            elapsed=0.;previous=0.
            for r in rr:
                t=r['elapsed_seconds']
                if t<previous:elapsed+=previous
                previous=t
            elapsed+=previous
            snapshots=[json.loads(x) for x in (folder/'checkpoints.jsonl').read_text().splitlines()]
            status=read(folder/'status.json');fits.append(dict(kind=kind,seed=seed,retained_loop_seconds=elapsed,
                snapshot_seconds=sum(x['save_seconds'] for x in snapshots),peak_GiB=status['peak_GiB'],
                final_bytes=checkpoint_path(kind,seed).stat().st_size,
                interrupted_updates=sum(len(x.read_text().splitlines()) for x in folder.glob('interrupted_*.jsonl'))))
    evaluation=[read(x) for x in (OUT/'evaluation').glob('*.json')]
    geometry=[read(x) for x in (OUT/'geometry').glob('*.json')]
    write_json(OUT/'cost_description.json',dict(complete=True,training=fits,
        retained_training_worker_hours=sum(x['retained_loop_seconds'] for x in fits)/3600,
        snapshot_worker_seconds=sum(x['snapshot_seconds'] for x in fits),peak_training_GiB=max(x['peak_GiB'] for x in fits),
        final_weights_bytes=sum(x['final_bytes'] for x in fits),evaluation_export_worker_seconds=sum(x['seconds'] for x in evaluation),
        geometry_preparation_worker_seconds=sum(x['seconds'] for x in geometry),
        cache_bytes=sum(x.stat().st_size for x in CACHE.rglob('*') if x.is_file()),
        full_warm_query=read(OUT/'cost.json'),
        timing_scope='Worker sums not wall time; retained loop includes most recurring snapshot I/O, not added again. Excludes data/model loading, final persistence and paused intervals. Shared GPUs, no exclusive speed claim.',
        state_scope='4319652byte unchanged scene state; derived geometry cache is a development accelerator, not per-scene deployment state.12research finals are shared-model seeds/variants, not per-scene weights.'))
    lines=['# 单次输入PE对照固定数表','','正降误差表示前者更好；区间为配对地点/种子95%区间。全部旧数据。',
        '','| 固定开发原四查询 | 空间log-MSE | 原始log-MSE | 完整全图AbsRel |','| --- | ---: | ---: | ---: |']
    for kind in ['L','P','S','G','Previous_L','Previous_R']:
        values=[s['summaries']['development/original/'+e+'/'+kind]['mean'] for e in ['missing/roi/centered_log_mse','missing/roi/log_mse','full/whole/absrel']]
        lines.append('| '+kind+' | '+' | '.join(f'{v:.9f}' for v in values)+' |')
    lines+=['','Previous为不同接口的历史背景，不作为新PE匹配控制。','','| 比较／查询 | 空间降误差及95%区间 | 三种子% |','| --- | --- | --- |']
    for view in ['original','heldout','pooled']:
        for pair in ['G-P','G-S','S-P','P-L','S-L','G-L']:
            v=s['comparisons'][f'development/{view}/missing/roi/centered_log_mse/'+pair];lo,hi=v['interval95_percent']
            lines.append(f"| {pair}/{view} | {v['relative_error_reduction_percent']:+.3f}% [{lo:+.3f},{hi:+.3f}]% | "+'/'.join(f'{x:+.3f}' for x in v['per_seed_relative_percent'])+' |')
    lines+=['','| 地点/其他查询 | G对P降空间误差% | G对L降空间误差% |','| --- | --- | --- |']
    for cohort in ['train_original','train_added','train_all']:
        values=[]
        for pair in ['G-P','G-L']:
            v=s['comparisons'][f'{cohort}/heldout/missing/roi/centered_log_mse/'+pair];lo,hi=v['interval95_percent']
            values.append(f"{v['relative_error_reduction_percent']:+.3f}% [{lo:+.3f},{hi:+.3f}]%")
        lines.append('| '+cohort+' | '+' | '.join(values)+' |')
    lines+=['','完整地点、种子、原始/偏差/支持条件和查询结果见[summary.json](scoring/summary.json)。','']
    (OUT/'report_tables.md').write_text('\n'.join(lines))


def main():
    while not all(done(j) for j in jobs()):
        queue=read(OUT/'queue_status.json') if (OUT/'queue_status.json').exists() else {}
        if queue.get('failed'):raise RuntimeError('Training failed; diagnose before evaluating incomplete comparison')
        write_json(OUT/'finish_status.json',dict(complete=False,state='waiting_for12fits_and_exports',updated_epoch=time.time()))
        time.sleep(30)
    while not (OUT/'cost.json').exists():
        states=resource.occupancy();idle=[r for r in states if r[1]<=512 and r[2]>=15360 and r[3]<=5]
        spare=[r for r in sorted(states,key=lambda r:(r[3],-r[2])) if r[2]>=15360]
        choices=idle or spare
        if not choices:time.sleep(30);continue
        code=subprocess.run([sys.executable,'-u','-m','experiments.native_input_pe_launch','--gpu',str(choices[0][0]),'--cap','9',
            '--admission','15360','--margin','6144','experiments.native_input_pe_cost'],cwd=ROOT).returncode
        if code not in [0,75]:raise RuntimeError('Full query cost check failed')
    for module in ['experiments.native_input_pe_score','experiments.native_input_pe_checks','experiments.native_input_pe_figures']:
        write_json(OUT/'finish_status.json',dict(complete=False,state=module,updated_epoch=time.time()));run(module)
    cost_tables()
    write_json(OUT/'finish_status.json',dict(complete=False,state='artifacts_ready',
        remaining=['actual image review','scientific report and decision','living documentation sync'],updated_epoch=time.time()))
    print('All artifacts ready; actual image review and scientific decision remain.',flush=True)


if __name__=='__main__':main()
