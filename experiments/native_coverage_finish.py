"""Wait for the finite queue, then score/check/plot; leave scientific review to caller."""
import json
import os
import subprocess
import sys
import time
from experiments.native_coverage_data import *
from experiments.native_coverage_queue import jobs,done
from experiments import gaussian_scene_orchestrate as resource


def run(module,args=()):
    env=os.environ.copy();env['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib_native_coverage')
    subprocess.run([sys.executable,'-u','-m',module,*args],cwd=ROOT,env=env,check=True)


def cost_tables():
    p=plan();s=read(OUT/'scoring/summary.json');fits=[];checkpoint_seconds=0.
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
            snapshot_seconds=sum(x['save_seconds'] for x in snapshots);checkpoint_seconds+=snapshot_seconds
            status=read(folder/'status.json');fits.append(dict(kind=kind,seed=seed,retained_loop_seconds=elapsed,
                snapshot_seconds=snapshot_seconds,peak_GiB=status['peak_GiB'],final_bytes=checkpoint_path(kind,seed).stat().st_size,
                interrupted_updates=sum(len(x.read_text().splitlines()) for x in folder.glob('interrupted_*.jsonl'))))
    evaluations=[read(x) for x in (OUT/'evaluation').glob('*.json')]
    write_json(OUT/'cost_description.json',dict(complete=True,training=fits,
        retained_training_worker_hours=sum(x['retained_loop_seconds'] for x in fits)/3600,
        snapshot_worker_seconds=checkpoint_seconds,peak_training_GiB=max(x['peak_GiB'] for x in fits),
        final_weights_bytes=sum(x['final_bytes'] for x in fits),
        evaluation_export_worker_seconds=sum(x['seconds'] for x in evaluations),
        source_construction=read(OUT/'carrier_summary.json'),
        cache_bytes=sum(x.stat().st_size for x in CACHE.rglob('*') if x.is_file()),
        timing_scope='Worker sums not wall time; training loop excludes model/data loading and final persistent saving. Snapshot I/O overlaps loop timing and is not added again; resumed timer segments reconstructed. Shared GPUs, no exclusive speed claim.',
        unchanged_fresh_query_cost_reference=str(BASE/'cost_description.json'),
        weight_scope='Research checkpoints for six shared models, not per-scene weights. Same architecture and4319652byte scene tensors.'))
    lines=['# 覆盖对照固定数表','','正降误差表示Render较好；区间为固定地点/种子配对95%区间。','','| 开发查询 | 12地点Render对Local | 32地点Render对Local | 覆盖交互及95%区间 |','| --- | --- | --- | --- |']
    for view,label in [('original','原四查询（主）'),('heldout','其他四查询'),('pooled','八查询合并')]:
        key=f'development/{view}/missing/roi/centered_log_mse';vals=[]
        for pre in ['N_','W_']:
            v=s['comparisons'][key+'/'+pre+'Render-'+pre+'Local'];lo,hi=v['interval95_percent'];vals.append(f"{v['relative_error_reduction_percent']:+.2f}% [{lo:+.2f},{hi:+.2f}]%")
        v=s['coverage_interactions'][key];lo,hi=v['interval95_absolute']
        lines.append('| '+label+' | '+' | '.join(vals)+f" | {v['mean_absolute']:+.9f} [{lo:+.9f},{hi:+.9f}] |")
    lines+=['','交互=(R−L)32−(R−L)12，单位为log方差；负值有利。三查询集合相关，不是三次独立确认。','','| 地点／查询 | 32地点Local空间误差 | 32地点Render空间误差 | R相对L降误差及95%区间 |','| --- | ---: | ---: | --- |']
    for cohort,cl in [('development','开发'),('train_original','原训练'),('train_added','新增训练'),('train_all','全部训练')]:
        for view,vl in [('original','拟合/原位置'),('heldout','留出/其他位置')]:
            v=s['comparisons'][f'{cohort}/{view}/missing/roi/centered_log_mse/W_Render-W_Local'];lo,hi=v['interval95_percent']
            lines.append(f"| {cl}/{vl} | {v['control']:.9f} | {v['candidate']:.9f} | {v['relative_error_reduction_percent']:+.2f}% [{lo:+.2f},{hi:+.2f}]% |")
    lines+=['','| 原开发四查询，32地点组相对12地点组 | 降误差及95%区间 |','| --- | --- |']
    for pair in ['W_Render-N_Render','W_Local-N_Local']:
        v=s['comparisons']['development/original/missing/roi/centered_log_mse/'+pair];lo,hi=v['interval95_percent'];lines.append(f"| {pair} | {v['relative_error_reduction_percent']:+.2f}% [{lo:+.2f},{hi:+.2f}]% |")
    lines+=['','| 开发原查询，32地点R对L | 降误差及95%区间 |','| --- | --- |']
    for endpoint in ['missing/roi/log_mse','missing/roi/bias_squared','full/whole/absrel','missing/roi_supported/centered_log_mse','missing/roi_unsupported/centered_log_mse']:
        v=s['comparisons']['development/original/'+endpoint+'/W_Render-W_Local'];lo,hi=v['interval95_percent'];lines.append(f"| {endpoint} | {v['relative_error_reduction_percent']:+.2f}% [{lo:+.2f},{hi:+.2f}]% |")
    lines+=['','完整地点、种子、条件和指标见[summary.json](scoring/summary.json)。','']
    (OUT/'report_tables.md').write_text('\n'.join(lines))


def main():
    OUT.mkdir(exist_ok=True)
    while not all(done(j) for j in jobs()):
        q=read(OUT/'queue_status.json') if (OUT/'queue_status.json').exists() else {}
        if q.get('failed'):raise RuntimeError('Training failure requires diagnosis; do not score incomplete comparisons')
        write_json(OUT/'finish_status.json',dict(state='waiting_for_six_fits_and_exports',complete=False,updated_epoch=time.time()))
        time.sleep(30)
    while not (OUT/'support_complete.json').exists():
        states=resource.occupancy();idle=[r for r in states if r[1]<=512 and r[2]>=8192 and r[3]<=5]
        spare=[r for r in sorted(states,key=lambda r:(r[3],-r[2])) if r[2]>=8192]
        choices=idle or spare
        if not choices:time.sleep(30);continue
        code=subprocess.run([sys.executable,'-u','-m','experiments.native_coverage_launch','--gpu',str(choices[0][0]),'--cap','2',
            '--admission','8192','--margin','6144','experiments.native_coverage_support'],cwd=ROOT).returncode
        if code not in [0,75]:raise RuntimeError('Source-support evaluation failed')
    for module in ['experiments.native_coverage_score','experiments.native_coverage_checks','experiments.native_coverage_figures']:
        write_json(OUT/'finish_status.json',dict(state=module,complete=False,updated_epoch=time.time()));run(module)
    cost_tables()
    write_json(OUT/'finish_status.json',dict(state='scored_checked_plotted_waiting_actual_review_and_decision',complete=False,
        required_remaining=['actual image review','scientific decision/report','living documentation sync'],updated_epoch=time.time()))
    print('All numerical artifacts ready; actual image review and scientific decision remain.',flush=True)


if __name__=='__main__':main()
