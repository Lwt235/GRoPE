"""Finite CPU postprocessing; scientific interpretation and actual review remain explicit."""
import fcntl
import time

from experiments.learned_projection_data import *
from experiments import learned_projection_score as scoring
from experiments import learned_projection_checks as checks
from experiments import learned_projection_figures as figures


def tables():
    s=read(OUT/'scoring/summary.json');parts=['# learned_projection_pe1 完整数表','',
        '旧数据、来源传感器及共同有效性特权诊断；不是纯RGB成绩。负的相对降误差表示退步。','',
        '|方法|缺失ROI空间log-MSE|缺失ROI原始log-MSE|完整ROI空间log-MSE|完整全图AbsRel|',
        '|---|---:|---:|---:|---:|']
    endpoints=['missing/roi/centered_log_mse','missing/roi/log_mse','full/roi/centered_log_mse','full/whole/absrel']
    for kind in ['parent_L','native_current','P','S','G','old_P_sensor','old_fixed_G_sensor']:
        values=[s['summaries'][e+'/'+kind]['mean'] for e in endpoints]
        parts.append('|'+kind+'|'+'|'.join(f'{v:.9f}' for v in values)+'|')
    parts+=['','## 主终点配对比较','','|比较|相对降误差%|95%区间|三种子%|','|---|---:|---|---|']
    for pair in ['G-P','G-S','S-P','P-parent_L','S-parent_L','G-parent_L','G-old_fixed_G_sensor','P-old_P_sensor']:
        c=s['comparisons']['missing/roi/centered_log_mse/'+pair]
        parts.append(f"|{pair}|{c['relative_error_reduction_percent']:.6f}|{c['interval95_percent']}|{c['per_seed_relative_percent']}|")
    t=read(OUT/'training_summary.json');a=read(OUT/'scoring/independent_recomputation.json')
    parts+=['',f"九次训练保留循环合计{t['retained_loop_worker_hours']:.6f} worker小时，未保存中断工作{t['discarded_loop_worker_hours']:.6f}小时。",
        f"{a['new_predictions']}份新预测独立复算最大差{a['max_abs_metric_difference']:.3g}；继承指标独立标记，不计新增证据。",'',
        '数表不替代图像实际审阅、成本范围及科学取舍。']
    (OUT/'report_tables.md').write_text('\n'.join(parts)+'\n')


def main():
    torch.set_num_threads(4)
    with (OUT/'postprocess.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stage='waiting_for_nine_fits'
        while True:
            q=read(OUT/'queue_status.json') if (OUT/'queue_status.json').exists() else {}
            write_json(OUT/'postprocess_status.json',dict(state=stage,completed_fits=len(q.get('complete',[])),
                expected_fits=9,goal_complete=False,updated_epoch=time.time()))
            if q.get('failed') and not q.get('running'):raise RuntimeError(q['failed'])
            if len(q.get('complete',[]))==9:break
            time.sleep(30)
        for stage,action in [('independent_scoring',lambda:None),('matched_and_PE_checks',checks.main),('figure_generation',figures.main),('tables',tables)]:
            write_json(OUT/'postprocess_status.json',dict(state=stage,goal_complete=False,updated_epoch=time.time()))
            if stage=='independent_scoring':
                rows,audit=scoring.collect();assert audit['complete'] and audit['new_predictions']==864
                folder=OUT/'scoring';folder.mkdir(exist_ok=True)
                write_json(folder/'metrics.json',dict(rows=rows));write_json(folder/'independent_recomputation.json',audit)
                write_json(folder/'summary.json',scoring.summarize(rows))
            else:action()
        write_json(OUT/'postprocess_status.json',dict(state='numerical_checks_and_figures_complete_review_pending',
            goal_complete=False,remaining=['full_query_cost','actual_figure_review','scientific_report_and_documentation'],updated_epoch=time.time()))
        print('All finite numeric postprocessing complete; cost, actual image review and interpretation remain.',flush=True)


if __name__=='__main__':main()
