"""Finite CPU export verification, scoring and plots after nine fixed fits."""
import fcntl
import time

from experiments.rendered_feature_data import *
from experiments import rendered_feature_score as scoring
from experiments import rendered_feature_checks as checks
from experiments import rendered_feature_figures as figures


def tables():
    s=read(OUT/'scoring/summary.json')
    lines=['# rendered_feature_native1 完整数表','',
        '旧数据、来源传感器/共同有效性、给定相机、静态场景、冻结骨干诊断；不提供PE增量或纯RGB结论。','',
        '|方法|缺失ROI空间log-MSE|缺失ROI原始log-MSE|完整ROI空间log-MSE|完整全图AbsRel|',
        '|---|---:|---:|---:|---:|']
    endpoints=['missing/roi/centered_log_mse','missing/roi/log_mse','full/roi/centered_log_mse','full/whole/absrel']
    for kind in ['Local','Global','Render','Render_shuffle_binding','parent_L','native_current','previous_P','previous_S','previous_G']:
        values=[s['summaries'][e+'/'+kind]['mean'] for e in endpoints]
        lines.append('|'+kind+'|'+'|'.join(f'{v:.9f}' for v in values)+'|')
    lines+=['','## 主终点配对比较','','正的相对降误差表示改善。旧接口比较仅为背景，打乱为依赖诊断。','',
        '|比较|相对降误差%|95%区间|三种子%|','|---|---:|---|---|']
    for pair in ['Render-Local','Render-Global','Global-Local','Render-Render_shuffle_binding',
                 'Local-parent_L','Render-previous_P','Render-native_current']:
        c=s['comparisons']['missing/roi/centered_log_mse/'+pair]
        lines.append(f"|{pair}|{c['relative_error_reduction_percent']:.6f}|{c['interval95_percent']}|{c['per_seed_relative_percent']}|")
    timing=read(OUT/'training_summary.json');audit=read(OUT/'scoring/independent_recomputation.json')
    lines+=['',f"九次保留训练循环{timing['retained_loop_worker_hours']:.6f} worker小时；未保存中断工作{timing['discarded_loop_worker_hours']:.6f}小时另计。",
        f"{audit['new_predictions']}份新预测全部独立复算，最大差{audit['max_abs_metric_difference']:.3g}。继承行不计新增或新确认。",'',
        '数表不代替图像实际审阅、成本范围和科学判断。']
    (OUT/'report_tables.md').write_text('\n'.join(lines)+'\n')


def main():
    torch.set_num_threads(4)
    with (OUT/'postprocess.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            markers=[OUT/'evaluation'/f'{k}_{s}.json' for s in SEEDS for k in KINDS]
            markers += [OUT/'interventions'/f'Render_{s}.json' for s in SEEDS]
            complete=sum(path.exists() and read(path).get('complete',False) for path in markers)
            write_json(OUT/'postprocess_status.json',dict(state='waiting_for_all_fits_and_interventions',
                complete_export_files=complete,expected_export_files=12,goal_complete=False,updated_epoch=time.time()))
            if complete==12:break
            q=read(OUT/'queue_status.json') if (OUT/'queue_status.json').exists() else {}
            if q.get('failed') and not q.get('running'):raise RuntimeError(q['failed'])
            time.sleep(20)
        for name,action in [('matched_checks',checks.main),('independent_scores',scoring.main),
                            ('tables',tables),('figures',figures.main)]:
            write_json(OUT/'postprocess_status.json',dict(state=name,goal_complete=False,updated_epoch=time.time()))
            action()
        write_json(OUT/'postprocess_status.json',dict(state='scores_and_figures_complete_actual_review_pending',
            goal_complete=False,remaining=['actual_figure_review','scientific_report_and_documentation'],
            cost_complete=(OUT/'cost.json').exists(),updated_epoch=time.time()))
        print('Finite postprocessing complete; actual review and scientific decision remain.',flush=True)


if __name__=='__main__':main()
