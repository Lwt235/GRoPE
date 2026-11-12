"""CPU postprocessing; scientific decision and actual image review remain manual."""
import fcntl
import subprocess
import sys
from experiments.native_receiver_data import *
from experiments.native_receiver_orchestrate import jobs,done


def main():
    with (OUT/'postprocess.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (OUT/'postprocess.pid').write_text(str(__import__('os').getpid())+'\n')
        while not (all(done(j) for j in jobs()) and (OUT/'persistence_complete.json').exists()):
            write_json(OUT/'postprocess_status.json',dict(state='waiting_for_fixed_fits_exports_and_cost',
                goal_complete=False,updated_epoch=time.time()))
            time.sleep(30)
        for module in ['experiments.native_receiver_score','experiments.native_receiver_checks','experiments.native_receiver_figures']:
            print('CPU '+module,flush=True);subprocess.run([sys.executable,'-u','-m',module],cwd=ROOT,check=True)
        s=read(OUT/'scoring/summary.json');lines=['# native_receiver_adaptation1：完整固定数表','',
            '三种子／12旧开发地点；误差越低越好。F为冻结原生接收、A为共同适配；F开发行继承旧结果。','',
            '| 方法 | 开发缺失ROI空间 | 开发缺失ROI原始 | 完整全图AbsRel | 训练缺失ROI空间 |','| --- | ---: | ---: | ---: | ---: |']
        for k in ['F_Local','F_Global','F_Render','A_Local','A_Global','A_Render']:
            values=[s['summaries'][key+'/'+k]['mean'] for key in ['development/missing/roi/centered_log_mse','development/missing/roi/log_mse','development/full/whole/absrel','train/missing/roi/centered_log_mse']]
            lines.append('| '+k+' | '+' | '.join(f'{v:.9f}' for v in values)+' |')
        lines+=['','| 开发主比较 | 相对降误差% | 地点/种子95%区间 | 三种子% |','| --- | ---: | --- | --- |']
        for pair in ['A_Render-A_Local','A_Render-A_Global','A_Global-A_Local','A_Render-A_Render_shuffle_binding']:
            v=s['comparisons']['development/missing/roi/centered_log_mse/'+pair];lines.append(f"| {pair} | {v['relative_error_reduction_percent']:.6f} | {v['interval95_percent']} | {v['per_seed_relative_percent']} |")
        lines+=['','配对交互（绝对误差单位，负值有利）：','']
        for control in ['Local','Global']:
            v=s['interactions']['development/missing/roi/centered_log_mse/Render-'+control];lines.append(f"- R−{control}: {v['mean_absolute']:.9f}, 95% {v['interval95_absolute']}, seeds {v['per_seed_absolute']}")
        (OUT/'report_tables.md').write_text('\n'.join(lines)+'\n')
        write_json(OUT/'postprocess_status.json',dict(state='scores_checks_figures_complete_actual_review_and_report_pending',
            goal_complete=False,remaining=['actual_figure_review','scientific_report_decision_documentation','final_process_audit'],updated_epoch=time.time()))


if __name__=='__main__':main()
