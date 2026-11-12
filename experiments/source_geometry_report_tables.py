"""Final numerical tables, quality manipulation and cost boundaries; no automated method verdict."""
import numpy as np
from experiments.source_geometry_data import *


def mean_defined(values):
    a=[x for x in values if x is not None];return float(np.mean(a)) if a else None

def quality_summary():
    p=plan();out={};place_values={}
    for mode in MODES:
        for split in ['training','development']:
            metas=p['training_catalog'] if split=='training' else p['catalog']
            for view in ['source','query']:
                if view=='query' and split=='training':continue
                metrics={}
                for stage in ['initial','final']:
                    for name in ['rgb_mse','covered_fraction_of_valid','whole/log_mse','whole/centered_log_mse','whole/absrel','covered/log_mse','covered/centered_log_mse','covered/absrel']:
                        by_place={}
                        for meta in metas:
                            rows=read(branch_out(mode)/'diagnostics'/f'{meta["id"]}.json')['records'];rows=[r for r in rows if r['view']==view]
                            values=[]
                            for r in rows:
                                if '/' in name:
                                    region,metric=name.split('/');value=r[stage]['depth_metrics'][region][metric]
                                else:value=r[stage][name]
                                values.append(value)
                            by_place[meta['place']]=mean_defined(values)
                        key=stage+'/'+name;metrics[key]=dict(mean=mean_defined(by_place.values()),per_place=by_place,estimable_places=sum(v is not None for v in by_place.values()),all_places=len(metas))
                for name in ['all_patch_mse','covered_patch_mse','covered_patch_cosine','source_mean_baseline_all_patch_mse','source_mean_baseline_cosine']:
                    by_place={meta['place']:mean_defined([r['features'][name] for r in read(branch_out(mode)/'diagnostics'/f'{meta["id"]}.json')['records'] if r['view']==view]) for meta in metas}
                    metrics['feature/'+name]=dict(mean=mean_defined(by_place.values()),per_place=by_place)
                out[f'{mode}/{split}/{view}']=metrics
    # No seed resampling: there is only one saved field per place and geometry.
    rng=np.random.default_rng(p['bootstrap']['seed']);n=p['bootstrap']['replicates'];places=[r['place'] for r in p['catalog']];sample=rng.integers(0,12,(n,12));effects={}
    for key in ['final/whole/log_mse','final/whole/centered_log_mse','final/whole/absrel','feature/all_patch_mse']:
        a=np.array([out['pred/development/query'][key]['per_place'][x] for x in places]);b=np.array([out['sensor/development/query'][key]['per_place'][x] for x in places])
        values=(a[sample]-b[sample]).mean(1);effects[key]=dict(pred=float(a.mean()),sensor=float(b.mean()),absolute_error_reduction=float((a-b).mean()),interval95_absolute_error_reduction=np.percentile(values,[2.5,97.5]).tolist(),positive_means_sensor_better=True,units='place-only,12old places, no seed replication of fields')
    result=dict(complete=True,summaries=out,development_query_effects=effects,scope='Source quality manipulation diagnostic; targets used for evaluation only, all places retained, covered-only denominators separate. Mean feature baseline calibrates an auxiliary proxy, not task utility.')
    write_json(OUT/'quality_summary.json',result);return result


def main():
    p=plan();s=read(OUT/'scoring/summary.json');audit=read(OUT/'scoring/independent_recomputation.json');assert audit['new_predictions']==2880 and s['complete']
    quality=quality_summary();cost=read(OUT/'cost/complete.json');training=read(OUT/'training_summary.json');assert training['complete']
    lines=['# source_geometry_control1完整数表','', '全部旧地点，来源传感器与共同有效性是额外诊断信息；以下不是纯RGB方法成绩。','', '## 原生输出','', '|组别|缺失ROI空间log-MSE|缺失ROI原始log-MSE|完整ROI空间log-MSE|完整全图AbsRel|','|---|---:|---:|---:|---:|']
    kinds=['parent_L','P_pred','G_pred','P_sensor','G_sensor']
    endpoints=['missing/roi/centered_log_mse','missing/roi/log_mse','full/roi/centered_log_mse','full/whole/absrel']
    for k in kinds:lines.append('|'+k+'|'+'|'.join(f'{s["summaries"][e+"/"+k]["mean"]:.8f}' for e in endpoints)+'|')
    lines+=['','## PE增量与来源质量变化','', '|比较|相对降误差%|95%区间|逐种子%|','|---|---:|---|---|']
    for pair in ['G_pred-P_pred','G_sensor-P_sensor','P_sensor-P_pred','G_sensor-G_pred','P_pred-parent_L','G_pred-parent_L','P_sensor-parent_L','G_sensor-parent_L']:
        r=s['comparisons']['missing/roi/centered_log_mse/'+pair]
        lines.append(f'|{pair}|{r["relative_error_reduction_percent"]:.4f}|{r["interval95_percent"]}|{r["per_seed_relative_percent"]}|')
    interaction=s['interactions']['missing/roi/centered_log_mse'];lines +=['',f'主交互（绝对误差差）：{interaction["interaction_absolute"]:.10f}，95%区间{interaction["interval95_absolute"]}；逐种子{interaction["per_seed_interaction_absolute"]}。',f'相对PE降误差的交互：{interaction["interaction_relative_percentage_points"]:.4f}个百分点，区间{interaction["interval95_relative_percentage_points"]}。','', '## 开发查询场景质量','', '|条件|最终全图深度log-MSE|最终空间深度log-MSE|覆盖内AbsRel|有效像素覆盖|特征全patchMSE|来源均值特征MSE|','|---|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        r=quality['summaries'][mode+'/development/query'];keys=['final/whole/log_mse','final/whole/centered_log_mse','final/covered/absrel','final/covered_fraction_of_valid','feature/all_patch_mse','feature/source_mean_baseline_all_patch_mse']
        lines.append('|'+mode+'|'+'|'.join(f'{r[k]["mean"]:.8f}' for k in keys)+'|')
    lines+=['','覆盖条件与全图取样不同，完整分母见quality_summary.json和逐场景diagnostics。无覆盖来源单位填值保留于全图评分。来源均值特征是辅助代理参照，不是新任务模型。','', '## 成本','', '|条件|预处理来源张量→保存场景秒（两训练片段）|P/G完整暖查询中位ms|','|---|---|---|']
    for mode in MODES:
        construction=[r['prepared_source_to_saved_field_seconds'] for r in cost['construction'] if r['mode']==mode]
        query={k:1000*np.median([t for r in cost['queries'] if r['mode']==mode and r['kind']==k for t in r['total_query_seconds']]) for k in KINDS}
        lines.append(f'|{mode}|{construction}|{query}|')
    lines+=['',cost['construction_scope'],'',cost['query_scope'],'',f'保留优化循环{training["retained_loop_worker_hours"]:.6f} worker小时，另有未保存中断工作{training["discarded_loop_worker_hours"]:.6f}小时。','', f'新输出{audit["new_predictions"]}份，独立NumPy最大指标差{audit["max_abs_metric_difference"]:.3g}。继承父轮指标{audit["inherited_parent_rows"]}行，单列来源，不计新增独立证据。','', '本文件只生成数值表；质量操纵是否有效、原生用途和PE增量的科学取舍需结合图表实际审阅后单独记录。']
    (OUT/'report_tables.md').write_text('\n'.join(lines)+'\n')
    inputs=['scoring/summary.json','scoring/independent_recomputation.json','quality_summary.json','cost/complete.json','training_summary.json']
    write_json(OUT/'report_tables_identity.json',dict(complete=True,table_sha256=sha256(OUT/'report_tables.md'),inputs_sha256={n:sha256(OUT/n) for n in inputs},scientific_decision_automated=False))
    print('Final numerical tables and quality summary ready; scientific review pending.',flush=True)

if __name__=='__main__':main()
