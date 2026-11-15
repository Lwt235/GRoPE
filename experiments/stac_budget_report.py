"""Complete fixed-budget report/figures, using only completed new metrics."""
import os,time,resource,csv
import numpy as np
import torch
from PIL import Image
from experiments import stac_budget_data as d
from experiments.stream_history_report import n,estimate,table

def main():
 started=time.monotonic();p=d.plan();s=d.read(d.OUT/'scoring/summary.json');c=d.read(d.OUT/'costs.json');cap=d.read(d.OUT/'capacity_lock.json');by=d.read(d.OUT/'scoring/by_place.json');counter=d.read(d.OUT/'scoring/all_counterexamples.json');pairs=d.read(d.OUT/'scoring/pairs.json');selection=d.read(d.OUT/'scoring/selection.json');comp=s['comparisons'];arms=['S','H','W'];keys=['S_vs_H','S_vs_W','H_vs_W'];lines=[]
 for name in ['frames','pairs','selection','by_place','support_strata','all_counterexamples']:
  rows=d.read(d.OUT/'scoring'/f'{name}.json');fields=list(dict.fromkeys(k for r in rows for k in r))
  with (d.OUT/'scoring'/f'{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 lines += ['# stac_native_budget1：真实作者政策、强内容与原生窗口\n',f"2026-11-15，round049完整实施[唯一科学合同]({d.SCIENCE})，SHA `{d.SHA}`。**主S/H重访旋转改善 E_H−E_S={estimate(s['primary'])}°**；S/W净用途和深度代价独立列出。12地点×96帧×三政策全部完成，0训练／数据或权重下载；全部used development，不称fresh、采用、总体成功或自动后继。\n"]
 lines.append(table(['比较（正值误差下降）','重访旋转° [95%配对区间]','全帧AbsRel改善 [区间]','预定旋转flag','明显AbsRel代价flag'],[[k,estimate(comp['revisit_coverage_ge5pct'][k]['relative_rotation_deg']),estimate(comp['all96frames'][k]['absrel']),s['predefined_flags'][k]['practical_rotation'],s['predefined_flags'][k]['obvious_AbsRel_cost']] for k in keys])+'\n')
 lines.append('旋转均值≥1°且配对下界>0、AbsRel增加≥.01且区间支持退步是合同的开发投入标记，不是普适数值门槛、等价或非劣性证明。S/H为主，S/W为原生净用途；仅胜窗口不能建立空间政策的额外价值。政策自身反馈、源位置和相机共同变化，本比较不唯一归因于几何或Gaussian／PE。科学处置交原Astra；[主控制总体目标](../autonomous_research_loop/goal_revision_after_round042/revised_goal.md)及全部历史关闭保持。\n')
 lines.append('S相对H/W的全帧AbsRel改善+0.008912 [0.003580,0.015169]；raw log-MSE改善+0.096687 [0.025322,0.186421]主要来自偏差平方改善+0.097143，去均值结构改善为−0.000456 [−0.000812,−0.000168]。全帧锚旋转／中心及绝对GT几何退步，重访平移／绝对几何区间跨零；这些正负证据均保留，深度辅助不替代主旋转。\n')
 lines += ['## 固定作者、输入和真实计算路径\n',f"固定 Rainzor/STAC commit `{p['author_commit']}`，原CUDA attention及segment merger；没有换Python参考融合。源码归档{d.read(d.OUT/'source_acquisition.json')['archive_bytes']:,}bytes，仅同commit源码，原918Gitblob逐一匹配；仅C++ wrapper header/bindings增加完整状态观察／恢复接口，算法kernel不改。实际现有官方checkpoint加载missing=[]、额外仅未用track_head；1,190,596,120参数／4,762,384,480权重内存bytes。继承权重SHA `{p['model_sha256_inherited']}`，没有重新哈希大权重。[源码身份](source_identity.json)、[真实加载与首帧无压缩对应](load_validation.json)记录兼容边界，不把静态推断冒称加载。\n"]
 lv=d.read(d.OUT/'load_validation.json');lines.append(table(['首固定帧：作者vs本地官方','meanAbs','maxAbs'],[[k,n(v['mean_abs'],9),n(v['max_abs'],9)] for k,v in lv['no_compression_correspondence'].items()])+'\n')
 lines.append('同原始共享权重和BF16路径仍有上述有限精度差；对应检查不是逐位等价保证，也未建立跨后端精度排名。没有把简化体素称完整STAC。本轮已实际运行作者原CUDA merger与默认激活读取；原round047静态未知及定位边界见[源事实报告](../stac_route_review1/report.md)，本轮运行证据按新范围补充，不覆写旧报告。\n')
 lines.append(table(['组','政策','共同读取／相机'],[['S','作者recent4、HH2、首帧；retrieval2、体素.05、buf8／pivot4、阈值.8/.8/.2、segment原合并','全部24global，chunk4，同sourceRoPE、原有效BF16 KV；作者point必算'],['H',f"首帧+recent4+HH{cap['capacity']['H']}，原作者attention colsum／更新，最大合法容量",'无空间库，point省去；其节省可增加容量'],['W',f"首帧+recent{cap['capacity']['W']}完整帧，最大合法窗口",'同原attention输出kernel，无HH／空间库，point省去']])+'\n')
 lines.append('传入recent_size=4；作者STACVoxelKV第109行将self.recent_size日志元数据覆盖成retrieval_size=2，但派生recent_token_size仍为4×1041。按真实作者执行，未为日志数字修政策。原1041=5特殊+1036patch，来源位置与特殊token未删；S使用真实代表count-bias、原相似度排序与几何读取次序，不视为空间均值替代。\n')
 lines.append('所有组kf_every=1，原12地点每处96独立稀疏RGB，不再stride10。每4帧联合处理，最多看到chunk内后3帧；相机对全已到达pose-token前缀重新四轮精化。对目标t和来源s取包含t的chunk提交时同一前缀姿态，深度取自身chunk输出；不用最终96姿态回溯评分。跨时点几何及首帧刚体gauge也取该提交前缀，无GT尺度拟合。此输入权限不同于旧逐帧实验，旧W分数既不复用也不重算。GT从未写入模型／政策／容量。\n')
 lines += ['## 先S成本、再容量、最后GT\n',f"先完成12条S，仅记录预测／成本。B在{cap['locked_utc']}、任何GT评分前锁定：实际S最大峰{cap['S_actual_peak_bytes']:,}bytes按64MiB向上取整，**B={cap['B_bytes']:,}bytes ({cap['B_bytes']/2**30:.6f}GiB)≤16GiB**。原旧B不被替换；这是固定96范围的预算点，不是任意长流全局有界部署。\n"]
 lines.append(table(['组','HH／recent容量','hot帧','最大reachable reserve','合法生命周期界GiB','实际历史峰GiB','B未用GiB'],[[a,cap['capacity'][a],cap['chosen'][a]['hot_frames'],cap['chosen'][a]['reserved_frames'],n(cap['chosen'][a]['legal_bound_bytes']/2**30,6),n(c['history_actual_peak_by_arm'][a]/2**30,6),n(c['unused_B_by_arm'][a]/2**30,6)] for a in ['H','W']])+'\n')
 lines.append(f"S实际峰{c['history_actual_peak_by_arm']['S']/2**30:.6f}GiB。H/W完整历史条件={cap['normal_full_history_H_W_identical']}；若两者保留全部96，HH没有淘汰选择，不能以假想最后gather压小普通对照。即使同输出，H仍维护attention scores，额外成本保留；本轮两种原政策分别实际执行，未把别名虚报为额外独立证据。\n")
 lines.append('生命周期三组共同固定首地点prefix8完整恢复。H/W物理buffer4→8→有限96可达最大reserve，不改变逻辑容量、HH打分／选择、KV精度或来源。一次扩展逐tensor释放旧buffer，真实旧／新副本重叠计入；全部可达整数与hot96不触发prune的非单调边界前GT计算。旧未使用的v1全96假想恢复上界仅压控制不公平，已在控制启动前撤销并保留版本。不设无用horizon填充，不把恢复假设当科学条件。后续外部资源礼让若发生仍独立遵守真实B，不能免费恢复或暗减容量。\n')
 lines.append('历史计费含GPU／CPU完整untyped backing、原全RGB可读文件、原作者CPU images/depth/conf及S point archive、pose-token view背后的完整聚合输出、前缀相机快照、索引／score／count／位置／体素／overflow／retrieval、CUDA原生workspace、Python对象、checkpoint与原子输出／chunk数值备份重叠。真实别名计一次；权重和当前推理激活另报VRAM，历史专属workspace不免费。作者所有CPU archive在流结束前保留且计费；最早9个永久文件保留全archive，后续v2只保留科学depth／prefixposes／costs，均不反馈未来预测。\n')
 lines += ['## 主端点、NA与逐地点\n','原固定63远端目标t=33…95，s≠0且t−s≥32，原GT min-z共视配对及5%重访身份不变。地点先平均，再对有定义地点等权；5000次原12地点ID配对bootstrap、seed20261114，NA不填0，保留所有留一地点结果。原selected-pair独占和all-far-union独占均扣原首帧+recent7参考支持；仅分层描述，不偷换为本轮有利窗口。\n']
 rows=[]
 for i,m in enumerate(p['catalog']):
  z={a:next(r for r in by if r['id']==m['id'] and r['arm']==a and r['group']=='revisit_coverage_ge5pct') for a in arms};rows.append([m['place'],z['S']['records'],*[n(z[a]['relative_rotation_deg']) for a in arms],n(s['primary']['per_place'][i])])
 lines.append(table(['地点','重访/63','S°','H°','W°','H−S°'],rows)+'\n');cov=s['coverage'];loo=[v for v in s['primary']['leave_one_place_out_means'] if v is not None]
 lines.append(f"全1152帧、756远端目标，{cov['revisit_targets']}重访／{cov['places_with_revisits']}有定义地点；{12-cov['places_with_revisits']}地点主NA、{cov['zero_selected_covisibility']}零共视、{cov['zero_selected_exclusive']}独占零覆盖均保留。主留一地点均值范围{n(min(loo))}…{n(max(loo))}°；GT近零平移仅方向辅助NA，{sum(r['GT_translation_near_zero'] for r in pairs if r['arm']=='S')}/756，不删旋转。真实远端时距{n(min(r['elapsed_seconds'] for r in selection),2)}…{n(max(r['elapsed_seconds'] for r in selection),2)}秒。\n")
 lines.append('![固定比较及逐地点](figures/primary.png)\n')
 lines += ['## 全部原生辅助、深度与支持\n']
 fm=['trajectory_rotation_anchor_deg','trajectory_center_error_raw_m','absolute_worldpoint_GT_error_raw_m','absrel','log_mse','centered_log_mse','bias_squared','RMSE_m','failure'];rows=[]
 for k in fm:rows.append([k,*[n(np.mean([r[k] for r in by if r['arm']==a and r['group']=='all96frames' and r[k] is not None])) for a in arms],*[estimate(comp['all96frames'][key][k]) for key in keys]])
 lines.append(table(['全96帧地点等权','S','H','W','H−S [CI]','W−S [CI]','W−H [CI]'],rows)+'\n')
 lines.append('raw log-MSE=去均值结构+偏差平方；AbsRel、RMSE、坏像素及真实失败并列。无GT尺度或仿射对齐。跨时一致性与绝对GT几何、GT采样discrepancy同时保留，避免塌缩伪装一致性。辅助正例不替换主旋转，物理尺度准确性／PE／各向异性未由本轮证明。\n')
 rows=[]
 for group in ['all63far_targets','revisit_coverage_ge5pct','nonrevisit_lt5pct']:
  for k in ['relative_rotation_deg','relative_translation_direction_deg','relative_translation_vector_error_raw_m','cross_time_point_consistency_raw_m','paired_absolute_GT_geometry_error_raw_m','failure']:rows.append([group,k,*[estimate(comp[group][key][k]) for key in keys]])
 lines.append(table(['目标集合','辅助指标','S/H','S/W','H/W'],rows)+'\n');lines.append('![raw、去均值与偏差](figures/depth.png)\n')
 strata=d.read(d.OUT/'scoring/support_strata.json');sp=d.read(d.OUT/'scoring/support_strata_paired.json');rows=[]
 for scope in ['selected_pair_exclusive','all_far_union_exclusive']:
  for st in ['0','(0,5%]','(5,20%]','>20%','undefined']:
   z=next(r for r in strata if r['scope']==scope and r['stratum']==st and r['revisit_only'] and r['arm']=='S');rr=[next(r for r in sp if r['scope']==scope and r['stratum']==st and r['revisit_only'] and r['candidate']==a and r['baseline']==b)['effects']['relative_rotation_deg'] for a,b in [('S','H'),('S','W'),('H','W')]];rows.append([scope,st,z['records'],z['places'],*[estimate(v) for v in rr]])
 lines.append(table(['原支持定义（重访）','层','记录','地点','S/H°','S/W°','H/W°'],rows)+'\n')
 lines.append('全部756与重访子集每层全部指标／配对区间在support_strata_paired.json，0／undefined不删。不据层内相关调整阈值或作因果结论。\n')
 for a,b in [('S','H'),('S','W'),('H','W')]:
  rr=[r for r in counter if r['candidate']==a and r['baseline']==b and r['revisit']][:5];lines.append(table([f'{a}/{b}最不利重访行','t/s','覆盖','独占','秒','改善°'],[[r['id'],f"{r['t']}/{r['s']}",n(r['coverage'],3),n(r['exclusive_fraction'],3),n(r['elapsed_seconds'],2),n(r['rotation_improvement_deg'])] for r in rr])+'\n')
 lines.append('全部正负／零覆盖／非重访原行保留all_counterexamples.csv。反例三张depth共用0至三图pooled99%分位色阶，仅展示相同目标自己的chunk深度；旋转值来自完整提交前缀姿态，不由颜色推断。![主比较最大反例](figures/counterexample.png)\n')
 lines += ['## 完整形成／读取／恢复成本与失败\n']
 rows=[]
 for a in arms:
  ww=[r for r in c['worker_receipts'] if r['arm']==a];rows.append([a,len(ww),n(sum(r['elapsed_seconds'] for r in ww),4),*[n(c['component_seconds_including_discarded'][a][k],4) for k in ['backbone_and_heads_seconds','camera_prefix_seconds','position_seconds','retrieval_seconds','prune_merge_seconds','chunk_wall_seconds']]])
 lines.append(table(['组','冷worker数','load至wait秒','backbone+heads','全前缀camera','几何／位置','retrieval','prune／merge','完整chunk秒'],rows)+'\n')
 lines.append('完整query是4张当前RGB到原生backbone、depth／S-point、全前缀四轮camera、实际几何／检索／淘汰、CPU作者输出及chunk数值保存。缓存attention计时不能代替完整query；camera前缀增长、首形成、RGB读取、CPU-GPU-DATA传输和恢复均非免费。组件为同步envelope，嵌套方法计时不可相加；完整Popen→actualwait包含import／加载／失败／I-O。实际共享调度／外部保护由原worker准入收据保存，不把非独占混合耗时作方法速度因果排名。单独PCIe精确字节／kernel-only时间未额外profile，未知不伪造0。\n')
 calls=d.read(d.OUT/'call_accounting.json');lines.append(table(['实际调用／状态','数量'],[[k,v] for k,v in calls.items() if isinstance(v,(int,float))])+'\n')
 lines.append('工程失败保留：①初次S首8帧恢复缺layer_idx，后状态不可用，2个chunk／8帧discard；失败新chunk尝试1次，完整修复后重建首流。②S/fidelity09已到96，末尾可选overflow getter空Tensor转标量失败，数值未提交且无可恢复状态，24个chunk／96帧discard；9条既完整S保持原样。两者所有调用／worker费用计入，不当科学阴性。修复002改可选count为unknown／NA，最小数值chunk备份和永久数值先提交。成功首流恢复保留全部tensor固定样本／RNG／24层index，完成chunk不重复；H/W同prefix8实测恢复。原作者recent元数据覆盖未修。最早9S永久文件为含作者archive的旧格式，其余为科学数值最小格式；实时推理archive均相同保留并计B，实际不同归档I/O成本留账，不将混合版本wall作纯算法速度因果排名。\n')
 lines.append(f"所有GPU计费上界 **{c['all_GPU_worker_hours_upper_bound']:.6f}worker小时≤8**；其中preflight含三组首地点真实恢复及辅助上界 **{c['preflight_GPU_worker_seconds_upper_bound']/3600:.6f}小时≤1**。管理worker实际Popen→wait{c['measured_manager_worker_seconds']:.6f}s；全部manager阶段wall {c['all_manager_wall_seconds']:.3f}s，首操作至成本汇总混合wall {c['round_mixed_wall_seconds_at_account']:.3f}s（含准备／工程／等待），实际兼容阶段 {c['source_compatibility_elapsed_seconds']:.3f}s；辅助960s为缺完整进程wall时的保守预算预约，不能称实际消耗，loadcheck body35.254485s另记。{c['actual_worker_processes']}GPU子进程全部actualwait，exit码{c['all_worker_exit_codes']}。原源码／兼容／构建90混合分钟内完成真实加载、CUDA及首完整S，未扩deadline。\n")
 lines.append(f"单worker allocated峰{c['peak_GPU_allocated_bytes']/2**30:.6f}GiB≤40，RSS{c['peak_RSS_worker_bytes']/2**30:.6f}GiB。DATA实际采样峰{c['DATA_peak_sampled_bytes']/2**30:.6f}GiB、保守完整原子checkpoint并发界{c['DATA_conservative_bound_bytes']/2**30:.6f}GiB≤48，永久最终{c['DATA_final_bytes']/2**30:.6f}GiB；成本汇总时项目源码／构建与已产物{c['project_bytes']/2**30:.6f}GiB≤2，最终以terminal_record当前字节为准。只用实时安全CUDA卡，不以余显存当空闲算力；保护他人进程、旧数据和GLIBC。\n")
 lines.append('![完整成本与历史峰](figures/costs.png)\n')
 z=s['independent'];lines += ['## 独立复算、范围与交接\n',f"新三组全部3456提交帧保留，其中有有效GT且预测有限的{z['unique_depth_records']}depth帧NumPy/Torch独立指标最大差{z['NumPy_Torch_depth_metric_max_difference']:.3g}、FP64resize差{z['NumPy_Torch_FP64_resize_max_difference']:.3g}；NumPy／官方Torch相机解码差{z['NumPy_official_Torch_pose_decode_max_difference']:.3g}、NumPy／SciPy旋转差{z['NumPy_SciPy_relative_rotation_max_difference_degrees']:.3g}°。原标签身份继承，0旧分数重计算、0评分模型调用。预测坏值保留显式failure，GT零支持NA不变。\n",'O2/O3真实作者空间状态与实际总byte、O4原生精度—成本为核心；不是完整3DGS、全局永久有界或O5/O6成功。C1共同chunk4／前缀camera权限、C2/C3三种原政策、C4 sourceRoPE、C5原旧稀疏输入、C6冻结均为临时配置。旧C/I/G、U/C、XYZ／PE关闭及正负保持。全部used，未来采用仍需真正未用确认；本报告只交事实，不自选后继。\n','完整证据：[唯一handoff](implementation_handoff.json)、[manifest](manifest.json)、[全部指标／flags](scoring/summary.json)、[每帧](scoring/frames.csv)、[逐对](scoring/pairs.csv)、[逐地点](scoring/by_place.csv)、[支持配对](scoring/support_strata_paired.json)、[所有反例](scoring/all_counterexamples.csv)、[字节](history_bytes.json)、[成本](costs.json)、[调用](call_accounting.json)、[终态](terminal_record.json)、[实际图审](figure_review.json)。\n']
 (d.OUT/'report.md').write_text(('\n'.join(lines)+'\n').replace('(figures/',f'({d.OUT}/figures/'))
 make_figures(p,s,c,cap,by,counter)
 d.write(d.OUT/'CPU_report_runtime.json',dict(complete=True,PID=os.getpid(),seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized(),new_model_calls=0));print(dict(report_complete=True,PID=os.getpid()),flush=True)

def make_figures(p,s,c,cap,by,counter):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':145})
 out=d.OUT/'figures';out.mkdir(exist_ok=True);colors=['#ad493a','#316b91','#6b7772'];keys=['S_vs_H','S_vs_W','H_vs_W']
 fig,ax=plt.subplots(1,2,figsize=(12,4),gridspec_kw={'width_ratios':[1,2]});vals=[s['comparisons']['revisit_coverage_ge5pct'][k]['relative_rotation_deg'] for k in keys]
 for i,v in enumerate(vals):ax[0].errorbar(v['mean'],i,xerr=[[v['mean']-v['interval95'][0]],[v['interval95'][1]-v['mean']]],fmt='o',color=colors[i],capsize=3)
 ax[0].set(yticks=range(3),yticklabels=keys,xlabel='Rotation improvement (deg), 95% paired CI');ax[0].axvline(0,color='gray',lw=.8);a=s['primary'];ax[1].bar(range(12),[np.nan if v is None else v for v in a['per_place']],color=colors[0]);ax[1].set(xticks=range(12),xticklabels=[m['place'] for m in p['catalog']],ylabel='H minus S (deg)');ax[1].tick_params(axis='x',rotation=65);ax[1].axhline(0,color='gray',lw=.8);fig.suptitle('Submitted-prefix revisit rotation | all 12 places retained; undefined = gaps');fig.tight_layout();fig.savefig(out/'primary.png');plt.close(fig)
 fig,ax=plt.subplots(1,4,figsize=(13,3.6))
 for j,k in enumerate(['absrel','log_mse','centered_log_mse','bias_squared']):
  for i,a in enumerate(['S','H','W']):ax[j].bar(i,np.mean([r[k] for r in by if r['arm']==a and r['group']=='all96frames']),color=colors[i])
  ax[j].set(title=k,xticks=range(3),xticklabels=['S','H','W'])
 fig.suptitle('All 96 frames per place | raw depth, no fitted scale');fig.tight_layout();fig.savefig(out/'depth.png');plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(11,3.8));components=['backbone_and_heads_seconds','camera_prefix_seconds','position_seconds','retrieval_seconds','prune_merge_seconds'];bottom=np.zeros(3)
 for k in components:
  vv=np.array([c['component_seconds_including_discarded'][a][k]/(12*96) for a in ['S','H','W']]);ax[0].bar(range(3),vv,bottom=bottom,label=k.replace('_seconds',''));bottom+=vv
 ax[0].set(xticks=range(3),xticklabels=['S','H','W'],ylabel='Component seconds / formal submitted frame');ax[0].legend(fontsize=7);ax[1].bar(range(3),[c['history_actual_peak_by_arm'][a]/2**30 for a in ['S','H','W']],color=colors);ax[1].axhline(cap['B_bytes']/2**30,color='black',ls='--',label='common B');ax[1].set(xticks=range(3),xticklabels=['S','H','W'],ylabel='All actual history bytes (GiB)');ax[1].legend();fig.suptitle('All spent costs include discarded chunks; cold loading/recovery also in worker ledger');fig.tight_layout();fig.savefig(out/'costs.png');plt.close(fig)
 worst=next(r for r in counter if r['candidate']=='S' and r['baseline']=='H' and r['revisit']);meta=next(m for m in p['catalog'] if m['id']==worst['id']);fig,ax=plt.subplots(1,5,figsize=(14,3))
 for j,t in enumerate([worst['s'],worst['t']]):ax[j].imshow(Image.open(meta['frames'][t]['rgb_path']));ax[j].set_title(('source ' if j==0 else 'target ')+str(t))
 dep=[d.load(d.DATA/'predictions'/f"{a}_{worst['id']}.pt")['depth'][worst['t']].numpy() for a in ['S','H','W']];vmax=float(np.quantile(np.concatenate([x.ravel() for x in dep]),.99))
 for j,(a,x) in enumerate(zip(['S','H','W'],dep),2):ax[j].imshow(x,vmin=0,vmax=vmax,cmap='magma');ax[j].set_title(a+' raw depth')
 for z in ax:z.axis('off')
 fig.suptitle(f"Least favorable S/H revisit: {worst['id']}, H-S={worst['rotation_improvement_deg']:.3f} deg | own chunk depth");fig.tight_layout();fig.savefig(out/'counterexample.png');plt.close(fig)
if __name__=='__main__':main()
