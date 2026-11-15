"""CPU figures, tabular exports and full report for the fixed finite contract."""
import csv,os,time,resource
import numpy as np
import torch
from experiments import stream_merge_data as d
from experiments.stream_history_report import n,estimate,table
KEYS=['G_vs_I','G_vs_C','I_vs_C','C_vs_W','I_vs_W','G_vs_W']
ARMS=['W','C','I','G']
COLORS=['#727785','#db9e29','#377da9','#20795c']
def figures():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
 p=d.plan();s=d.read(d.OUT/'scoring/summary.json');c=d.read(d.OUT/'costs.json');line=d.read(d.OUT/'lineage_validation.json');figs=d.OUT/'figures';figs.mkdir(exist_ok=True)
 for name in ['frames','pairs','selection','by_place','support_strata','all_counterexamples']:
  rows=d.read(d.OUT/'scoring'/f'{name}.json');fields=list(dict.fromkeys(k for r in rows for k in r))
  with (d.OUT/'scoring'/f'{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 fig,ax=plt.subplots(1,2,figsize=(13.5,5.8))
 for i,k in enumerate(KEYS):
  z=s['comparisons']['revisit_coverage_ge5pct'][k]['relative_rotation_deg'];v=z['mean'];lo,hi=z['interval95'];ax[0].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=COLORS[(i+1)%4],capsize=4)
 for i,k in enumerate(KEYS[:3]):
  vals=s['comparisons']['revisit_coverage_ge5pct'][k]['relative_rotation_deg']['per_place'];ax[1].bar(np.arange(12)+(i-1)*.25,[np.nan if v is None else v for v in vals],width=.24,label=k,color=COLORS[i+1])
  if i==0:
   for j,v in enumerate(vals):
    if v is None:ax[1].text(j,0,'NA',ha='center',fontsize=7)
 ax[0].axvline(0,color='.5');ax[0].axvline(1,color='.75',ls=':');ax[0].set(yticks=range(6),yticklabels=[k+(' (primary)' if i==0 else '') for i,k in enumerate(KEYS)],xlabel='Rotation error improvement (degrees)',title='Fixed revisits / paired 95% place intervals');ax[0].invert_yaxis();ax[1].axhline(0,color='.5');ax[1].set(xticks=range(12),xticklabels=[m['place'] for m in p['catalog']],ylabel='Error improvement (degrees)',title='All12places retained; no fresh data');ax[1].tick_params(axis='x',rotation=65);ax[1].legend(fontsize=8);fig.tight_layout();fig.savefig(figs/'primary.png');plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(13,5.3))
 for i,k in enumerate(KEYS):
  z=s['comparisons']['all96frames'][k]['absrel'];v=z['mean'];lo,hi=z['interval95'];ax[0].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4,color=COLORS[(i+1)%4])
 for i,k in enumerate(['G_vs_I','G_vs_C','G_vs_W']):
  for j,metric in enumerate(['log_mse','centered_log_mse','bias_squared']):
   z=s['comparisons']['all96frames'][k][metric];v=z['mean'];lo,hi=z['interval95'];ax[1].errorbar(v,j+(i-1)*.17,xerr=[[v-lo],[hi-v]],fmt='o',label=k if j==0 else None,capsize=3,color=COLORS[i+1])
 ax[0].axvline(0,color='.5');ax[0].axvline(-.01,color='.75',ls=':');ax[0].set(yticks=range(6),yticklabels=KEYS,xlabel='AbsRel improvement (positive helps)',title='All96frames; raw depth cost');ax[0].invert_yaxis();ax[1].axvline(0,color='.5');ax[1].set(yticks=range(3),yticklabels=['raw log-MSE','centered log-MSE','bias squared'],xlabel='Error improvement',title='No GT fitted scale / structure and bias separate');ax[1].invert_yaxis();ax[1].legend(fontsize=8);fig.tight_layout();fig.savefig(figs/'depth.png');plt.close(fig)
 fig,ax=plt.subplots(2,2,figsize=(12.5,8));components=['RGB_read_transfer_seconds','native_aggregator_seconds','native_camera_depth_seconds','full_update_seconds'];labels=['RGB read/transfer','native aggregator/read','camera4 + depth','full geometry/merge/update'];bottom=np.zeros(4)
 for key,label in zip(components,labels):
  vals=[sum(r['components_seconds'][key] for r in c['per_stream'] if r['policy']==a)/sum(r['frames'] for r in c['per_stream'] if r['policy']==a) for a in ARMS];ax[0,0].bar(ARMS,vals,bottom=bottom,label=label);bottom+=vals
 ax[0,0].set(ylabel='Seconds / complete frame',title='Actual shared formal pipeline (not exclusive ranking)');ax[0,0].legend(fontsize=7)
 times=[c['first_fixed_case_times'][a] for a in ARMS];ax[0,1].bar(ARMS,times,color=COLORS);ax[0,1].set(ylabel='Worker seconds / fixed96frames',title='First-place single-flow costs incl. failure/recovery')
 size=[p['state_copy_bound_bytes']]*3+[p['history_work_bound_bytes'],p['RGB_readable_batch_backing_bytes']];left=0
 for value,label in zip(size,['GPU state','CPU copy','disk atomic','workspace','readable RGB']):
  ax[1,0].barh(['C / I / G'],value/2**20,left=left,label=label);left+=value/2**20
 ax[1,0].axvline(d.B/2**20,color='.5',ls=':',label='B ceiling');ax[1,0].set(xlabel='MiB of budgeted history chain',title=f'Total {p["total_history_bound_bytes"]:,} B; M={p["M"]}');ax[1,0].legend(fontsize=7,ncol=2,loc='lower left')
 for a,col in zip(ARMS[1:],COLORS[1:]):
  rows=[r for r in line['per_stream'] if r['policy']==a];med=np.array([r['mass_quantiles'] for r in rows]);ax[1,1].plot(range(5),np.median(med,axis=0),'-o',label=a,color=col)
 ax[1,1].set(xticks=range(5),xticklabels=['min','25%','50%','75%','max'],yscale='log',ylabel='Members per remote slot',title='Final source mass quantiles (median across12places)');ax[1,1].legend();fig.tight_layout();fig.savefig(figs/'costs.png');plt.close(fig)
 counter=d.read(d.OUT/'scoring/all_counterexamples.json');worst=next(r for r in counter if r['candidate']=='G' and r['baseline']=='I' and r['revisit']);pairs=d.read(d.OUT/'scoring/pairs.json');frames=d.read(d.OUT/'scoring/frames.json');rid=worst['id'];fig,ax=plt.subplots(1,2,figsize=(12.5,4.7))
 for a,col in zip(ARMS,COLORS):
  rr=[r for r in pairs if r['id']==rid and r['arm']==a];ff=[r for r in frames if r['id']==rid and r['arm']==a];ax[0].plot([r['t'] for r in rr],[r['relative_rotation_deg'] for r in rr],label=a,color=col);ax[1].plot([r['t'] for r in ff],[r['absolute_worldpoint_GT_error_raw_m'] for r in ff],label=a,color=col)
 ax[0].axvline(worst['t'],ls='--',color='.5');ax[0].set(xlabel='Target frame',ylabel='Far-pair rotation error (degrees)',title=f'Largest G/I revisit counterexample: {rid}, t={worst["t"]}');ax[1].set(xlabel='Current frame',ylabel='Absolute GT point error (raw m)',title='Own online geometry; no fitted scale');ax[0].legend();ax[1].legend();fig.tight_layout();fig.savefig(figs/'counterexample.png');plt.close(fig)
 print(dict(figures_complete=True),flush=True)

def report():
 p=d.plan();s=d.read(d.OUT/'scoring/summary.json');c=d.read(d.OUT/'costs.json');val=d.read(d.OUT/'validation.json');line=d.read(d.OUT/'lineage_validation.json');by=d.read(d.OUT/'scoring/by_place.json');comp=s['comparisons'];primary=s['primary'];counter=d.read(d.OUT/'scoring/all_counterexamples.json');sel=d.read(d.OUT/'scoring/selection.json');pairs=d.read(d.OUT/'scoring/pairs.json');lines=[]
 lines.append('# stream_spatial_merge1：有界有效K/V与三维支撑合并\n')
 lines.append(f"2026-11-15，round044完整实施[round043唯一合同]({d.SCIENCE})，SHA `{d.SCIENCE_SHA}`。**主 I_shape=E_I−E_G 重访旋转改善 {estimate(primary)} 度**。C/I/G各12×96完整原生预测，0训练／下载，全部used开发；W原12流预测／评分继承，固定首地点一次96帧完整成本重放。全结果交原Astra取舍，本报告不自选后继、不宣称fresh或总体目标完成。\n")
 lines.append(table(['预定比较（正值误差降低）','重访旋转° [95%配对区间]','全帧AbsRel改善 [区间]','旋转材料flag','明显AbsRel代价flag'],[[k,estimate(comp['revisit_coverage_ge5pct'][k]['relative_rotation_deg']),estimate(comp['all96frames'][k]['absrel']),s['predefined_flags'][k]['practical_rotation'],s['predefined_flags'][k]['obvious_AbsRel_cost']] for k in KEYS])+'\n')
 lines.append('主比较始终G/I；G/C、I/C辨别空间用途，C/I/G各对W分列原生净用途。≥1°且95%配对下界>0为预定开发材料信号；AbsRel增加≥.01且区间支持退步为明显代价。这不是普适门槛、等价／非劣性或自动后继决定。总体与近期目标保持[主控制canonical目标](../autonomous_research_loop/goal_revision_after_round042/revised_goal.md)；先保留主形状增量、普通／球形控制和W净用途的各自结论，再由Astra按合同作分支。\n')
 lines.append('合同事前取舍为：G/I主增量、G/C与G/W净用途均支持且无明显深度代价，才保留一个各向异性开发候选；G/I未立而I/C与I/W成立则只考虑简单球形；仅C/W成立则只保留普通聚合用途。没有成立路线则结束本固定内容候选集＋空间体积合并及自动变体，不扫描候选数、正则、近期配额、层、描述子、精度或数据。实际分支须由原Astra将完整结果与成本对应判断，本实现不自动晋升或派后继。\n')
 lines.append('## 固定假设、政策、信息与反馈\n')
 lines.append('检验预测三维支撑能否在内容相近的低Ward代价候选中区分沿表面和跨薄轴合并，从而保留原生重访几何。反事实为同内容、同距离、同迹但薄轴方向不同：球形分数相同，完整协方差更惩罚跨薄轴。竞争解释包括普通内容已足够、内容近不等于同表面、原生位置不服从局部表面、预测漂移／早期不可逆分配／累计质量使空间规则更坏。支撑不是定位置信度；机制与在线净用途分开。\n')
 lines.append(table(['组','超容量时从内容候选中选择','共同保存／读取'],[['C','最低全内容Ward（全局最小，候选必包含）','全24层有效K/V计数均值、n/m、μ和完整Σ'],['I','最低同迹球形体积增量g_I','仅分配评分球形化；仍形成／保存完整Σ'],['G','最低完整协方差体积增量g_G','Σ不进入PE或attention'],['W','旧首帧＋最近7完整原生帧','旧预测／分数继承，仅首地点一次成本重放']])+'\n')
 lines.append(f"首帧＋最近3完整patch为精确部分；特殊5token与全部4轮CameraHead保持首帧＋最近7。刚退出近期的patch才进入远端，每来源恰好一次，首帧不重复。槽容量M={p['M']}，各槽跨24层／16head／64维，n为整数成员数；有效key严格是原k_norm→自身source二维RoPE后的FP32向量，value原生BF16。远端K/V均值FP32保存，按n更新；读时按共同原生BF16读取，远端加log n，与精确质量1条目共用完整softmax和原WO。均值不再次归一／旋转，无新PE、query几何、训练、门控、质量衰减或连续高斯attention。\n")
 lines.append('各来源patch以自身当前depth和camera反投影14×14全部196像素，有限正深度点取均值及总体协方差；每有效patch计m=1，不把有效像素数当token质量。μ／Σ按m累计组内＋中心散布，FP64计算／保存。δ²固定首个几何有效来源帧：patch有效深度中位数平方乘(fx⁻²+fy⁻²)/2，再取有效patch尺度中位数；偶数中位取中间两值均值，未定义前共同走C。缺失几何保留，超容量先合并涉及m=0槽的最低内容对，之后几何只累计有效成员。无confidence筛选或正则倍率扫描。\n')
 lines.append('内容使用全49152维有效K/V，各层／head的K、V分别以首帧patch RMS固定缩放，同地点三组首源RMS/δ完全相同。Ward=n_i n_j||f_i−f_j||²/(n_i+n_j)，当前N有效几何槽的全部无序对中取最低N对，平局按最小来源ID对；每次合并后刷新。C直接取全局最低Ward是等价捷径。I用3[m_ij log(s_ij+δ²)−m_i log(s_i+δ²)−m_j log(s_j+δ²)]；G用m_ij logdet(Σ_ij+δ²I)减两原项，Σ_ij含中心散布。三组候选算法共同、分化后内容和候选各自形成，无共同教师。FP32全维dot禁TF32、FP64范数／Ward；每次直接从当前保存均值重算合并行，不用近邻、低秩或距离递推漂移。内容L2的负舍入值仅截0。\n')
 lines.append('各组自己的有界状态产生下一帧K/V和写入时固定的预测几何，当前帧预测完才成为未来来源。没有Full教师、W影子、其他组、淘汰raw或GT回流；预测／合并provenance仅离线输出。因分配改变未来内容与几何，这是政策总效应，不是固定内容纯形状因果或准确物理表面证明。\n')
 lines.append(f"官方提交 `{p['official_commit']}`；5,026,743,569-byte冻结eval权重SHA继承 `{p['model_sha256_inherited']}`，未重哈大权重。原12地点×96、完整518×392 RGB、预处理／相机／标签不变。聚合器原BF16 autocast；q_norm/sourceRoPE输出实际FP32 key，native SDPA实际共同读取BF16，camera/depth FP32。配置SHA `{d.sha(d.OUT/'configuration.json')}`；原配置SHA `{p['old_configuration_sha256']}`。GT仅离线；0新训练、下载、旧像素评分或新标签投影。\n")
 lines.append('## 接口验证、失败和精确恢复\n')
 lines.append('首次容量账误将有效source key当BF16：v1 M1153在第4帧后state_storage断言失败，前5帧aggregator/camera/depth均实际完成，17.0164386worker秒；0科学结果、无checkpoint。保留v1源码／配置／失败收据，5个丢弃预检帧在v2重做并完整计费。按实际FP32key/BF16value格式在首个合并和效果前改正合法M401，未按效果改容量、精度或科学规则。选择保留正确FP32缓存，没有为了提高M扫描raw缓存／重算格式。\n')
 lines.append('无合并D+质量1重排CPU FP64最大差7.77e−16；log n=log3与三份相同条目最大差3.33e−16，覆盖全部16head和6query。实际nativeBF16 SDPA对独立FP64的相对L2为.0016942，按浮点材料影响披露、不设新门禁。全49152维字面NumPy Ward对生产FP32 dot相对差1.70746e−8；共同候选／平局／m0优先的CPU例核对、GPU薄轴反事实均保留。首地点C/I/G在合并前0…4帧depth/pose对W最大差0；唯一W96对旧输出depth/pose最大差0。详见validation.json与toy_validation.json，不重新评分旧像素。\n')
 lines.append(f"独立CPU重放全部{line['merge_events_replayed']:,}次合并和{line['causal_completed_frames']}完整帧来源，核对ID、n/m链接、每来源仅入库一次和最终充分统计质量。每组每流最终远端95,312来源、{p['M']}槽，总几何有效质量独立等于来源帧有效patch计数之和；全部地点三组初始RMS与δ一致。ID/质量不是准确几何真值，m0单例首次状态0/1与后续链接、总量分开验证。\n")
 lines.append('首G在frame70后人为cooperative保存，next_frame71完整恢复；首I因GPU3外部PID到达在89后保存，从90恢复。正式同卡共驻升级前唯一manager129286/owner75772实际exit75，活动I02/G02/C03分别保存next83/55/77，全部旧worker实际wait后新唯一manager14557/owner85235续缺失帧；没有另起旁路、未重复有效帧。所有后续外部礼让与完整记录见costs checkpoint/restore表及workers。每次保存全状态、4相机缓存、RNG、输出；恢复核对shape/dtype/样本、全部remoteIDs/n/m，下一forward前删除消费的旧backing。突发无checkpoint错误不暗中回滚。\n')
 g=d.read(d.OUT/'geometry_reference.json');lines.append(f"固定首G流另以NumPy独立相机解码、全部{g['valid_backprojected_pixels']:,}有效反投影像素及原合并事件重建最终几何：μ相对L2 {g['final_mu_relative_L2']:.3g}、完整Σ相对L2 {g['final_covariance_relative_L2']:.3g}，δ²相对差 {g['delta_squared_relative_difference']:.3g}。生产相机先FP32解码、独立参考FP64，有限舍入差披露，不据此增加新门禁；这只验证固定成员的统计形成／合并，无GT准确性或新政策效果主张。见geometry_reference.json。\n")
 lines.append('CPU初版来源核对parser把未合并帧缺省events误作必有字段，owner49067/PID21919实际exit1；0GPU调用、0已完成核对流，改用空表后各流只核对一次并缓存。独立新指标按9＋2＋1地点增量缓存，最后统一全部12地点配对，旧W不重评分。缓存v2以显式编码保留非有限值；早期v1实际预测无失败后才复用，不删失败行。该短CPU parser异常前未记录body计时，不补造；实际exit与所有可用CPU计时保留usage/engineering_revisions/owner_exit_receipts。\n')
 lines.append('## 主终点、全部地点、原NA和支持\n')
 rows=[]
 for i,m in enumerate(p['catalog']):
  z={a:next(r for r in by if r['id']==m['id'] and r['arm']==a and r['group']=='revisit_coverage_ge5pct') for a in ARMS};rows.append([m['place'],z['W']['records'],*[n(z[a]['relative_rotation_deg']) for a in ARMS],n(primary['per_place'][i])])
 lines.append(table(['地点','重访/63','W°','C°','I°','G°','I−G°'],rows)+'\n')
 cv=s['coverage'];loo=[x for x in primary['leave_one_place_out_means'] if x is not None]
 lines.append(f"原t33…95、s≥1、t−s≥32，原GT min-z共视和最佳source平局规则继承；≥5%重访及20%层未改。{cv['far_targets']}远端目标，{cv['revisit_targets']}重访来自{cv['places_with_revisits']}/12地点；三处无重访NA、{cv['zero_selected_covisibility']}零共视、{cv['zero_selected_exclusive']}选定独占零支持均保留。先地点内均值再有定义地点等权；5000次配对bootstrap重采样全部12地点ID，seed20261114，NA不填0，区间条件于有定义抽样分母。主逐一留出范围{n(min(loo))}…{n(max(loo))}°，正向{primary['improved_places']}/{primary['defined_places']}；实际远端时间{n(min(r['elapsed_seconds'] for r in sel),2)}…{n(max(r['elapsed_seconds'] for r in sel),2)}秒。GT近平移零只使方向辅助NA，目标数{sum(r['GT_translation_near_zero'] for r in pairs if r['arm']=='W')}/756，主旋转不删。\n")
 lines.append('![主比较、净用途与全部地点](figures/primary.png)\n')
 lines.append('## raw深度、偏差、平移、绝对几何及一致性\n')
 fm=['trajectory_rotation_anchor_deg','trajectory_center_error_raw_m','absolute_worldpoint_GT_error_raw_m','absrel','log_mse','centered_log_mse','bias_squared','RMSE_m','failure'];rows=[]
 for metric in fm:
  raw=[n(np.mean([r[metric] for r in by if r['arm']==a and r['group']=='all96frames' and r[metric] is not None])) for a in ARMS];rows.append([metric,*raw,*[estimate(comp['all96frames'][k][metric]) for k in KEYS]])
 lines.append(table(['全96帧地点等权','W','C','I','G',*KEYS],rows)+'\n')
 rows=[]
 for group in ['all63far_targets','revisit_coverage_ge5pct','nonrevisit_lt5pct']:
  for metric in ['relative_rotation_deg','relative_translation_direction_deg','relative_translation_vector_error_raw_m','cross_time_point_consistency_raw_m','paired_absolute_GT_geometry_error_raw_m','failure']:rows.append([group,metric,*[estimate(comp[group][k][metric]) for k in KEYS]])
 lines.append(table(['目标集合','指标',*KEYS],rows)+'\n')
 lines.append('raw log-MSE=去均值结构＋偏差平方；完整AbsRel／RMSE／失败并列。仅首帧刚体gauge对齐，不拟合GT尺度／仿射；原生深度和平移raw单位保持。跨时点一致性同时报告绝对GT点误差和GT投影采样差，不能以塌缩伪装几何成功。全部帧与全部756目标见评分JSON/CSV，故不只展示重访优势。\n![完整深度代价](figures/depth.png)\n')
 lines.append('## 原支持层、质量分配和全部反例\n');strata=d.read(d.OUT/'scoring/support_strata.json');paired=d.read(d.OUT/'scoring/support_strata_paired.json');rows=[]
 for scope in ['selected_pair_exclusive','all_far_union_exclusive']:
  for st in ['0','(0,5%]','(5,20%]','>20%','undefined']:
   z=next(r for r in strata if r['arm']=='W' and r['scope']==scope and r['stratum']==st and r['revisit_only']);effects=[next(r for r in paired if r['candidate']==a and r['baseline']==b and r['scope']==scope and r['stratum']==st and r['revisit_only'])['effects']['relative_rotation_deg'] for a,b in [('G','I'),('G','C'),('I','C'),('C','W'),('I','W'),('G','W')]];rows.append([scope,st,z['records'],z['places'],*[estimate(e) for e in effects]])
 lines.append(table(['重访支持定义','层','记录数','地点数',*KEYS],rows)+'\n')
 lines.append('selected最佳source独占与all-far-union独占均仍扣除原W首帧＋最近7参考mask，不改成有利于本轮的覆盖。所有756目标／重访子集、两定义、零／undefined层、全部指标和地点配对区间见support_strata_paired.json；层内相关不当因果单调性。\n')
 rows=[]
 for a in ['C','I','G']:
  rr=[r for r in line['per_stream'] if r['policy']==a];rows.append([a,sum(r['merges'] for r in rr),sum(r['invalid_remote_source_patches'] for r in rr),*[n(np.median([r['mass_quantiles'][i] for r in rr])) for i in range(5)],n(np.median([r['selected_Ward_quantiles'][2] for r in rr])),n(min(r['final_covariance_eigenvalue_min'] for r in rr),8)])
 lines.append(table(['组','合并总数','缺失来源patch','最终n min','n q25','n median','n q75','n max','所选Ward中位的地点中位','最终Σ最小特征值'],rows)+'\n')
 lines.append('所有地点逐帧m0／有效质量、选择原因、内容与几何分数分位、n/m分布、δ²和协方差迹／特征值在lineage_validation.json；完整逐合并事件和所选两分数保存在各预测归档。C的几何评分0是未使用字段，不冒充球形／完整体积的可比数值。分数和质量只是解释，没有同表面GT；各组未来内容与几何不同。\n')
 for a,b in [('G','I'),('G','C'),('I','C'),('C','W'),('I','W'),('G','W')]:
  rr=[r for r in counter if r['candidate']==a and r['baseline']==b and r['revisit']][:4];lines.append(table([f'{a}/{b}最差重访（含负例）','t/s','覆盖','独占','秒','旋转改善°'],[[r['id'],f"{r['t']}/{r['s']}",n(r['coverage'],3),n(r['exclusive_fraction'],3),n(r['elapsed_seconds'],2),n(r['rotation_improvement_deg'])] for r in rr])+'\n')
 lines.append('全部756×6=4536比较行、全部正负／非重访／零覆盖见all_counterexamples.csv，不只挑选最佳个例。\n![主比较最大重访反例](figures/counterexample.png)\n')
 lines.append('## 实际B、容量、生命周期和成本\n')
 lines.append(f"效果前固定总历史B=**{d.B:,}bytes**，共同M=**{p['M']}**。单份状态上界S={p['state_copy_bound_bytes']:,}，workspace={p['history_work_bound_bytes']:,}，全部12可读RGB={p['RGB_readable_batch_backing_bytes']:,}；3S＋workspace＋RGB=**{p['total_history_bound_bytes']:,}≤B**，余量{d.B-p['total_history_bound_bytes']:,}，M+1超限。精确4×1036patch＋8×5特殊每token有效K/V147456B（FP32 K＋BF16 V）＋ID8＋FP64μΣ96＋valid1；remote每槽FP32均值196608＋n/m/ID24＋几何96。相机2MiB、RMS3072、δ8，metadata/Python/RNG保守未分配128KiB。不是填充最优前沿或长期千帧证明。\n")
 lines.append('合并真实保有old remote means、M+1036 expanded means及incoming转换，另有全维缩放f和FP64Ward矩阵；workspace显式按最大M+1036计算。f/Ward在compaction前释放；旧kv在聚合器返回后释放，raw/current/入库临时在exit保存前删除。正常运行不留周期CPU副本或旧diskcheckpoint；save才GPU当前＋CPU copy＋一个新atomic disk状态，restore在下一forward前删除消费backing。history_lifecycle_bound逐段核算最大可达副本，不仅以三份固定S掩盖expanded峰。native模型／当前activation／allocator另报，历史专属临时工作区与全部可读RGB已计B；输出/GT/provenance不供forward读取，但物理DATA全部另计。\n')
 rows=[]
 for a in ARMS:
  rr=[r for r in c['per_stream'] if r['policy']==a]
  for key in ['RGB_read_transfer_seconds','native_aggregator_seconds','native_camera_depth_seconds','source_geometry_seconds','merge_seconds','full_update_seconds','wall_seconds']:
   total=sum(r['components_seconds'][key] for r in rr);rows.append([a,key,n(total,5),n(total/(96*len(rr))*1000,4)])
 lines.append(table(['实际组','环节（部分嵌套）','总秒','ms/完整帧'],rows)+'\n')
 lines.append(table(['含加载／失败／恢复／永久IO的实际组','完整流数','全部worker秒','摊销ms/已完成帧'],[[a,len([r for r in c['per_stream'] if r['policy']==a]),n(sum(r['actual_worker_seconds'] for r in c['per_stream'] if r['policy']==a),5),n(1000*sum(r['actual_worker_seconds'] for r in c['per_stream'] if r['policy']==a)/sum(r['frames'] for r in c['per_stream'] if r['policy']==a),4)] for a in ARMS])+'\n')
 lines.append('形成包括当前完整encoder／24层有效K/V及自身camera/depth/全部像素几何；读取在真实native aggregator中，与原sourceRoPE、完整分母、WO和当前帧一起发生。full_update包含source_geometry与merge，merge又含全内容矩阵／逐合并候选重算／CPU同步和compaction，不能把嵌套各项相加。完整query是RGB到原生camera4轮与depth、真实merge/update和输出；frame wall不含最后永久保存，worker wall包含加载、所有I/O／恢复／失败。没有缓存小reader冒充完整query或免费几何／传输。\n')
 a=d.read(d.OUT/'profiling_admission.json');lines.append(f"科学效果评分前{a['registered_utc']}固定首流成本已准入：C {a['first_fixed_stream_seconds']['C']:.6f}s（含失败17.016s），I {a['first_fixed_stream_seconds']['I']:.6f}s，G {a['first_fixed_stream_seconds']['G']:.6f}s，W {a['first_fixed_stream_seconds']['W']:.6f}s；预检共{c['preflight_cost_seconds']:.6f}s={c['preflight_cost_seconds']/3600:.6f}h≤1，保守全轮投影{a['projected_total_worker_seconds']:.6f}s≤28800。首CIG结果复用正式。本点不构成加速结果：固定首流C/I/G全链worker代价均大于W，真实几何／合并成本必须与指标收益一起权衡。正式曾尝试两流共驻以填补小kernel／CPU同步间隙，但现有正式日志每帧耗时近翻倍，未见明确材料吞吐收益，因此resource_settings动态降回1流／卡，已有流自然排空、不重跑；全部尝试收费。首流独占与共享调度wall分开，不把混合计时当C/I/G独占速度因果排名。全部同卡重叠由PID/GPU/start-end派生保留costs.cohost_overlap，无另跑benchmark。\n")
 lines.append(f"全部GPU进程Popen→actualwait合计 **{c['all_GPU_worker_hours']:.6f}worker小时≤8**，{c['actual_worker_processes']}worker全actualwait；model-load {c['model_load_body_seconds']:.6f}s，checkpoint IO {c['checkpoint_IO_seconds']:.6f}s，restore {c['restore_seconds']:.6f}s，失败开销未扣除；初次失败进程缺少异常前详细RSS/I-O终结记录，已保存17.016秒wall、全部模块调用和NVML采样，不补造该细项。NVML采样单worker峰{c['peak_NVML_worker_MiB']/1024:.6f}GiB，allocated {c['peak_GPU_allocated_bytes']/2**30:.6f}、reserved {c['peak_GPU_reserved_bytes']/2**30:.6f}GiB≤40；单worker RSS {c['peak_RSS_worker_bytes']/2**30:.6f}GiB。DATA采样峰{c['DATA_peak_sampled_bytes']/2**30:.6f}GiB，全部永久输出＋保守8并发完整B（含GPU/CPU/disk历史副本、workspace／RGB）及输出原子暂存，并与16并发disk-only exit界取max的峰界{c['DATA_conservative_bound_bytes']/2**30:.6f}GiB≤32，实际final {c['DATA_final_bytes']/2**30:.6f}GiB。project和所有CPU阶段终态见terminal_record／usage。\n")
 lines.append('按现场CUDA/NVML、外部PID、CPU/RAM及两盘准入所有安全空卡，受限exec不可见GPU不误诊驱动。共驻每份12GiB保守准入、第二份需free>14GiB，最多2流／自有卡；foreign到达只礼让自有进程，冷却稳定后再续，忙卡余显存不当空算力。不改GLIBC、不删旧结果；同卡两份每流B和DATA恢复峰均独立收费。传输逻辑bytes、全部进程I/O与保存文件bytes见costs，不声称PCIe bandwidth profiling。\n![实际完整成本、字节与成员分配](figures/costs.png)\n')
 calls=d.read(d.OUT/'call_accounting.json');lines.append(table(['实际调用／保存／恢复','数量'],[[k,v] for k,v in calls.items() if isinstance(v,(int,float))])+'\n')
 lines.append('## 独立评分、O/C与证据边界\n');z=s['independent']
 lines.append(f"新C/I/G全部{z['unique_depth_records']}份depth的独立NumPy/Torch指标最大差{z['NumPy_Torch_depth_metric_max_difference']:.3g}，双路径FP64 resize差{z['NumPy_Torch_FP64_resize_max_difference']:.3g}，独立相机解码对官方差{z['NumPy_official_Torch_pose_decode_max_difference']:.3g}，NumPy/SciPy旋转差{z['NumPy_SciPy_relative_rotation_max_difference_degrees']:.3g}°。W1152帧＋756对原行直接继承，0旧像素重评分／新标签投影／评分模型调用。所有原分母／失败／零支持／NA、配对区间／LOO未按结果改变。\n")
 lines.append('O2公共空间统计、O3真实有界递推、O4原生精度—成本为核心；不是完整3DGS，O5渲染／O6 PE无新结论。C1自身因果形成、C2有效均值、C3固定内容候选＋三种分配、C4原source位置、C5旧稀疏流、C6冻结均是临时选择。与round041响应矩／round042关闭XYZ斜率不同，本轮只改成员分配；与U/C未融合保存不同，计数均值保留全部入库来源质量；coarse_memory静态raw池化的历史正负不重命名。旧T/D整帧、所有PE与XYZ斜率路线关闭保留。当前固定点阴性不否定所有三维记忆，阳性也只是开发线索，真正未用确认需另经历史身份审查。\n')
 lines.append('完整证据：[评分](scoring/summary.json)、[每帧](scoring/frames.csv)、[每对](scoring/pairs.csv)、[地点](scoring/by_place.csv)、[支持区间](scoring/support_strata_paired.json)、[全部反例](scoring/all_counterexamples.csv)、[来源质量](lineage_validation.json)、[接口](validation.json)、[调用](call_accounting.json)、[完整成本](costs.json)、[真实字节](history_bytes.json)、[图实看](figure_review.json)、[执行身份](execution_identity.json)、[全部终态](terminal_record.json)、[manifest](manifest.json)、[唯一短handoff](implementation_handoff.json)。36新预测＋首W成本预测的永久SHA在completed索引，完整来源/配置/代码版本与失败历史保留。最终科学处置归主→原Astra，本实现actualFINAL释放写租约。\n')
 (d.OUT/'report.md').write_text(('\n'.join(lines)+'\n').replace('(figures/',f'({d.OUT}/figures/'))
 print(dict(report=str(d.OUT/'report.md')),flush=True)

def main():
 start=time.monotonic();torch.set_num_threads(2);print(dict(CPU_report_PID=os.getpid(),CUDA_initialized=torch.cuda.is_initialized()),flush=True);figures();report();d.write(d.OUT/'CPU_report_runtime.json',dict(complete=True,PID=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized(),new_model_calls=0))
if __name__=='__main__':main()
