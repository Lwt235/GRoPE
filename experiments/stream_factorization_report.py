"""Complete fixed-contract report, without choosing a scientific successor."""
import os,time,resource
import numpy as np
import torch
from experiments import stream_factorization_data as d
from experiments.stream_history_report import n,estimate,table
def main():
 start=time.monotonic();torch.set_num_threads(2);p=d.plan();s=d.read(d.OUT/'scoring/summary.json');c=d.read(d.OUT/'costs.json');b=d.read(d.OUT/'history_bytes.json');calls=d.read(d.OUT/'call_accounting.json');by=d.read(d.OUT/'scoring/by_place.json');sel=d.read(d.OUT/'scoring/selection.json');counter=d.read(d.OUT/'scoring/all_counterexamples.json');inter=d.read(d.OUT/'scoring/interaction_rows.json');pre=d.read(d.OUT/'preflight.json');comp=s['comparisons'];primary=s['primary'];cov=s['coverage'];keys=['WF_vs_WW','FW_vs_WW','FF_vs_WW','A_vs_FF'];names=['I_read（主）=E_WW−E_WF','I_write=E_WW−E_FW','joint=E_WW−E_FF','direct A/FF=E_FF−E_A'];lines=[]
 lines.append('# stream_history_factorization1：历史形成 × 当前读取\n')
 lines.append(f"2026-11-15，round036完整实施[round035唯一合同]({d.SCIENCE})，SHA `{d.SCIENCE_SHA}`。**主I_read重访旋转改善 {estimate(primary)} 度**，正值表示W形成的旧条目在固定实际W相机前缀后可降低当前读取误差。原12地点×96完整RGB、固定189重访／756远端目标；四格输出全部完成，0训练／下载。全部数据used development；不声明独立确认、同byte部署优胜、高斯用途或总体成功，后续科学决定归原Astra。\n")
 rows=[]
 for label,k in zip(names,keys):rows.append([label,estimate(comp['revisit_coverage_ge5pct'][k]['relative_rotation_deg']),estimate(comp['all96frames'][k]['absrel']),str(s['predefined_flags'][k]['practical_rotation']),str(s['predefined_flags'][k]['obvious_AbsRel_cost'])])
 rows.append(['interaction=(E_FW−E_FF)−(E_WW−E_WF)',estimate(s['interaction']['revisit_coverage_ge5pct']['relative_rotation_deg']),estimate(s['interaction']['all96frames']['absrel']),'预定辅助','直接配对，不由显著性差推断'])
 lines.append(table(['固定效应','旋转度 [95%配对区间]','AbsRel误差对比 [区间]','旋转材料flag','AbsRel材料代价flag'],rows)+'\n')
 lines.append('误差差的正值均按表中明确公式解释。interaction正值表示F形成下的直接长读取改善比W形成下更大；它不是独立新候选。direct A/FF正值表示旧A误差比新FF低，才直接支持固定W相机隐藏值后旧A收益下降的解释；不能仅凭FF未显著就归因相机。旋转≥1°且下界>0、AbsRel增加≥.01且区间支持退步是本轮开发投入标记，不是通用阈值、等价证明或自动后继。辅助正向、负向、宽区间、地点差异和原零支持均不删。\n')
 lines.append('当前总体／近期目标以[主控制的权威目标](../autonomous_research_loop/goal_revision_20261115/revised_goal.md)为准：总体保持revision31，先完整交付本轮再据结果决定至多一个后继；不重跑已完P0/P1、不自动P2。本报告不改主goal/state/events或目标状态。\n')
 lines.append('## 唯一干预与因果范围\n')
 lines.append('假设是W形成的旧K/V在固定接收端后仍有直接旋转用途；竞争解释为完整历史主要改变随后写入、读取／形成必须耦合，或旧收益大部分经相机隐藏值传播。相对round032只匹配相机保留规则、round034同时改变未来写入的固定配额保存，本次实质变化是交叉来源／读取并固定实际相机tensor。I_read、I_write、joint／interaction和直接A/FF各自回答不同分支；完整结果交原Astra按唯一合同决定，不在这里自动采用／扩数据或重开旧T/D／PE家族。\n')
 lines.append('两条参考聚合器流独立按原生规则写入：F依赖自己的完整历史；W仅依赖自己的首帧＋最近7个完成帧。在W淘汰前仅把本帧24层raw K/V记录到完整W影子库；影子库只供WF诊断读取，绝不回流W参考。当前RGB在WW、FF、WF、FW各自全24层重新计算，保持原位置、5个特殊token、当前自身K/V、原SDPA和精度；WW聚合器同时是W参考当前形成，FF聚合器同时是F参考当前形成，符合合同的等价复用。\n')
 lines.append(table(['四格','形成来源／过去读取','未来提交'],[['WW','W／首帧＋最近7','独立W聚合器状态＋唯一共同W相机状态'],['WF','W影子／全部到达帧','本次交叉输出全部丢弃'],['FW','F／首帧＋最近7','本次交叉输出全部丢弃'],['FF','F／全部到达帧','F参考仅提交自己的聚合器写入；其相机输出丢弃']])+'\n')
 lines.append('同一帧所有camera worklist均由同一个尚未提交的WW前缀list浅复制，元素直接引用逐tensor相同的不可变原生K/V；每个分支完成原4轮，native cat形成新工作状态而不改旧tensor。前缀shape/dtype/固定样本在每分支前后核对；只有WW返回状态在整帧所有分支结束后提交，再统一淘汰至首帧＋最近7。不是仅规则相同或只固定相机窗口长度。\n')
 lines.append('**这是逐帧条件反事实集合，不是四条独立在线系统轨迹。** F/W参考只接受自己的聚合器输出，WF/FW/FF相机或交叉聚合器输出不进入未来。跨时旋转／平移／点几何由这类条件输出构成，应据此解读，可能存在状态分布失配。FF使用实际W相机隐藏值，所以不是旧A或旧F；旧A/FF仅作预定接收端辅助。即使形成收益存在，也不能把完整F教师生产的紧凑读取当有界在线实现。\n')
 lines.append('前0…8帧读取前缀尚完全相同：每帧只算一次WW，F参考写入同一数学raw K/V，四格camera/depth也共享同一次。第9帧起固定四次原生当前图像计算。该预定工程等价只减少重复调用，没有按结果筛选；原完整96身份和每个逻辑输出均保留。\n')
 lines.append('## 身份、实现与恢复验证\n')
 lines.append(f"同一冻结官方提交 `{p['official_commit']}`、5,026,743,569-byte权重，继承SHA `{p['model_sha256_inherited']}`，未重新哈希大权重。BF16原生聚合器、FP32 camera/depth、518×392完整RGB、4次camera精化，eval且所有权重requires_grad=False。输入、原pose、顺序及标签沿用旧配置SHA `{p['old_configuration_sha256']}`；新配置SHA `{d.sha(d.OUT/'configuration.json')}`。GT只离线评分，不进入保留／前向，0下载／训练／新标签投影／旧像素评分。\n")
 lines.append(f"首地点96帧作为固定成本预检且复用正式输出；WW与旧W全部depth/pose最大绝对差分别为{pre['WW_depth_max_abs']:.8g}、{pre['WW_pose_max_abs']:.8g}；新FF depth与旧F depth最大差{pre['FF_depth_max_abs_vs_old_F']:.8g}，该head不依赖camera。这是原生staging／参考形成身份验证，不要求新FF pose等于旧A/F，不把数值身份当效果证据。\n")
 lines.append('固定首流在完整frame55主动安全保存，原worker actualwait75；后继从56恢复，无已完成帧重复。完整F/W源库是同一组CPU共享映射文件，fsync后保持原字节，不重新构造教师；独立W bounded K/V、共同camera4轮缓存、CPU/CUDA RNG、ID、已完成输出和next_frame保存／恢复。恢复核对全部bounded/camera shape/dtype/固定样本与配置身份并删除消费checkpoint；F/W库完整原精度映射继续，只在当前帧所在列写入。突发未保存错误将停止，不悄悄回滚重跑。一次工程validation001仅补全最终全96回放检查，模型数值路径未变；实际外部礼让若有均见workers／恢复记录。\n')
 lines.append('## 端点、全地点与反例\n')
 lines.append('固定s≥1、t−s≥32，t=33…95；原210×280深度前向min-z共视容差max(.05m,.05目标深度)，每目标最大覆盖者／平局最早s、覆盖≥5%重访。标签不重算，不改变20%分层，不按预测筛支持。先每地点重访均值再已定义地点等权；5000次配对bootstrap重采样全部12地点ID，seed20261114，NA不填0，区间条件于有定义重采样分母。每个对比及交互都直接用配对每地点差构造，不相减两个区间。\n')
 rows=[]
 for i,m in enumerate(p['catalog']):
  r={a:next(v for v in by if v['id']==m['id'] and v['arm']==a and v['group']=='revisit_coverage_ge5pct') for a in ('WW','WF','FW','FF','A')};rows.append([m['place'],r['WW']['records'],*[n(r[a]['relative_rotation_deg']) for a in ('WW','WF','FW','FF','A')],n(primary['per_place'][i]),n(s['interaction']['revisit_coverage_ge5pct']['relative_rotation_deg']['per_place'][i])])
 lines.append(table(['地点','重访/63','WW','WF','FW','FF','旧A','I_read','interaction'],rows)+'\n')
 loo=[v for v in primary['leave_one_place_out_means'] if v is not None];lines.append(f"原{cov['far_targets']}远端目标、{cov['revisit_targets']}重访来自{cov['places_with_revisits']}/12地点；三处NA不删。共视零覆盖{cov['zero_selected_covisibility']}、选定独占零支持{cov['zero_selected_exclusive']}完整保留。I_read正向地点{primary['improved_places']}/{primary['defined_places']}，逐一留出均值范围{n(min(loo))}…{n(max(loo))}°；真实远端时距{n(min(r['elapsed_seconds'] for r in sel),2)}…{n(max(r['elapsed_seconds'] for r in sel),2)}s。\n")
 lines.append('![全部固定效应与地点](figures/primary.png)\n')
 lines.append('## Raw深度、偏差、平移与条件几何\n')
 metrics=['trajectory_rotation_anchor_deg','trajectory_center_error_raw_m','absolute_worldpoint_GT_error_raw_m','absrel','log_mse','centered_log_mse','bias_squared','RMSE_m','failure'];rows=[]
 for metric in metrics:
  vals=[n(np.mean([r[metric] for r in by if r['arm']==a and r['group']=='all96frames' and r[metric] is not None])) for a in ('WW','WF','FW','FF')];rows.append([metric,*vals,*[estimate(comp['all96frames'][k][metric]) for k in keys[:3]]])
 lines.append(table(['全96帧地点等权','WW','WF','FW','FF','I_read','I_write','joint'],rows)+'\n')
 lines.append('raw log-MSE=去均值结构项＋bias平方逐项保留，完整AbsRel、RMSE和无效像素／相机失败原数不删。全帧与跨时几何只做首帧刚体gauge，不拟合GT scale或仿射；不能把尺度塌缩的一致性当绝对几何准确。所有交叉输出的轨迹统计是条件诊断，不能当独立在线闭环。\n')
 rows=[]
 for group in ('all63far_targets','revisit_coverage_ge5pct','nonrevisit_lt5pct'):
  for metric in ('relative_rotation_deg','relative_translation_direction_deg','relative_translation_vector_error_raw_m','cross_time_point_consistency_raw_m','paired_absolute_GT_geometry_error_raw_m','failure'):
   rows.append([group,metric,*[estimate(comp[group][k][metric]) for k in keys],estimate(s['interaction'][group][metric])])
 lines.append(table(['固定集合','指标','I_read','I_write','joint','direct A/FF','interaction'],rows)+'\n')
 lines.append('所有目标的GT采样点差、平移近零NA及模型失败显式保留在逐对表；没有删除无支持目标来改善几何分数。![全部raw深度代价](figures/depth.png)\n')
 lines.append('## 原支持层与全部反例\n')
 strata=d.read(d.OUT/'scoring/support_strata.json');sp=d.read(d.OUT/'scoring/support_strata_paired.json');si=d.read(d.OUT/'scoring/support_strata_interaction.json');rows=[]
 for scope in ('selected_pair_exclusive','all_far_union_exclusive'):
  for st in ('0','(0,5%]','(5,20%]','>20%','undefined'):
   r=next(x for x in strata if x['arm']=='WW' and x['scope']==scope and x['stratum']==st and x['revisit_only']);z=[next(x for x in sp if x['scope']==scope and x['stratum']==st and x['revisit_only'] and x['candidate']==a and x['baseline']==base)['effects']['relative_rotation_deg'] for a,base in [('WF','WW'),('FW','WW'),('FF','WW')]];ii=next(x for x in si if x['scope']==scope and x['stratum']==st and x['revisit_only'])['effects']['relative_rotation_deg'];rows.append([scope,st,r['records'],r['places'],*[estimate(x) for x in z],estimate(ii)])
 lines.append(table(['原独占定义／重访子集','原支持层','记录数','地点数','I_read','I_write','joint','interaction'],rows)+'\n')
 lines.append('两种独占定义仍扣除原首帧＋最近7参考mask：选定best-pair与all-far-union分列。层内结果是关联，不当因果、单调趋势或调阈值依据；原全部756目标和重访子集的每层所有指标、地点分母与配对区间在support_strata_paired／support_strata_interaction，零支持与undefined全保留。\n')
 for label,k in zip(names,keys):
  a,base=k.split('_vs_');worst=[r for r in counter if r['candidate']==a and r['baseline']==base and r['revisit']][:5];lines.append(table([label+'最大重访反例','t/s','覆盖','独占','秒数','旋转改善°'],[[r['id'],f"{r['t']}/{r['s']}",n(r['coverage'],3),n(r['exclusive_fraction'],3),n(r['elapsed_seconds'],2),n(r['rotation_improvement_deg'])] for r in worst])+'\n')
 lines.append('全部比较756×5反例和756项直接交互原行均导出，不只保存最好／最坏子集。![最大主比较反例及绝对几何](figures/counterexample.png)\n')
 lines.append('## 真实历史库、教师与转移字节\n')
 lines.append(f"每来源96帧bank容量{b['bank_capacity_each_source_bytes']:,}bytes，两份F/W合计{b['both_source_bank_file_bytes']:,}bytes={b['both_source_bank_file_bytes']/2**30:.6f}GiB，从分配时即按完整逻辑文件容量计账，包括尚未到达的未写列；未写列从不读入forward。F当前rawKV只按列附加到自身full库，W当前rawKV只写影子库；W参考另有最多{b['common_W_bounded_CPU_max_bytes']:,}bytes的独立有界CPU状态，唯一共同W相机GPU {b['common_W_camera_GPU_max_bytes']:,}bytes。当前native窗口追加至9后再淘汰，瞬态和恢复另计。\n")
 lines.append('CPU staging不是免费：两组来源共96个rawKV文件以shared mmap在CPU寻址，CPU驻留页和磁盘backing均披露，是同一映射逻辑storage的两种硬件占用，不另伪造免费独立历史库。RSS包含页驻留，磁盘按完整18.3GiB文件而非du稀疏已分配块收费。各bank仅供本流进行期间使用，永久四格输出与完成SHA原子提交后删除本轮自有bank；未来重读新query必须重新形成并付费。共享映射写入经过OS页缓存，未声称每个正常结束流都物理落盘全库；唯一次安全恢复实际fsync计时与kernelIO记录明确分列。00–06每个global attention prehook把本层所选raw past复制到GPU；07–11以同一rawKV的18.30GiB GPU常驻镜像避免重复大H2D，CPU／disk镜像仍保留并分别收费，GPU镜像在段完成后、checkpoint／最终输出I-O前释放。WW始终用独立有界参考缓存，不回读W影子。原native qkv/cat、qk_norm／原RoPE／SDPA及当前自身K/V照常执行；只改变缓存硬件位置，GPU镜像模式每帧释放allocator未用保留区。引用F的FW/FF同时依赖完整F形成；WF依赖W形成和完整影子库；所有分支还依赖WW实际相机形成。共用数学工作只在联合实际账计一次，各单格依赖仍列出，不能把“共享”说成独立可部署的便宜方法。\n')
 lines.append(table(['保守硬件历史账（非DATA文件cap）','GiB'],[[k,n(v/2**30,6)] for k,v in b['conservative_history_hardware_ledger'].items() if isinstance(v,int)])+'\n')
 lines.append(table(['单格依赖','真实形成／保存范围'],[[k,v] for k,v in b['formation_dependencies'].items()])+'\n')
 lines.append(f"原完整12流可读RGB backing {b['whole_selected_readable_RGB_backing_bytes']:,}bytes保守列为每格共同依赖；实际forward只单调读取当前RGB，没有旧图重编码暗库。必要teacher／影子／CPU-GPU-DATA传输、原生位置与索引、恢复副本、原子写均收费。旧GT与预测归档不回流forward；物理新增输出仍计DATA。本轮不是旧8帧B、不是等byte候选，不给有界部署优越性结论。\n")
 lines.append('## 成本、精确调用与实际终态\n')
 lines.append(f"固定首流成本准入在{c['admission']['registered_utc']}登记、科学评分前；实测GPU worker {c['admission']['first96_actual_worker_seconds']:.4f}s，含两次加载和完整恢复／原子输出，另11流同案例×1.25余量投影总{c['admission']['total_projected_seconds']:.4f}s≤14400。该首流预测复用，没有额外benchmark或为比速度重跑；冷载指新进程加载，不保证物理文件uncached。不能从分段预检开销宣称单格独立部署加速。\n")
 tier=d.read(d.OUT/'tier_validation_admission.json');lines.append(f"尚未开始的07–11启用GPU镜像，普通工程006验证复用了完整07：WW depth/pose与旧W最大差{tier['WW_depth_max_abs']:.8g}/{tier['WW_pose_max_abs']:.8g}，FF depth与旧F差{tier['FF_depth_max_abs_vs_old_F']:.8g}，frame9两交叉对亦为0。07全进程{tier['full_formal_case_worker_seconds']:.6f}s另计入预检／成本，连原profile合计{tier['total_preflight_cost_worker_seconds']:.6f}s≤1800，所有输出计正式，0额外query。CPU模式00的分段成本准入与GPU镜像07的资源校验均在科学评分前登记；没有中断或重做已完成流。\n")
 lines.append(table(['实际新调用','数量'],[[k,v] for k,v in calls.items() if isinstance(v,(int,float))])+'\n')
 lines.append('每流实际9＋87×4=357次完整聚合器／camera／depth，12流4284；每次camera4轮共17136。逻辑四格输出4608，不把复用视为漏帧；W参考1152次，F参考1044次＋108次共享前缀，WF/FW共2088次。每次完整当前query包括RGB encoder、全部24层、原生camera4轮／depth及真实选择／转移，缓存读取延迟未假装完整query。单格表是模块计时，history索引选择及Python bookkeeping残差保留在四格完整envelope内，不冒称单格独立完整query。\n')
 rows=[]
 for cell in ('WW','WF','FW','FF'):
  rr=[r['cells'][cell] for r in c['per_stream']];cnt=sum(v['actual_aggregator_calls'] for v in rr);rows.append([cell,cnt,n(sum(v['aggregator_seconds'] for v in rr),4),n(sum(v['camera_depth_seconds'] for v in rr),4),sum(v['shared_first9_current_queries'] for v in rr)])
 lines.append(table(['条件实际独立调用','次数','aggregator含transfer秒','camera4轮＋depth秒','共享WW前缀逻辑帧'],rows)+'\n')
 transfer={k:sum(r['transfers'].get(k,0) for r in c['per_stream']) for k in sorted({k for r in c['per_stream'] for k in r['transfers']})};lines.append(table(['所有流实测CPU暂存环节','实际合计'],[[k,n(v/2**30,5)+' GiB' if k.endswith('bytes') else n(v,5)+' s'] for k,v in transfer.items()])+'\n')
 lines.append(f"联合两参考形成的aggregator共{c['joint_source_formation_aggregator_seconds']:.4f}s，同时产出WW/FF当前结果，不重复向联合账收费；WF/FW额外交叉当前读取共{c['joint_crossed_current_read_aggregator_seconds']:.4f}s。四格完整帧envelope摊销{c['complete_four_cell_query_amortized_seconds_per_arrival']:.6f}s/到达帧；全部进程含冷载／I-O／恢复摊销{c['all_processes_amortized_seconds_per_arrival']:.6f}s/到达帧。两者都含实际转移；前者不含帧间进程冷载，后者含。本轮无独立单格冷载测速或纯attention微基准。\n")
 lines.append('H2D hook计时含同步、真实CPU→CUDA复制和有关调度，不冒称纯PCIe带宽；native norm／RoPE／attention在aggregator剩余时间内，native raw复制瞬态／allocator保留也未隐藏。CPU bank write与D2H分列；联合完整query envelope及逐帧成本可重建。前七流较低瞬时GPU利用率主要来自逐层同步暂存；后五流采用同raw字段GPU镜像，全部容量／D2D写入亦计费，没有增加训练／扫描来填卡。不同固定地点与硬件模式的实际时耗分列，不当成单格方法速度的匹配试验。![完整成本及转移随历史增长](figures/costs.png)\n')
 lines.append(f"GPU所有Popen→actualwait合计 **{c['all_GPU_worker_hours']:.6f}worker小时≤4**；预检／成本{c['preflight_cost_seconds']/3600:.6f}小时≤.5，{c['actual_processes']}实际进程均已wait。模型加载body合计{c['process_model_load_body_seconds']:.4f}s，完整保存IO{c['checkpoint_IO_seconds']:.4f}s；失败／礼让／恢复从未扣掉。单worker NVML峰{c['peak_NVML_worker_MiB']/1024:.6f}GiB≤40，allocated{c['peak_GPU_allocated_bytes']/2**30:.6f}、reserved{c['peak_GPU_reserved_bytes']/2**30:.6f}GiB，RSS峰{c['peak_RSS_worker_bytes']/2**30:.6f}GiB。DATA采样峰{c['DATA_peak_sampled_bytes']/2**30:.6f}GiB、含全部永久输出＋两源库＋完整有界保存／原子暂存的保守峰{c['DATA_conservative_bound_bytes']/2**30:.6f}GiB≤32；最终{c['DATA_final_bytes']/2**30:.6f}GiB，project运行期采样峰与封装前实数在terminal_record，最终封装实数另见sealer回执。\n")
 lines.append('宿主CUDA可访问，restricted sandbox不可见设备不被误诊驱动坏。全8卡持续观察外部PID与稳定空闲，只使用当时安全资源；两个完整source bank在32GiB DATA内只容一条活跃／恢复流，是本轮可核算存储约束，不是默认两卡上限。外部任务到达只礼让自有worker，完整保存后missing-only续跑；纯CPU manager等待不计GPUworker，但自身wall及所有GPU加载／IO实耗均记录。不修改系统libc，不动其他用户文件／进程，不删旧结果。\n')
 lines.append('GPU没有意外失败；一次预定完整保存exit75已从56恢复并复用其输出。CPU版本回执helper曾因/tmp缺PYTHONPATH在任何模型／数据工作前退出，补路径后复用同一helper；figure007只修正实际看图发现的颜色图例，不改指标。常规修复、前后源版本与CPU阶段用时均保留。\n')
 lines.append('## 独立复算、边界与交付索引\n')
 q=s['independent'];lines.append(f"新WF/FW/FF全部{q['unique_depth_records']}份depth用NumPy FP64／Torch独立指标核对，最大差{q['NumPy_Torch_depth_metric_max_difference']:.3g}；FP64双线性resize最大差{q['NumPy_Torch_FP64_resize_max_difference']:.3g}；NumPy／官方Torch pose解码差{q['NumPy_official_Torch_pose_decode_max_difference']:.3g}；NumPy／SciPy相对SO3角差{q['NumPy_SciPy_relative_rotation_max_difference_degrees']:.3g}度。WW和旧A的2304逐帧／1512逐对指标直接继承，0旧像素评分／标签重投影／新模型调用。交互与直接A/FF从相同地点／相同对原行配对，线性恒等式和独立bootstrap核对见interface_validation。\n")
 lines.append('本次只辨别值得研究的是旧K/V直接读取、状态形成或它们耦合；实际固定W相机隐藏值的比较仍是条件干预，不唯一解释所有在线失败。完整F/W影子库和全部形成费用不是有界方法，旧T/D固定配额及全部PE家族关闭保持；旧round032旋转阳性／raw深度代价也不抹去。服务O3有界实现机制与O4原生用途；O2高斯结构不新增，O5渲染／O6 PE仍可选；C1两条自己写入、C2未压缩K/V、C3当前交叉＋同一相机、C4无新PE、C5旧12×96、C6冻结都是临时实验选择。没有fresh、总体完成、采用方法或自选后继。\n')
 lines.append('完整证据：[manifest](manifest.json)、[唯一实现交接](implementation_handoff.json)、[评分全数](scoring/summary.json)、[每帧](scoring/frames.csv)、[每对](scoring/pairs.csv)、[地点](scoring/by_place.csv)、[支持层](scoring/support_strata_paired.json)、[支持交互](scoring/support_strata_interaction.json)、[全部反例](scoring/all_counterexamples.csv)、[直接交互原行](scoring/interaction_rows.csv)、[真实字节](history_bytes.json)、[成本](costs.json)、[调用](call_accounting.json)、[实际看图](figure_review.json)、[终态](terminal_record.json)。所有新预测路径／SHA由completed索引，原配置／标签／权重保持旧位置。\n')
 (d.OUT/'report.md').write_text(('\n'.join(lines)+'\n').replace('(figures/',f'({d.OUT}/figures/'));d.write(d.OUT/'CPU_report_runtime.json',dict(complete=True,PID=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized(),new_model_calls=0));print(dict(report=str(d.OUT/'report.md')),flush=True)
if __name__=='__main__':main()
