"""Round034 full scientific evidence report, without selecting a successor."""
import os,time,resource
import numpy as np
import torch
from experiments import stream_retention_data as d
from experiments.stream_history_report import n,estimate,table

def main():
 start=time.monotonic();p=d.plan();s=d.read(d.OUT/'scoring/summary.json');c=d.read(d.OUT/'costs.json');b=d.read(d.OUT/'history_bytes.json');B=b['B'];calls=d.read(d.OUT/'call_accounting.json');by=d.read(d.OUT/'scoring/by_place.json');strata=d.read(d.OUT/'scoring/support_strata.json');counter=d.read(d.OUT/'scoring/all_counterexamples.json');sel=d.read(d.OUT/'scoring/selection.json');frames=d.read(d.OUT/'scoring/frames.json');pairs=d.read(d.OUT/'scoring/pairs.json');comp=s['comparisons'];primary=s['primary'];cov=s['coverage'];lines=[]
 lines.append('# stream_native_retention1：同总历史字节的原生保存政策\n')
 lines.append(f"2026-11-15，round034完整执行[round033唯一科学合同]({d.SCIENCE})，SHA `{d.SCIENCE_SHA}`。**主T/W重访旋转改善 {estimate(primary)} 度；正数有利T。** W既有12×96评分继承，T/D各12×96新预测全部完成，0训练／下载。D/T、D/W为预定辅助，Full/A仅继承不等预算背景。全部数据used development，科学取舍归原Astra，不自动P2/P3，不宣称fresh、高斯优越性或总体成功。\n")
 lines.append(table(['固定比较','重访旋转改善度 [95%配对区间]','全帧AbsRel改善 [95%区间]','≥1°且下界>0','AbsRel增加≥.01且有据'],[[k,estimate(comp['revisit_coverage_ge5pct'][k]['relative_rotation_deg']),estimate(comp['all96frames'][k]['absrel']),s['predefined_flags'][k]['practical_rotation'],s['predefined_flags'][k]['obvious_AbsRel_cost']] for k in ('T_vs_W','D_vs_T','D_vs_W')])+'\n')
 depthTW=comp['all96frames']['T_vs_W']['absrel'];depthDT=comp['all96frames']['D_vs_T']['absrel'];depthDW=comp['all96frames']['D_vs_W']['absrel']
 lines.append(f"T全帧AbsRel确有小幅退步：增加{-depthTW['mean']:.6f} [ {-depthTW['interval95'][1]:.6f}, {-depthTW['interval95'][0]:.6f} ]；未达到合同均值.01明显代价线不等于没有损伤或已证安全。D相对T的AbsRel改善{estimate(depthDT)}，但D/W为{estimate(depthDW)}且区间跨0；不把深度辅助改善追认为旋转主成功。旋转宽区间及地点差异均保留，不据均值负值证明所有少量历史保存无效。\n")
 lines.append('“旋转实用线索”与“明显深度代价”是合同中的开发投入排序量，不是通用标准、非劣性证明或自动方法采用。小幅不确定负值不自动否决，区间跨0也不称等价。主T/W保持主身份，不能事后用D的最好分数追认T成功；若只能保留方向用途，也不称全面净收益。\n')
 lines.append('## 固定问题、政策与匹配权限\n')
 lines.append('检验同样约8个完整原生帧状态的预算下，分散旧观测能否保留旋转价值且避免长缓存的深度代价。竞争解释：近期连续性更重要；收益需要大量重复历史；保存只稳定全局上下文而非专门重访检索。相对round032，实质改变是同总byte的在线淘汰，未增加高斯、合并、教师、PE、GT选帧或学习。旧长历史旋转阳性及raw深度代价保持，未通过尺度拟合消掉。\n')
 lines.append(table(['政策','原生聚合器保留','相机缓存'],[['W','首帧＋最近7个完成帧（旧预测/指标继承）','共同首帧＋最近7，4轮全部完成后淘汰'],['T','首帧＋最近3＋至多4远端；原近期帧退出时唯一入候选','同W'],['D','同T容量/入库/5选4候选；普通内容minimax余弦选择','同W']])+'\n')
 lines.append('T对现有远端4个＋当前退出近期者c，枚举5选4；在连续采样序号区间[0,c]，以首帧固定为覆盖点，最小化到最近保留时间点的最大距离。使用双倍radius=max(相邻保留序号gap,2×末端gap)精确整数比较；平局按保留ID集合字典序。没有使用未来时间点、真实秒数、raw frame ordinal或GT重新优化。D描述子为本次实际原生单图patch_embed输出x_norm_patchtokens的patch均值（FP32）再L2归一；仅hook已有编码，不额外encoder。D对同5候选，最小化候选到所选远端＋首帧＋最近3的最大余弦距离；CPU FP64归约来自FP32描述子，平局同样字典序。未满4远端时全部保留，只存当前保留帧描述子；pool未用行清零，无已淘汰描述子库。\n')
 lines.append('全部24层raw K/V及每帧5个特殊token随整帧保留；原patch位置和原global frame index保持，不融合、不加log-count、不重投影。各组从自己的因果隐藏状态继续写入；首帧和近期隐藏值可能间接携带更早信息。没有Full教师或他组cache输入。相机规则相同但隐藏值随自身轨迹产生，是独立在线政策总效应，非共同教师压缩机制隔离。\n')
 lines.append(f"固定同一官方权重/提交`{p['official_commit']}`，5,026,743,569-byte旧权重，继承SHA `{p['model_sha256_inherited']}`，不重哈希大权重。518×392完整RGB，BF16聚合器、FP32相机/depth、原SDPA和4轮相机精化；所有权重冻结eval，0训练。原12×96不同RGB-D身份、原full pose坐标、顺序、GT标签/189重访/支持层直接沿用[旧配置]({d.OLDOUT/'configuration.json'})，SHA `{p['old_configuration_sha256']}`。新配置SHA `{d.sha(d.OUT/'configuration.json')}`；不按效果或覆盖选数据。GT仅离线评分，未进入预测/状态/保留选择。\n")
 lines.append('## 必要接口、预检与精确恢复\n')
 pre=d.read(d.OUT/'preflight.json')
 lines.append('固定首地点W独立原生完整96帧重放，只作本轮同机成本与身份回放；与旧W全部depth/pose最大绝对差均0，没有重算旧像素指标。T00/D00前8帧容量尚覆盖全部历史，native depth/pose也与W各差0；这16帧计入各自正式96，并封完整状态，后续从8接续，不重复调用。T连续区间选择有88例独立中点覆盖核对，D有一例独立Torch FP64余弦核对，96帧保留容量检查通过；[接口证据](policy_interface_validation.json)、[原生预检](preflight.json)。这些是实现证据，不是方法收益。\n')
 lines.append('原configuration中的preflight_fixed_frames=3是未被run接口读取的旧草稿计数字段；实际首次执行前的profile代码即为每组8帧并复用正式前缀。原配置SHA不改写，[计数修正记录](engineering_revisions/preflight_count001.json)与原生成器副本保留，未来生成器文字修正为8；不增加任何预测/拟合或改变科学条件。\n')
 lines.append('采用共同cooperative-exit-only恢复：正常运行不保留旧周期checkpoint。完整当前帧/4轮相机/淘汰完成后，安全退出才保存当前≤8帧raw K/V、相机、共同arena、ID/远端/描述子/RNG与不可回流输出归档；恢复核对全部28对/56tensor的shape/dtype/固定样本和RNG/完成输出身份，先删消费checkpoint再推进下一query。两次预检前缀恢复有实际日志，无已淘汰帧留在可读恢复暗库。突发未保存崩溃将停止并保留失败，不能悄悄重跑有效帧。本轮所有实际保存/恢复、失败若有和I/O均计入预算；唯一队列与actualwait见[workers](workers)。\n')
 lines.append('## 主终点、覆盖、逐地点与全部反例\n')
 lines.append('原远端选择和mask完全继承：t=33…95；s≥1、t−s≥32；210×280源深度前向min-z投影，目标有效像素一致容差max(.05m,.05目标深度)，覆盖最大者、平局最早s。原覆盖≥5%为189重访，不改20%阈值/不重选。先每序列重访相对旋转均值，再已定义地点等权；12地点ID保留，3NA不填0。配对bootstrap5000×12地点ID，seed20261114，区间条件于有定义抽样分母；各层/留出也保留NA。\n')
 rows=[]
 for i,m in enumerate(p['catalog']):
  rr={x['arm']:x for x in by if x['id']==m['id'] and x['group']=='revisit_coverage_ge5pct'};rows.append([m['place'],rr['T']['records'],n(rr['W']['relative_rotation_deg']),n(rr['T']['relative_rotation_deg']),n(rr['D']['relative_rotation_deg']),n(primary['per_place'][i])])
 lines.append(table(['地点','重访/63','W度','T度','D度','W−T度'],rows)+'\n')
 loo=[v for v in primary['leave_one_place_out_means'] if v is not None]
 lines.append(f"原756远端目标/12地点：189重访来自9处；原共视零覆盖{cov['zero_selected_covisibility']}、选定独占零支持{cov['zero_selected_exclusive']}均保留。主T/W正向地点{primary['improved_places']}/{primary['defined_places']}，逐一留出均值范围{n(min(loo))}–{n(max(loo))}度。全部远端真实时距{n(min(r['elapsed_seconds'] for r in sel),2)}–{n(max(r['elapsed_seconds'] for r in sel),2)}s，每对原索引gap/秒数/支持完整导出[selection.csv](scoring/selection.csv)。无GT平移方向定义的目标数{sum(r['GT_translation_near_zero'] for r in pairs if r['arm']=='W')}/756，只将该辅助量记NA，不删除主旋转。\n")
 lines.append('![固定主比较与全部地点](figures/primary.png)\n')
 lines.append('## 全帧深度、几何与支持分层\n')
 fm=['trajectory_rotation_anchor_deg','trajectory_center_error_raw_m','absolute_worldpoint_GT_error_raw_m','absrel','log_mse','centered_log_mse','bias_squared','RMSE_m','failure'];rows=[]
 for metric in fm:
  vals=[]
  for arm in ('W','T','D'):
   z=[r[metric] for r in by if r['arm']==arm and r['group']=='all96frames'];z=[v for v in z if v is not None];vals.append(n(np.mean(z) if z else None))
  rows.append([metric,*vals,estimate(comp['all96frames']['T_vs_W'][metric]),estimate(comp['all96frames']['D_vs_W'][metric])])
 lines.append(table(['全96帧（地点等权）','W','T','D','W−T [区间]','W−D [区间]'],rows)+'\n')
 lines.append('raw log-MSE、去均值结构误差、偏差平方与全图AbsRel分列，不用结构项掩盖整体深度损伤；不通过GT尺度/仿射拟合修复输出。轨迹与绝对点误差只作首帧刚体gauge对齐，保留raw尺度。跨时一致性用固定GT对应，并列两端绝对GT点误差及GT投影采样差，防止缩小/塌缩伪装一致。相机非有限保留180°旋转/failure；深度坏值保留异常，不能筛有限子集得漂亮区间。全部新原生有限性及失败原数见调用/逐帧表，不推断物理场景尺度已准确。\n')
 pm=['relative_rotation_deg','relative_translation_direction_deg','relative_translation_vector_error_raw_m','cross_time_point_consistency_raw_m','paired_absolute_GT_geometry_error_raw_m','failure'];rows=[]
 for group in ('all63far_targets','revisit_coverage_ge5pct','nonrevisit_lt5pct'):
  for metric in pm:
   row=[group,metric]
   for arm in ('W','T','D'):
    v=[r[metric] for r in by if r['arm']==arm and r['group']==group];v=[x for x in v if x is not None];row.append(n(np.mean(v) if v else None))
   row.append(estimate(comp[group]['T_vs_W'][metric]));rows.append(row)
 lines.append(table(['目标集合','辅助端点','W','T','D','W−T [区间]'],rows)+'\n')
 lines.append('![全帧深度代价和偏差](figures/depth_tradeoff.png)\n')
 rows=[]
 for scope in ('selected_pair_exclusive','all_far_union_exclusive'):
  for st in ('0','(0,5%]','(5,20%]','>20%','undefined'):
   z={r['arm']:r for r in strata if r['scope']==scope and r['stratum']==st and r['revisit_only']};rows.append([scope,st,z['W']['records'],z['W']['places'],n(z['W']['relative_rotation_deg']),n(z['T']['relative_rotation_deg']),n(z['D']['relative_rotation_deg'])])
 lines.append(table(['原独占定义（重访子集）','支持层','目标数','地点数','W度','T度','D度'],rows)+'\n')
 lines.append('两个独占定义均扣除原首帧＋最近7参考mask：selected-best-pair为主描述，all-far-union为另列诊断，不因T/D改用有利参照。未筛0/低支持；全部756目标和重访子集各层的全部指标/配对地点区间/原NA见[support_strata.csv](scoring/support_strata.csv)、[support_strata_paired.json](scoring/support_strata_paired.json)。分层不是因果或单调性证明。\n')
 for a,base in [('T','W'),('D','T'),('D','W')]:
  worst=[r for r in counter if r['candidate']==a and r['baseline']==base and r['revisit']][:5];lines.append(table([f'{a}/{base}最大重访反例','t/s','覆盖','独占','秒数','改善度'],[[r['id'],f"{r['t']}/{r['s']}",n(r['coverage'],3),n(r['exclusive_fraction'],3),n(r['elapsed_seconds'],2),n(r['rotation_improvement_deg'])] for r in worst])+'\n')
 lines.append('全部756×3比较含非重访、原零覆盖和正负个例完整保存[all_counterexamples.csv](scoring/all_counterexamples.csv)。\n\n![最大主比较反例与raw绝对几何](figures/counterexample.png)\n')
 lines.append('## 同一总历史字节上限与可恢复执行\n')
 lines.append(f"共同B在{B['registered_utc']}、任何T/D科学评分前登记为 **{B['B_total_history_bytes']:,}bytes={B['B_total_history_bytes']/2**30:.6f}GiB**。W8实测raw K/V＋相机{B['raw8_frame_KV_camera_bytes']:,}bytes，共同128KiB元数据arena，各组相同。B=3×(raw8＋arena)＋整个12流可读RGB库存{B['whole_selected_RGB_batch_backing_bytes']:,}bytes＋1MiB原生位置/RoPE/Python/RNG/ZIP/manifest与描述子临时对象余量。D描述子/时间索引不免费；W/T不靠冗余整帧副本让D额外放帧，所有组硬约束≤8完整保留帧。\n")
 rows=[]
 for arm in ('W','T','D'):
  rr=[r for r in c['per_stream'] if r['policy']==arm];maxforward=max(r['max_forward_history_bytes'] for r in rr);rows.append([arm,len(rr),n(maxforward/2**30,6),n((B['raw8_frame_KV_camera_bytes']+d.METADATA_ARENA_BYTES)*3/2**30,6),n(B['B_total_history_bytes']/2**30,6)])
 lines.append(table(['政策','本轮完整流数','实测forward最大GiB','GPU/CPU/disk三份raw＋arena GiB','共同总B GiB'],rows)+'\n')
 lines.append('native当前帧追加与out-of-place淘汰会短暂并存old9/new8 raw history；用2×9/8×raw8作保守上界，小于三份8帧save/restore上界。退出save：GPU完整8＋CPU完整8＋一个磁盘8，首次原子写无另一个旧checkpoint；restore同样三份，消费后先删除磁盘文件，继续时无旧淘汰状态暗库。CPU arena、描述子选择临时复制及原生grid/RoPE/Python/文件头均在共同余量内；恢复receipt实际file bytes见[history_bytes.json](history_bytes.json)。当前normalized/RoPE attention中间量、权重、当前激活和allocator reserve另报；原raw history独立复制不藏作免费临时激活。无普适显存对账<5%门槛。\n')
 lines.append('原历史RGB文件仍可读，故将整个选定batch最坏backing保守计给每政策，即使真实forward只单调读当前RGB。GT深度/pose标签、预测depth/pose和partial输出归档没有forward回流接口，不算预测历史；这些物理新增字节仍计DATA。无教师形成或免费离线cache。T/D从各自实际状态写入，缓存中的完整内容保留原始dtype/顺序/patch位置。\n')
 lines.append('![实际原生保留轨迹](figures/retention.png)\n')
 lines.append('## 完整成本、调用与资源\n')
 wc=c['W_fixed_case'];lines.append(f"本轮固定W首地点完整96帧的进程冷启动至最终输出观察wall为{wc['cold_process_to_complete96_output_wall_seconds']:.6f}s；其中imports＋模型加载envelope {wc['process_imports_plus_model_load_envelope_seconds']:.6f}s，加载完成至96输出{wc['preloaded_stream_plus_output_wall_seconds']:.6f}s。时间来自本机manager启动/segment UTC观察，不把OS文件cache热状态冒称物理uncached冷载。不是round032共享F/A流水线借来的独立测速。\n")
 lines.append(table(['固定首地点实际执行','预检8帧envelope s','正式88帧含冷载/restore/output s','合计s'],[[arm,n(c['fixed_first_place_complete_execution_envelopes'][arm]['prefix_envelope_seconds'],6),n(c['fixed_first_place_complete_execution_envelopes'][arm]['formal_worker_wall_seconds'],6),n(c['fixed_first_place_complete_execution_envelopes'][arm]['sum_prefix_and_formal_wall_seconds'],6)] for arm in ('T','D')])+'\n')
 lines.append('T/D首8预检在已加载同一权重上各自从空状态形成，prefix envelope含必要验证bookkeeping/原子保存；正式worker各自真实冷进程载入、restore及剩余88帧/最终输出再计。共同profile加载依赖只在真实总wall计一次并明确列出；这个人为验证分段不是最优无中断单进程部署，不能直接用不等验证开销宣称W/T/D加速。完整per-stream、per-frame原生查询、写入/选择/传输残差、加载/恢复/存储可重建账见[costs.json](costs.json)。纯attention读取延迟未另隔离，不冒充完整query或免费形成。\n')
 rows=[]
 for arm in ('W','T','D'):
  rr=[r for r in c['per_stream'] if r['policy']==arm]
  for key in ('aggregator_seconds','camera_depth_seconds','selection_eviction_seconds','RGB_read_seconds','frame_wall_seconds'):
   total=sum(r['frame_components_seconds'][key] for r in rr);rows.append([arm,key,n(total,4),n(total/(96*len(rr))*1000,4),96*len(rr)])
 lines.append(table(['政策','实测环节','合计s','均值ms/帧','帧数'],rows)+'\n')
 lines.append(f"D描述子tap envelope合计{c['descriptor_hook_envelope_seconds']:.6f}s，已含在D aggregator时间；CPU拷贝会同步尚未完成的patch encoder，不能解释成纯边际描述子开销。extra encoder=0，不等于描述子计算免费。完整保存I/O {c['exit_checkpoint_IO_seconds']:.6f}s；模型加载body合计{c['model_load_seconds']:.6f}s；各restore从model-loaded至验证/消费删除的envelope分别为{[round(r['seconds'],6) for r in c['recovery_setup_envelopes']]}s，含少量hook/setup，均计workerwall。\n")
 lines.append(table(['实际新调用','数量'],[[key,value] for key,value in calls.items() if isinstance(value,(int,float))])+'\n')
 lines.append(f"所有GPU worker实际Popen→wait合计 **{c['all_GPU_worker_hours']:.6f}小时≤2**，其中预检/成本 **{c['preflight_cost_seconds']/3600:.6f}小时≤.25**；{c['actual_worker_processes']}进程均actualwait。单worker NVML峰{c['peak_NVML_worker_MiB']/1024:.6f}GiB≤40，allocated峰{c['peak_GPU_allocated_bytes']/2**30:.6f}GiB、reserved峰{c['peak_GPU_reserved_bytes']/2**30:.6f}GiB、RSS峰{c['peak_RSS_worker_bytes']/2**30:.6f}GiB。DATA采样峰{c['DATA_peak_sampled_bytes']/2**30:.6f}GiB；包含全部25输出＋两原始prefixcheckpoint＋四并发完整exit与完整output原子暂存的保守界{c['DATA_conservative_bound_bytes']/2**30:.6f}GiB≤8，最终{c['DATA_final_bytes']/2**30:.6f}GiB。项目/临时与终态实数见[terminal_record.json](terminal_record.json)。没有删除旧结果、改GLIBC或下载。\n")
 lines.append('执行前宿主CUDA=True/8卡；restricted sandbox不可见/dev/nvidia不报驱动坏。实时全8卡占用/外部PID准入，实际用卡逐worker记录；max4并发来自8GiB所有exit峰＋永久输出边界，不是默认两卡。其他用户忙卡不因剩余显存被当空闲算力；新安全空卡可动态接独立未完流。manager纯CPU安全等待单列，不伪装GPU有效计算时间；加载/恢复/失败/IO均已在GPU workerwall计费。\n')
 lines.append('## 独立评分、边界与索引\n')
 z=s['independent'];lines.append(f"新T/D全部{z['unique_depth_records']}份depth独立NumPy/Torch指标复算最大差{z['NumPy_Torch_depth_metric_max_difference']:.3g}；FP64 resize最大差{z['NumPy_Torch_FP64_resize_max_difference']:.3g}；NumPy/官方Torch pose解码最大差{z['NumPy_official_Torch_pose_decode_max_difference']:.3g}；NumPy/SciPy旋转最大差{z['NumPy_SciPy_relative_rotation_max_difference_degrees']:.3g}度。旧W1152逐帧和756逐对指标直接继承，0旧像素重评分、0新GT投影/标签重算、0评分新模型调用。全部输入/标签/源/配置/旧权重继承身份见[execution_identity.json](execution_identity.json)。\n")
 lines.append('Full/A的旧不等预算背景和其raw深度代价单独保留在summary.background_F_A_inherited，没有重训/重跑，也不把Full当上界或用不稳定差距比例包装恢复率。本轮只检验固定配额原生整帧时间/内容选择；失败不推出高斯或空间结构普遍无用，阳性也不证明高斯机制。O3有界保存/O4原生精度—成本服务新目标；O2高斯公共结构暂不增加，O1语义澄清、O5渲染/O6 PE及所有旧关闭保持。C1自身在线写入、C2原生整帧、C3固定政策、C4不增PE、C5旧稀疏12×96、C6冻结均为本阶段选择，非永久用户要求。没有fresh或连续千帧部署确认。\n')
 lines.append('完整交付：[manifest.json](manifest.json)、[唯一实现交接](implementation_handoff.json)、[全部指标/区间与预定flags](scoring/summary.json)、[逐帧](scoring/frames.csv)、[逐对](scoring/pairs.csv)、[逐地点](scoring/by_place.csv)、[全反例](scoring/all_counterexamples.csv)、[同byte证明](history_bytes.json)、[成本](costs.json)、[调用](call_accounting.json)、[图实际审阅](figure_review.json)、[实际终态](terminal_record.json)。永久新预测路径/SHA从25份completed记录索引，旧W与标签保持原位置；不重复大权重或旧评分审计。科学分支交原Astra，本实现不自选后继。\n')
 (d.OUT/'report.md').write_text('\n'.join(lines)+'\n');d.write(d.OUT/'CPU_report_runtime.json',dict(complete=True,PID=os.getpid(),PID_namespace='Tool namespace',wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized(),new_model_calls=0));print(dict(report=str(d.OUT/'report.md'),bytes=(d.OUT/'report.md').stat().st_size))
if __name__=='__main__':main()
