"""Complete frozen round037 contract results, without a successor decision."""
import os,time,resource
import numpy as np
import torch
from experiments import stream_patch_data as d
from experiments.stream_history_report import n,estimate,table

def main():
 start=time.monotonic();p=d.plan();s=d.read(d.OUT/'scoring/summary.json');c=d.read(d.OUT/'costs.json');by=d.read(d.OUT/'scoring/by_place.json');comp=s['comparisons'];primary=s['primary'];cov=s['coverage'];pre=d.read(d.OUT/'preflight.json');val=d.read(d.OUT/'interface_validation.json');counter=d.read(d.OUT/'scoring/all_counterexamples.json');sel=d.read(d.OUT/'scoring/selection.json');pairs=d.read(d.OUT/'scoring/pairs.json');keys=['C_vs_U','C_vs_W','U_vs_W'];lines=[]
 lines.append('# stream_patch_retention1：同实际历史字节的原生patch保存\n')
 lines.append(f"2026-11-15，round038完整实施[round037唯一合同]({d.SCIENCE})，SHA `{d.SCIENCE_SHA}`。**主I_space=E_U−E_C重访旋转改善 {estimate(primary)} 度**。U/C各12×96完整原生预测；W旧12流输出／评分继承，仅固定首地点一次96帧匹配成本重放。0训练／下载，所有数据used development。完整结果交原Astra作取舍，不自动高斯／P2，不宣称fresh或总体成功。\n")
 lines.append(table(['固定比较（正值误差降低）','重访旋转度 [95%配对区间]','全帧AbsRel改善 [区间]','旋转材料flag','明显AbsRel代价flag'],[[k,estimate(comp['revisit_coverage_ge5pct'][k]['relative_rotation_deg']),estimate(comp['all96frames'][k]['absrel']),s['predefined_flags'][k]['practical_rotation'],s['predefined_flags'][k]['obvious_AbsRel_cost']] for k in keys])+'\n')
 lines.append('≥1°且地点配对下界>0、AbsRel增加≥.01且区间支持退步是预定开发投入标记，不是普适门槛／等价／非劣性证明。C/U始终是主比较，C/W和U/W净用途分别保留；不以较差对照上的增量替代原生净用途，也不把深度辅助换成旋转主成功。当前总体目标由[主控制的canonical目标](../autonomous_research_loop/goal_revision_20261115/revised_goal.md)与[round037科学取舍](../autonomous_research_loop/round_037/science_plan.md)限定，本实现不改goal/state/events。\n')
 lines.append('## 固定假设、政策和实际源位置\n')
 lines.append('检验独立原生patch束能否在有界、自身反馈的流中保留长间隔旋转用途，并且预测3D中心覆盖是否比普通分散保存更有用。竞争解释包括普通样本已足够、预测几何／离群点误导覆盖、原生上下文不能这样稀疏保留或近期容量削减代价仍更大。相对已关闭T/D整帧政策，此次改变不可分割保存单位，不换描述子或扫描配额；相对round036无完整F教师／W影子／条件交叉。旧长历史收益及raw深度代价、旧FPS静态池化正负和PE家族关闭均保留。\n')
 lines.append(table(['政策','patch历史','特殊token／相机'],[['W','首帧＋最近7完整原生帧，旧输出／评分继承','原首帧＋最近7；每次camera4轮'],['U','首帧＋最近3完整patch，远端M个固定64b哈希最小者','5特殊token另保首帧＋最近7；自身相机同规则'],['C','与U同容量／候选；近期中心锚定的欧氏FPS远端','同U规则，实际隐藏值各组自己形成']])+'\n')
 lines.append(f"共同M在效果前一次固定为 **{p['M']}**。来源token ID=1041×frame＋localtoken（特殊0…4、patch5…1040）；全24层同一patch的raw K/V作为束保留，未融合、质量为1。U以splitmix64(ID xor20261115)无符号升序取最小M，哈希碰撞按ID。C每次仅从自身已保留远端＋刚退出近期patch选择，以首帧／最近3完整patch的有效中心为锚，依次取到已覆盖集合最近欧氏距离最大者，平局按源ID；距离用FP64平方欧氏，输入中心FP32。未满M全留；无效候选不进距离，若有效不足M则用同U优先级补足，无有效锚时初距无穷且按ID选首个。没有额外confidence、尺度或阈值筛选。\n")
 lines.append('每组当前原生depth在14×14 patch内排序，取中间两值均值作为中位深度；射线经过原图patch像素中心(x=14c+7,y=14r+7)，用该组当前FP32预测相机／内参反投影到预测世界系。非有限／非正中位深度或非有限相机使中心无效，不删输入或指标。中心在写入时固定，U同样得到并付费；当前RGB及未来状态均由各自保留历史形成，另一组、GT、Full与淘汰库从不进入forward。C/U是在线政策总效应，内容与未来几何可随之不同，不当固定内容的纯几何因果或准确物理中心证据。\n')
 lines.append('全24层保留未归一化raw K/V；每个保留token以自身原source二维位置做原qk_norm／RoPE，再原SDPA。特殊位置仍(0,0)，patch为原(row+1,col+1)；按来源ID排序只改变过去集合，不重投影，不加log n／新PE／融合。当前1041token、全部24层及4轮camera均原生完整计算。聚合器全局attention的packed接口使past具有单个packed维，非首帧特殊embedding与原生相同；不规则key positions逐条保留，不把整帧pos重复到错误patch。相机和特殊token各自首帧＋最近7；patch首帧＋最近3＋远端M，各类统一在整帧结束后淘汰。\n')
 lines.append(f"官方提交 `{p['official_commit']}`，5,026,743,569-byte权重继承SHA `{p['model_sha256_inherited']}`，未重新大哈希。完整518×392 RGB，BF16聚合器、FP32 camera/depth、原SDPA、冻结eval。旧配置SHA `{p['old_configuration_sha256']}`，新配置SHA `{d.sha(d.OUT/'configuration.json')}`；原12×96身份、时间、c2w/K/标签全继承，GT仅离线。\n")
 lines.append('## 必要验证、恢复与调用身份\n')
 lines.append('固定首地点先作唯一W96完整成本重放，再完成U96/C96并复用正式输出。W96与旧W depth/pose最大差0；U/C容量尚未淘汰的前7帧，原生输出各差0。没有用旧W像素指标重复验收。C的单次5点GPU FPS toy核对精确欧氏／对称平局选取；正式首帧自身位置与完整过去同序时可回放原接口。恢复在U00整帧47后完整保存并actualexit75，同一fit由48恢复，已完成query无重复。恢复核对全部K/V和camera shape/dtype/固定样本、源ID／中心样本和CPU/CUDA RNG，消费checkpoint在下一forward前删除。全部逐帧来源保留／哈希／不复活淘汰ID的CPU独立核对见interface_validation。\n')
 lines.append('正常运行不留周期旧checkpoint；cooperative退出才保存当前精确状态、RNG及输出。突发未保存错误停止并保留失败，不悄悄回滚重跑。profile的局部分段回放摘要不是最终验证来源，CPU使用已完成归档前缀一次核对；普通metadata002保留profile源码并给未开始formal增加begin/complete调用记录，没有改模型路径。一次CPU配置helper在模型／数据工作前遇到旧字段缺失，按同一catalog求实际RGB总bytes修复，记录CPU_configuration001。\n')
 lines.append('## 主指标、地点、NA与全部原反例\n')
 lines.append('原t=33…95，s≥1、t−s≥32；原210×280 GT前向min-z共视max(.05m,.05目标深度)、最大覆盖／平局最早source、≥5%重访和20%层完全继承，0新投影／标签重选。先各地点重访误差均值再已定义地点等权。配对bootstrap5000次重采样全部12地点ID，seed20261114；NA不填0，区间条件于有定义抽样分母，所有留出地点均保留。\n')
 rows=[]
 for i,m in enumerate(p['catalog']):
  z={a:next(r for r in by if r['id']==m['id'] and r['arm']==a and r['group']=='revisit_coverage_ge5pct') for a in ('W','U','C')};rows.append([m['place'],z['W']['records'],*[n(z[a]['relative_rotation_deg']) for a in ('W','U','C')],n(primary['per_place'][i])])
 lines.append(table(['地点','重访/63','W°','U°','C°','U−C°'],rows)+'\n');loo=[v for v in primary['leave_one_place_out_means'] if v is not None]
 lines.append(f"756远端目标，189重访来自9/12地点；三处NA、{cov['zero_selected_covisibility']}零共视、{cov['zero_selected_exclusive']}选定独占零支持均保留。主正向{primary['improved_places']}/{primary['defined_places']}，逐一留出均值范围{n(min(loo))}…{n(max(loo))}°；远端真实时距{n(min(r['elapsed_seconds'] for r in sel),2)}…{n(max(r['elapsed_seconds'] for r in sel),2)}s。GT平移近零仅方向辅助记NA，目标数{sum(r['GT_translation_near_zero'] for r in pairs if r['arm']=='W')}/756，不删主旋转。\n")
 lines.append('![固定比较和全部地点](figures/primary.png)\n')
 lines.append('## 全帧raw深度、平移和几何\n')
 fm=['trajectory_rotation_anchor_deg','trajectory_center_error_raw_m','absolute_worldpoint_GT_error_raw_m','absrel','log_mse','centered_log_mse','bias_squared','RMSE_m','failure'];rows=[]
 for metric in fm:
  z=[n(np.mean([r[metric] for r in by if r['arm']==a and r['group']=='all96frames' and r[metric] is not None])) for a in ('W','U','C')];rows.append([metric,*z,*[estimate(comp['all96frames'][k][metric]) for k in keys]])
 lines.append(table(['全96帧地点等权','W','U','C','U−C [区间]','W−C [区间]','W−U [区间]'],rows)+'\n')
 lines.append('raw log-MSE=去均值结构＋偏差平方，各项／完整AbsRel／RMSE／坏像素和失败均保留。只作首帧刚体gauge对齐，无GT尺度／仿射拟合。跨时点一致性并列绝对GT几何和GT投影采样差，避免塌缩伪装一致；这些是自身因果在线轨迹，仍非准确物理地图或连续千帧确认。\n')
 rows=[]
 for group in ('all63far_targets','revisit_coverage_ge5pct','nonrevisit_lt5pct'):
  for metric in ('relative_rotation_deg','relative_translation_direction_deg','relative_translation_vector_error_raw_m','cross_time_point_consistency_raw_m','paired_absolute_GT_geometry_error_raw_m','failure'):
   rows.append([group,metric,*[estimate(comp[group][k][metric]) for k in keys]])
 lines.append(table(['目标集合','指标','I_space','C/W','U/W'],rows)+'\n');lines.append('![全帧深度代价](figures/depth.png)\n')
 lines.append('## 原支持层和完整反例\n');strata=d.read(d.OUT/'scoring/support_strata.json');paired=d.read(d.OUT/'scoring/support_strata_paired.json');rows=[]
 for scope in ('selected_pair_exclusive','all_far_union_exclusive'):
  for st in ('0','(0,5%]','(5,20%]','>20%','undefined'):
   rr=next(r for r in strata if r['arm']=='W' and r['scope']==scope and r['stratum']==st and r['revisit_only']);z=[next(r for r in paired if r['candidate']==a and r['baseline']==b and r['scope']==scope and r['stratum']==st and r['revisit_only'])['effects']['relative_rotation_deg'] for a,b in [('C','U'),('C','W'),('U','W')]];rows.append([scope,st,rr['records'],rr['places'],*[estimate(v) for v in z]])
 lines.append(table(['原定义（重访子集）','支持层','记录数','地点数','C/U','C/W','U/W'],rows)+'\n')
 lines.append('两独占定义仍扣除原W首帧＋最近7的参考mask，不换为有利的U/C覆盖；selected最佳source与all-far-union分列。全部756目标及重访子集、每层全部指标和配对地点区间在support_strata_paired.json，0／undefined不删。层内相关不是因果／单调性结论，不调阈值。\n')
 for a,b in [('C','U'),('C','W'),('U','W')]:
  rr=[r for r in counter if r['candidate']==a and r['baseline']==b and r['revisit']][:5];lines.append(table([f'{a}/{b}最大重访反例','t/s','覆盖','独占','秒数','旋转改善°'],[[r['id'],f"{r['t']}/{r['s']}",n(r['coverage'],3),n(r['exclusive_fraction'],3),n(r['elapsed_seconds'],2),n(r['rotation_improvement_deg'])] for r in rr])+'\n')
 lines.append('全部756×3比较的原行、正负个例／零覆盖和非重访均保存all_counterexamples.csv，不只挑最佳／最差。![最大C/U反例](figures/counterexample.png)\n')
 lines.append('## 实际同byte上限、所有副本与淘汰\n')
 lines.append(f"固定B=**{d.B:,}bytes**。M={p['M']}时单份最大状态payload上界{p['state_copy_bound_bytes']:,}bytes；每个保留token包含24层raw K/V {d.RAW:,}bytes＋int64源ID8＋FP32中心12＋valid1，来源位置从ID无损派生；最大patch为4×1036＋M，特殊为8×5，相机{d.CAM:,}bytes，另同128KiB arena。三份GPU／CPU／单个原子disk状态＋全部12可读RGB {p['RGB_readable_batch_backing_bytes']:,}＋8MiB grid／RoPE／Python／索引／FPS距离／RNG／ZIP余量，总 **{p['total_history_bound_bytes']:,}≤B**；M+1已超限。U/C同一副本／恢复策略，不以无效中心减少账或暗增容量，未满容量时仍按最大容量比较。\n")
 lines.append('普通forward每层原生追加后，旧GPU K/V在完整aggregator返回后立即释放，再构造新的保留state；old＋appended与appended＋selected各峰都低于三份最大state，不额外保存旧库。camera旧引用同样在淘汰前释放。save时GPU当前state＋CPU精确copy＋唯一新disk原子文件；继续前消费并删除checkpoint，淘汰raw不在旧可读库。全部序列的源ID轨迹只作输出provenance，不被forward读取。native归一化／RoPE中间量、当前激活／模型／allocator reserve另报，原raw历史副本不藏为免费activation。8MiB共同余量覆盖FPS当前距离／索引及GPU／CPU metadata临时对象；原始完整RGB库存保守计给每组，实际只按当前顺序读取一次。\n')
 lines.append('预测depth／pose、原GT与输出provenance／partial归档均无forward回流，不作预测历史输入；物理新增仍全部计DATA。无教师／影子／他组／已淘汰raw内容暗库。源码、实际state-file bytes与恢复、原生grid／实测forward总bytes见history_bytes与execution_identity。W原完整8帧自己的3份／128KiB／1MiB旧预算B保持，新增W只作同机成本回放。\n')
 lines.append('![真实patch来源保留轨迹](figures/retention.png)\n')
 lines.append('## 完整形成、更新、读取、传输和资源成本\n')
 a=d.read(d.OUT/'profiling_admission.json');lines.append(f"科学评分前{a['registered_utc']}，固定首地点W/U/C含4次实际冷进程（U完整恢复分段）共{a['fixed_first_stream_worker_seconds']:.6f}s；U {a['per_U_C_case_worker_seconds']['U']:.6f}s，C {a['per_U_C_case_worker_seconds']['C']:.6f}s。含25%余量的全轮投影{a['total_projected_seconds']:.6f}s≤14400。C预检包含CUDA扩展首次编译和唯一必要FPS toy检查，U含人为恢复开销，均实际收费，输出复用正式；不以不等验证／编译开销声称方法速度排名，冷进程也不保证物理uncached权重。\n")
 rows=[]
 for arm in ('W','U','C'):
  rr=[r for r in c['per_stream'] if r['policy']==arm]
  for key in ('aggregator_seconds','camera_depth_seconds','center_selection_eviction_seconds','RGB_read_seconds','frame_wall_seconds'):
   total=sum(r['components_seconds'][key] for r in rr);rows.append([arm,key,n(total,5),n(total/(96*len(rr))*1000,4),96*len(rr)])
 lines.append(table(['实际完整流','环节','总秒','均值ms/到达帧','帧数'],rows)+'\n')
 lines.append('形成包含当前完整encoder／24层raw写入与自身相机深度；更新包含同组中心构建、候选／GPU FPS或哈希／淘汰和实际copy；读取使用原norm／source RoPE／SDPA。完整query是当前RGB到aggregator、camera4轮、depth及真实选择／传输和输出，不把缓存attention当完整query。模块计时残差、CPU/GPU同步和archive写入保留于完整帧／进程wall，选择envelope不能冒充孤立纯GPU距离kernel时间。CPU-GPU-DATA真实转移、RNG／源ID小copy、编译／保存／恢复全部进程计费，没有免费cache或教师。转移字节及其可观测范围在costs.transfer_ledger；未另加profiler／benchmark重放。\n')
 lines.append(table(['固定首地点实际全部进程wall','秒'],[[r['policy'],n(r['actual_cold_process_worker_seconds'],6)] for r in c['fixed_first_case']])+'\n')
 lines.append(table(['实际新调用','数量'],[[k,v] for k,v in d.read(d.OUT/'call_accounting.json').items() if isinstance(v,(int,float))])+'\n')
 lines.append(f"GPU所有Popen→actualwait合计 **{c['all_GPU_worker_hours']:.6f}worker小时≤4**，预检／成本{c['preflight_cost_seconds']/3600:.6f}小时≤.5；{c['actual_worker_processes']}进程均已wait，失败／恢复／加载／I-O未扣除。模型加载body {c['model_load_body_seconds']:.6f}s，完整保存IO {c['checkpoint_IO_seconds']:.6f}s。单worker NVML峰{c['peak_NVML_worker_MiB']/1024:.6f}GiB≤40，allocated {c['peak_GPU_allocated_bytes']/2**30:.6f}、reserved {c['peak_GPU_reserved_bytes']/2**30:.6f}GiB，RSS峰{c['peak_RSS_worker_bytes']/2**30:.6f}GiB。DATA采样峰{c['DATA_peak_sampled_bytes']/2**30:.6f}GiB；全部25永久输出＋保守8worker同时完整exit/输出原子暂存＋扩展编译backing的峰界{c['DATA_conservative_bound_bytes']/2**30:.6f}GiB≤12，最终{c['DATA_final_bytes']/2**30:.6f}GiB。project≤1GiB与实际终态见terminal_record。\n")
 lines.append('宿主CUDA／NVML实际准入，restricted exec看不到设备不被误诊驱动故障；全8卡稳定空闲观察，独立U/C剩余流按CPU/RAM／DATA峰动态用所有安全卡，无继承单worker／两卡上限。忙卡不因余显存当空算力，外部任务保护只礼让自有worker；唯一manager和actualwait收据保留，等待不重算。0下载，不改GLIBC／系统库，不删旧结果。\n')
 lines.append('## 独立评价、O/C范围及交接\n')
 z=s['independent'];lines.append(f"新U/C全部{z['unique_depth_records']}份depth独立NumPy／Torch复算最大差{z['NumPy_Torch_depth_metric_max_difference']:.3g}，resize最大差{z['NumPy_Torch_FP64_resize_max_difference']:.3g}，NumPy／官方Torch pose差{z['NumPy_official_Torch_pose_decode_max_difference']:.3g}，NumPy／SciPy旋转差{z['NumPy_SciPy_relative_rotation_max_difference_degrees']:.3g}°。原W1152逐帧＋756逐对指标逐行继承，0旧像素重评分／新标签投影／评分模型调用。直接原行配对bootstrap、{val['U_hash_policy_frames']}帧U哈希独立选择和{val['causal_source_ID_lineage_frames']}帧无淘汰ID复活核对见interface_validation。\n")
 lines.append('本轮仅选择保存粒度与空间强基线，不是高斯／体素／真实STAC胜负，不证明各向异性或原生patch等于物理点。O3有界状态和O4原生精度—成本为核心，O2仅预测中心空间参照，O5渲染／O6 PE无新增；C1自身因果形成、C2全层未融合束、C3固定U/C淘汰、C4原二维位置、C5旧稀疏SUN3D、C6冻结均为临时选择。旧整帧T/D及所有PE家族关闭、旧条件旋转正证据及raw深度代价继续保留。材料flag仅决定科学后续值得判断的对象，结果完整交原Astra，不自选后继／扩容量／改锚或哈希，全部used，无独立确认或总体完成。\n')
 lines.append('完整交付：[唯一handoff](implementation_handoff.json)、[manifest](manifest.json)、[评分及flags](scoring/summary.json)、[每帧](scoring/frames.csv)、[每对](scoring/pairs.csv)、[地点](scoring/by_place.csv)、[支持配对](scoring/support_strata_paired.json)、[全反例](scoring/all_counterexamples.csv)、[byte与恢复](history_bytes.json)、[完整成本](costs.json)、[精确调用](call_accounting.json)、[接口](interface_validation.json)、[图实看](figure_review.json)、[实际终态](terminal_record.json)。永久预测SHA及路径由25份completed索引，原W/GT与旧结果保持原路径，模型大权重不重哈希。\n')
 (d.OUT/'report.md').write_text(('\n'.join(lines)+'\n').replace('(figures/',f'({d.OUT}/figures/'));d.write(d.OUT/'CPU_report_runtime.json',dict(complete=True,PID=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized(),new_model_calls=0));print(dict(report=str(d.OUT/'report.md')),flush=True)
if __name__=='__main__':main()
