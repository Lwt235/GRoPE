"""Render complete fixed-probe result report and compact tabular evidence."""
import os,time,json,csv,ast,resource
from pathlib import Path
import numpy as np
import torch
from experiments import stream_history_data as d

def n(x,digits=4):return 'NA' if x is None else f'{x:.{digits}f}'
def estimate(r):return f"{n(r.get('mean'))} [{', '.join(n(v) for v in r['interval95'])}]" if r.get('interval95') else 'NA'
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
def main():
 start=time.monotonic();p=d.plan();s=d.read(d.OUT/'scoring/summary.json');c=d.read(d.OUT/'costs.json');b=d.read(d.OUT/'history_bytes.json');calls=d.read(d.OUT/'call_accounting.json');by=d.read(d.OUT/'scoring/by_place.json');sel=d.read(d.OUT/'scoring/selection.json');strata=d.read(d.OUT/'scoring/support_strata.json');counter=d.read(d.OUT/'scoring/all_counterexamples.json')
 for name in ('frames','pairs','selection','by_place','support_strata','all_counterexamples'):
  rows=d.read(d.OUT/'scoring'/f'{name}.json');keys=list(dict.fromkeys(k for r in rows for k in r))
  with (d.OUT/'scoring'/f'{name}.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 # The actual numerical streaming step is unchanged by metadata/recovery fixes.
 original=d.OUT/'source_snapshot_P0/stream_history_run.py';current=d.ROOT/'experiments/stream_history_run.py'
 def function(path,name):return ast.dump(next(x for x in ast.parse(path.read_text()).body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name),include_attributes=False)
 same=function(original,'step')==function(current,'step');assert same
 identities=dict(science_plan=str(d.SCIENCE),science_plan_sha256=d.sha(d.SCIENCE),configuration_sha256=d.sha(d.OUT/'configuration.json'),source_files={str(f):d.sha(f) for f in sorted((d.ROOT/'experiments').glob('stream_history_*.py'))},P0_numerical_step_AST_equals_final=same,source_versions='P0 snapshots cover early workers. run_before_restore001.py plus controlled_restore001.json and resume002/pool snapshots preserve metadata/recovery changes. Numerical step unchanged; current script hash is not falsely assigned to every earlier worker.',large_weights_sha256_inherited=p['model_sha256_inherited'],large_weights_not_rehashed=True,official_commit=p['official_commit'],model_creation='gpt-6.1-sol/high',Fast_parameter_supported=False,effective_backend_configuration=None)
 d.write(d.OUT/'execution_identity_final.json',identities)
 cov=s['coverage'];r=s['primary'];lines=[]
 lines.append('# stream_history_value1：固定原生历史价值探针实施结果\n')
 lines.append(f"2026-11-14，round032实施已采纳的[round031唯一科学合同]({d.SCIENCE})（SHA256 `{d.SCIENCE_SHA}`）。**12地点×96不同RGB-D全部完成；主A/W相对旋转改善为 {estimate(r)} 度，正数有利A。** 这是已用开发数据上的固定历史政策比较，科学分支由原Astra评价；不预先宣称高斯胜出、同字节优胜、fresh确认或总体目标完成。\n")
 q=s['comparisons']['all96frames']['A_vs_W']['absrel'];z=s['comparisons']['revisit_coverage_ge5pct']['F_vs_A']['relative_rotation_deg']
 lines.append(f"主旋转收益与代价必须同时读：全帧AbsRel的W−A为{estimate(q)}，即A的绝对AbsRel增加{-q['mean']:.5f}；F/A重访旋转差{estimate(z)}度。重访绝对GT几何与跨时点一致性的A/W区间均跨0，不能从旋转单项推定全面几何改善。三处无5%重访地点保留NA，效应地点差异明显，不作fresh或通用性结论。\n")
 lines.append('## 问题、固定比较与身份\n')
 lines.append('本探针检验超过首帧＋最近7帧的原生聚合器历史是否改善预定义远端几何。竞争解释包括聚合器长历史有效、仅相机头长历史有效、近期与首帧已足够、长期缓存干扰或模型没有利用较早证据。主比较A/W固定相机缓存保留规则，各自隐藏值随自身轨迹产生，是政策总效应；F/W和F/A分开报告。A/W正向是否具有材料性、是否由单一地点或全局损伤取得，以及覆盖不足或区间宽的影响，均由Astra依合同判断；没有机械跨零门禁或自动P2。\n')
 lines.append('F为完整聚合器＋完整相机缓存；W两者均为首帧＋最近7个完成帧；A完整聚合器＋与W相同相机保留规则。每帧4轮相机精化全部完成后才淘汰，每帧4条相机状态和全部5个camera/register特殊token随帧保留，原始位置与全局frame index保持。W仍可能通过近期隐藏值间接携带更早上下文，不能称从未见历史。F不是精度上界。F/A真实共享同一聚合器输出和depth结果，分别运行相机头；不把共享工作重复报为实际调用。\n')
 lines.append(f"同一冻结官方权重 `{p['model_path']}`，5,026,743,569 bytes，继承SHA `{p['model_sha256_inherited']}`，不重读大权重做哈希。官方源码提交 `{p['official_commit']}`。官方518×392完整RGB，BF16 aggregator、FP32 camera/depth、原SDPA与位置，0训练/下载/方法选择。输入与配置事前锁定SHA `{d.sha(d.OUT/'configuration.json')}`；[configuration.json](configuration.json)保存所有路径、大小、mtime、小元数据SHA及原始pose ordinal。全部12既有地点按可用匹配RGB-D时间顺序均匀取96不同帧，不看重访率或预测效果挑地点；原full pose文件提供统一坐标，没有拼接重锚clip poses。\n")
 lines.append(table(['ID','地点','可用匹配帧','原索引首—末','真实时间跨度(s)'],[[m['id'],m['place'],m['available_matched_frames'],f"{m['frames'][0]['index']}—{m['frames'][-1]['index']}",n((m['frames'][-1]['timestamp_us']-m['frames'][0]['timestamp_us'])/1e6,2)] for m in p['catalog']])+'\n')
 lines.append('## P0、恢复与独立验证\n')
 pre=d.read(d.OUT/'preflight.json');v=pre['unordered_all_original_position_no_merge']
 lines.append(f"[P0接口](preflight.json)：固定3帧管理Full与原官方streaming、容量覆盖全部3帧的Window，depth/pose最大差与RMS均0。全部raw K/V＋原位置、反序检索、不合并小例，同BF16/SDPA最大差{n(v['max_abs'],8)}、RMS{n(v['RMS'],10)}，参考RMS{n(v['reference_RMS'],8)}；有限精度差明示，不增普适1e−5门槛。固定首流96帧成本在查看任何科学效果前通过[实测准入](profiling_admission.json)，该预测作为首条正式结果复用。\n")
 lines.append('完整恢复采用不可变32帧full-KV delta chunks、最新W与F/W/A相机状态、精确RNG、完成输出和next-frame manifest。一次预定工程恢复在fidelity07/frame31安全退出，再从32续跑；[保存](controlled_restore_saved.json)与[验证](controlled_restore_verified.json)证明60对/120个KV tensor的shape/dtype/固定样本、三相机4轮状态与CPU/CUDA RNG一致，32份已有输出保留，没有重复query。数值step的AST与P0一致。\n')
 lines.append('另一次manager24955收SIGTERM，旧handler只记录requested_own_stop，发送来源未知，不能称用户主动暂停或科学失败；owner70299实际wait75，fidelity09保存到79。唯一后继manager57577从80补缺，再完成10/11；旧终结、版本及unknown保存在[工程记录](engineering_revisions/resume002.json)。普通yield/加载/恢复全部计账；没有丢弃已有效帧或重做旧轮。以后manager记录signal编号和UTC。\n')
 ind=s['independent'];vals=[d.read(q) for q in (d.OUT/'label_validation').glob('*.json')]
 lines.append(f"所有12条GT标签均有固定首目标的独立NumPy几何投影核对，mask最大不一致{max(x['independent_NumPy_consistent_mask_disagreements'] for x in vals)} pixels。[独立评分](scoring/summary.json)：{ind['unique_depth_records']}份唯一F/W depth帧全部NumPy/Torch复算，指标最大差{ind['NumPy_Torch_depth_metric_max_difference']:.3g}；FP64 resize最大差{ind['NumPy_Torch_FP64_resize_max_difference']:.3g}；全部pose的NumPy/官方Torch解码最大差{ind['NumPy_official_Torch_pose_decode_max_difference']:.3g}；NumPy/SciPy相对旋转最大差{ind['NumPy_SciPy_relative_rotation_max_difference_degrees']:.3g}度。0旧像素指标重算，0额外模型调用。\n")
 lines.append('## 主终点、分母与地点差异\n')
 lines.append('每目标t=33…95，候选s≠0且t−s≥32。GT深度在210×280网格前向min-z投影，误差≤max(0.05m,0.05目标深度)为一致；选最大目标有效像素覆盖，平局最早s。覆盖≥5%为重访，不以20%筛选。所有GT只用于离线配对、标签和评分，没有进入预测或缓存。主值先每地点重访对平均，再对已定义地点等权；全部12地点ID和NA分母保留。5000次配对地点bootstrap，seed20261114；缺失地点不填0，区间条件于有定义的重采样分母。\n')
 lines.append(table(['比较（前者为候选）','旋转改善度 [95%配对区间]','有定义地点'],[[k,estimate(s['comparisons']['revisit_coverage_ge5pct'][k]['relative_rotation_deg']),s['comparisons']['revisit_coverage_ge5pct'][k]['relative_rotation_deg']['defined_places']] for k in ('A_vs_W','F_vs_W','F_vs_A')])+'\n')
 rows=[]
 for m in p['catalog']:
  rr=[x for x in by if x['id']==m['id'] and x['group']=='revisit_coverage_ge5pct'];idx={x['arm']:x for x in rr};j=next(x for x in cov['per_place'] if x['id']==m['id']);i=[z['id'] for z in p['catalog']].index(m['id']);rows.append([m['place'],j['revisit_targets'],n(j['mean_coverage'],3),n(idx['F']['relative_rotation_deg']),n(idx['A']['relative_rotation_deg']),n(idx['W']['relative_rotation_deg']),n(r['per_place'][i])])
 lines.append(table(['地点','重访/63','平均覆盖','F误差度','A误差度','W误差度','W−A度'],rows)+'\n')
 loo=[x for x in r.get('leave_one_place_out_means',[]) if x is not None]
 lines.append(f"完整756远端目标，重访{cov['revisit_targets']}，有重访地点{cov['places_with_revisits']}/12；原共视零覆盖{cov['zero_selected_covisibility']}，选定远端独占零支持{cov['zero_selected_exclusive']}。主改善正向地点{r.get('improved_places',0)}/{r['defined_places']}；逐一留出地点的主均值范围{n(min(loo) if loo else None)}至{n(max(loo) if loo else None)}度。全部时距范围{n(min(x['elapsed_seconds'] for x in sel),2)}–{n(max(x['elapsed_seconds'] for x in sel),2)}秒，原索引gap与每对真实秒数完整保留。\n")
 lines.append('![主比较与全部地点](figures/primary.png)\n')
 lines.append('## 全体结果、几何防塌缩、支持层与反例\n')
 metrics=['trajectory_rotation_anchor_deg','trajectory_center_error_raw_m','absolute_worldpoint_GT_error_raw_m','absrel','log_mse','centered_log_mse','bias_squared','RMSE_m','failure'];rows=[]
 for k in metrics:
  vals=[]
  for arm in ('F','A','W'):
   v=[x[k] for x in by if x['arm']==arm and x['group']=='all96frames'];vals.append(n(np.mean([q for q in v if q is not None]) if any(q is not None for q in v) else None))
  rows.append([k,*vals,estimate(s['comparisons']['all96frames']['A_vs_W'][k])])
 lines.append(table(['全96帧端点（地点等权）','F','A','W','W−A及区间'],rows)+'\n')
 metrics=['relative_rotation_deg','relative_translation_direction_deg','relative_translation_vector_error_raw_m','cross_time_point_consistency_raw_m','paired_absolute_GT_geometry_error_raw_m','failure'];rows=[]
 for group in ('all63far_targets','revisit_coverage_ge5pct','nonrevisit_lt5pct'):
  for k in metrics:
   row=[group,k]
   for arm in ('F','A','W'):
    a=[x[k] for x in by if x['arm']==arm and x['group']==group];row.append(n(np.mean([v for v in a if v is not None]) if any(v is not None for v in a) else None))
   row.append(estimate(s['comparisons'][group]['A_vs_W'][k]));rows.append(row)
 lines.append(table(['目标集合','辅助端点','F','A','W','W−A及区间'],rows)+'\n')
 lines.append('轨迹和绝对3D点误差只消除首帧刚体gauge，不对任何方法拟合尺度；m为原SUN3D GT度量坐标下的raw native量。跨时一致性使用选定GT对应，另报对应两端绝对GT点误差和GT采样投影误差，不能用缩小/塌缩伪装一致。相对平移方向仅GT范数≤1e−6m时NA，不删主旋转；非零GT的零预测平移保留180°。相机非有限保留180°旋转与failure，深度异常保留失败/非有限错误而不偷偷筛有限子集。本次全部1152正式帧pose/depth有限，逐帧、逐对failure和全部raw/log空间/偏差分解均在[全数评分表](scoring/summary.json)。\n')
 rows=[]
 for scope in ('selected_pair_exclusive','all_far_union_exclusive'):
  for st in ('0','(0,5%]','(5,20%]','>20%','undefined'):
   rr=[x for x in strata if x['scope']==scope and x['stratum']==st and x['revisit_only']];idx={x['arm']:x for x in rr};rows.append([scope,st,idx['A']['records'],idx['A']['places'],n(idx['F']['relative_rotation_deg']),n(idx['A']['relative_rotation_deg']),n(idx['W']['relative_rotation_deg'])])
 lines.append(table(['独占定义','支持层（重访子集）','目标数','地点数','F度','A度','W度'],rows)+'\n')
 lines.append('选定best-pair独占量扣除首帧及W最近7帧各自一致mask的union；额外all-far-union独占量单独报告，不替换主配对。0及未定义层保留；全部远端目标、不仅重访的各层/辅助指标亦完整导出[支持分层](scoring/support_strata.csv)，每层A/W、F/W、F/A的配对地点区间及NA分母在[support_strata_paired.json](scoring/support_strata_paired.json)。分层和相关性是描述，不是单调性门槛或因果证明。\n')
 lines.append('![覆盖与零支持](figures/coverage.png)\n')
 worst=[x for x in counter if x['candidate']=='A' and x['baseline']=='W' and x['revisit']][:10]
 lines.append(table(['最大A/W反例','t/s','覆盖','独占','真实间隔s','W−A度'],[[x['id'],f"{x['t']}/{x['s']}",n(x['coverage'],3),n(x['exclusive_fraction'],3),n(x['elapsed_seconds'],2),n(x['rotation_improvement_deg'])] for x in worst])+'\n')
 lines.append('所有756目标×3比较，包括非重访、原零覆盖与正负反例均保留在[全反例](scoring/all_counterexamples.csv)，没有只报最大改善。\n\n![最大主比较反例](figures/counterexample.png)\n')
 lines.append('## 实际历史总字节、耗时与调用\n')
 rows=[]
 finalmem=[x for x in b['records'] if x['complete']];ref=finalmem[-1]
 for arm in ('F','A','W'):
  z=ref['arms'][arm];rows.append([arm,n(z['forward_history_measured_bytes']/2**30,6),n(z['own_recovery_copy_file_bytes']/2**30,6),n(z['total_operational_history_including_recovery_bytes']/2**30,6),n(z['conservative_total_with_readable_RGB_bytes']/2**30,6)])
 lines.append(table([f"96帧参考{ref['id']}",'实测forward GiB','独立恢复backing GiB','含恢复与manifest GiB','再含可读历史RGB GiB'],rows)+'\n')
 lines.append('表中正常驻留总数不是保存/恢复瞬时峰：checkpoint会生成完整CPU serialization副本，并在原子提交前短暂并存旧/新bounded文件；恢复会读取CPU full chunks并构建GPU状态。各路径另给保守上界＝forward＋3份own-recovery体积＋2份manifest，覆盖CPU staging和整个原子重复副本（full不可变chunks通常不会全部重复，故为宽上界，非声称实测达到）。参考W该上界'+n(ref['arms']['W']['conservative_history_peak_with_CPU_staging_and_atomic_duplicate_bytes']/2**30,6)+'GiB，另加可读历史RGB；[history_bytes.json](history_bytes.json)同时保留各phase字段。未来同预算系统比较必须固定/计入这些工程恢复副本，不能只取最小forward payload。\n')
 lines.append('B_forward8含W所有24层raw K/V、4个相机trunk层各帧4轮条目、特殊token、原位置/RoPE缓存、实际Python索引与容器元数据；逻辑估算和实际storage/metadata分别保留。运行期间独立GPU/磁盘恢复副本亦可被恢复入口读取，故B_operational额外计费；整份共享manifest保守计到每路径。完整F/A的full-KV共享storage/chunk在实际联合物理账只计一次，各路径依赖账各自计费。恢复file是暂存也不能免费。永久depth/pose评分归档与GT离线标签从不供forward读取，另报物理存储。原RGB文件仍物理可读，表中另给连同所有已见RGB的保守总数；实际forward只按严格递增t读当前RGB，没有历史重读，未来有历史重读权限的方法应以含该backing的上限比较。此阶段F/A和W预算不等，不宣称等byte优胜。\n')
 lines.append(f"原选定RGB库存共{b['source_RGB_archive_bytes']/2**20:.3f}MiB，depth库存{b['source_depth_archive_bytes']/2**20:.3f}MiB；旧权重/输入不复制为新DATA。单worker NVML峰{c['peak_NVML_worker_MiB']/1024:.4f}GiB≤40；PyTorch allocated峰{c['peak_GPU_allocated_bytes']/2**30:.4f}GiB，reserved峰{c['peak_GPU_reserved_bytes']/2**30:.4f}GiB，单worker RSS峰{c['peak_RSS_single_worker_bytes']/2**30:.4f}GiB。权重、临时激活、CUDA context和allocator reserve与持久历史分列，不要求NVML与KV账差<5%。没有改精度、CPU offload或长度救资源。\n")
 lines.append(f"所有GPU worker实际Popen到wait合计 **{c['all_worker_hours']:.6f}小时≤4**，其中P0 **{c['profiling_seconds']/3600:.6f}小时≤0.5**，GT-only GPU标签{c['phases']['labels']:.3f}s；{c['worker_processes']}个进程（含完整保存退出/恢复）均actualwait。模型进程加载合计{c['model_load_seconds']:.3f}s，checkpoint I/O {c['checkpoint_IO_seconds']:.3f}s。DATA采样峰{c['DATA_peak_sampled_bytes']/2**30:.6f}GiB≤16；最终{c['DATA_final_bytes']/2**30:.6f}GiB，项目目前{d.size(d.OUT)/2**20:.3f}MiB≤1GiB。采样峰不是连续NVML/磁盘理论上界；所有永久预测＋最大完整恢复集＋整份最大bounded/output重复保存＋一个完整预测原子暂存的保守工程上界{c['DATA_conservative_engineering_bound_bytes']/2**30:.6f}GiB≤16，最终资源收据另在终结记录，未删除旧结果。\n")
 rows=[]
 for k,v in c['stage_component_times'].items():rows.append([k,n(v['total_seconds'],3),n(v['mean_seconds']*1000,3),n(v['p95_seconds']*1000,3),v['records']])
 lines.append('完整checkpoint主清单有37次、198.425s；下表永久frame_cost归档中的checkpoint_seconds只有35次，因为两次跨退出的归档先于该次计时字段追加而提交。预算与主I/O总账使用完整逐事件journal，未把缺少的两次当免费。\n')
 lines.append(table(['完整实际模块/CPU环节','合计s','均值ms','p95ms','条数'],rows)+'\n')
 lines.append(f"联合完整三路径GPU worker总wall/1152帧为{c['all_worker_seconds']/1152:.6f}s（含P0、GT标签、所有加载、恢复、I/O，非纯native逐query延迟）；逐流联合完整wall、逐帧input/聚合器/4轮相机/depth/传输淘汰及checkpoint均在[costs.json](costs.json)。每路径F/A/W的模块时间总和逐流列出，F/A共享聚合器＋depth各自作为依赖收费，但实际只算一次，不能把两份依赖和加成实际联合耗时。当前测量含模块内部read＋write和同步/输出拷贝；没有额外forward隔离纯attention读取，也未测uncached-OS冷载或稳态独立pipeline；仅读取延迟为unknown，不能冒充免费核缓存或完整querybenchmark。历史形成是本次因果逐帧aggregator写入的累积过程，无独立的免费教师/形成依赖；首帧、完整prefix、query阶段、模型载入和恢复各自可从实测记录重建。\n")
 lines.append(table(['实际调用','数量'],[[k,v] for k,v in calls.items() if isinstance(v,(int,float))])+'\n')
 lines.append('![状态与完整模块成本](figures/cost_state.png)\n')
 lines.append('## 资源、范围与交付索引\n')
 lines.append('宿主CUDA通路验证成功后运行；restricted exec缺/dev/nvidia仅是沙箱可见性，不报驱动故障。实时观察所有8卡，实际只用安全空闲GPU4，其余卡外部忙；不把显存余量当空算力，也不为占卡增加训练/benchmark。单活跃stream由约10GiB精确恢复缓存＋原子暂存＋永久输出的16GiB DATA峰决定，而非默认两卡上限；GT标签沿同一有锁管理器接口顺序执行。所有外部PID只作占用监测，不读他人研究文件、不操作外部任务。旧GLIBC、原始数据和全部旧结果保持。\n')
 lines.append('O2/O3有界历史总字节与O4原生长间隔用途得到本次探针证据；O1语义澄清保持，O5渲染/O6参数PE是可选且须独立证明的组件，本轮未比较高斯/体素/STAC/PE。C1因果流、C2raw K/V、C3官方读取、C4不增PE、C5稀疏已用12地点、C6冻结是阶段实现选择/临时范围，不升格永久要求。96稀疏采样帧不等于连续千帧；位姿尺度和模型训练域限制、重访/独占覆盖、地点异质性及所有阴性保留。旧native_history1的−0.24% ROI结果、raw_history_information1远端独占稀少、受控遮挡正例及round029等关闭决定不改变；不以新目标改写历史。没有fresh数据或确认候选。\n')
 lines.append('核心产物：[产物总清单](manifest.json)、[独立评分](scoring/summary.json)、[全部逐帧](scoring/frames.csv)、[全部逐对](scoring/pairs.csv)、[GT锁定选择](scoring/selection.csv)、[逐地点](scoring/by_place.csv)、[原始候选覆盖](label_pairs)、[总字节](history_bytes.json)、[成本](costs.json)、[精确调用](call_accounting.json)、[源/配置身份](execution_identity_final.json)、[CPU评分](CPU_scoring_runtime.json)、[作业实际wait](workers)、[图审阅](figure_review.json)、[终结资源](terminal_record.json)、[唯一交接](implementation_handoff.json)。永久预测和标签路径/新文件SHA引用各自completed与label_pairs记录；不再重哈希旧大权重或复算旧分数。\n')
 (d.OUT/'report.md').write_text('\n'.join(lines)+'\n')
 d.write(d.OUT/'CPU_report_runtime.json',dict(complete=True,PID=os.getpid(),wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized(),new_model_calls=0))
 print(dict(report=str(d.OUT/'report.md'),bytes=(d.OUT/'report.md').stat().st_size))
if __name__=='__main__':main()
