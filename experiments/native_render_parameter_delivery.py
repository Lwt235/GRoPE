"""Human-readable finite result and evidence handoff; no model execution."""
import argparse
from experiments.native_render_parameter_data import *


def comparison(s,scope,pair,endpoint='missing/roi/centered_log_mse'):
    return s['comparisons'][scope+'/'+endpoint+'/'+pair]


def brief(v):
    lo,hi=v['interval95_percent']
    return f"{v['relative_error_reduction_percent']:+.4f}% [{lo:+.4f}, {hi:+.4f}]%"


def report():
    s=read(OUT/'scoring/summary.json');c=read(OUT/'cost_description.json');d=read(OUT/'method_decision.json');a=read(OUT/'scoring/independent_recomputation.json');cost=read(OUT/'cost.json');attempt=read(OUT/'training_attempt_accounting.json');visual=read(OUT/'figures/visual_review.json')
    assert all(x['complete'] for x in [s,c,d,a,cost,visual])
    lines=['# native_render_parameter_pe1：保留原Render的直接参数Q/K位置编码检验','',f'{now()}。9次配对拟合各2000步、6336新预测与4224继承评分、独立复算、200次完整暖查询成本及两图实际审阅已完成。结果交回同一Astra作方法取舍；实现阶段不选择下一科学方案或宣布总体目标达成。','','## 固定问题与方法','','本轮检验：在原Render内容、局部合成、覆盖与融合全部保留时，基元intrinsic log-cov作为原生Q/K位置编码，能否超出强中心字段并保持下游用途。竞争解释是中心集合/原Render已足够、球形范围足够、完整形状确有额外用途，或此固定Q/K接法不能使参数产生可用增量。与旧单次输入attention替换及局部/覆盖分解不同，本轮不替换内容读取、不以历史分解残差加做扫描。','','三组均有中心均值3、多尺度Fourier30、中心间离散二阶矩6、中心—射线7共46维。末六维P为零、S为同迹球形intrinsic log-cov、G为完整intrinsic log-cov；按原实际合成贡献绑定，FP64构建后FP32保存，保存旋转和正scale直接计算log-cov。共同中心间离散二阶矩不能算成基元协方差独有收益。','','唯一共享52→64→1024 MLP（68,928参数，GELU、无末层bias、末层零初始），e=a·PE(z)加入24个既有global attention的patch Q/K，在q_norm/k_norm和原2D RoPE之前；特殊token和零覆盖e=0，V无直接改动。原Render内容r、覆盖a、融合、原生frame/global/DPT保留；没有新增reader/head/gate。P有384个不活跃首层系数，S/G有效容量相同；原Render内容已含高斯信息，P不是移除全部高斯的表示。','','44份既有场景（32训练、12开发）、四个静态来源、2048基元及1024维内容只读复用。拟合查询[2,5,8,11]，其他查询[1,4,7,10]，full及中央约25%均值填充missing；448输入、210×210评分。全部是已用数据，其他查询不等于未见场景独立确认。','','种子20261110/11/12配对；每fit共同官方预训练重置，原fusion初值与actual schedule精确匹配，PE初始化隔离RNG。637,383,906原生+787,712原fusion+68,928 PE=638,240,546可训练参数；编码器/场景冻结。FP32参数/Adam、原BF16算术，batch2、固定2000步、原loss/两学习率/clip5，不按开发结果选检查点。','','主终点为开发原查询缺失ROI centered log-MSE。5000次配对seed/place重采样，seed20261113，地点/查询等权；3种子12开发地点范围有限。至少1%降误差、区间下界>0且三种子同向为预定主筛；材料性开发损伤为至少1%增加且区间支持。均值/训练片段权衡按范围报告，不施加微小负均值零容差。完整字段、终点与决定分支见[事前方案](../../../docs/native_render_parameter_pe_execution_plan.md)。','','## 完整开发主结果','','|方法|空间log-MSE|原始ROI log-MSE|完整全图AbsRel|','|---|---:|---:|---:|']
    for kind in ['Previous_L','Previous_R','P','S','G']:
        vals=[s['summaries']['development/original/'+ep+'/'+kind]['mean'] for ep in ['missing/roi/centered_log_mse','missing/roi/log_mse','full/whole/absrel']]
        lines.append('|'+kind+'|'+'|'.join(f'{x:.9f}' for x in vals)+'|')
    lines+=['','正值表示前者降误差。','','|比较|降误差及95%区间|三种子降误差%|','|---|---|---|']
    for pair in ['G-P','G-S','S-P','P-Previous_R','S-Previous_R','G-Previous_R','P-Previous_L','S-Previous_L','G-Previous_L']:
        v=comparison(s,'development/original',pair);lines.append('|'+pair+'|'+brief(v)+'|'+' / '.join(f'{x:+.4f}' for x in v['per_seed_relative_percent'])+'|')
    lines+=['','所有原始/偏差/支持分层/完整输入/逐地点反例均保留在[固定数表](report_tables.md)、[全部指标](scoring/all_summary_cells.csv)及[配对区间](scoring/summary.json)。继承R/Local是相同32地点与2000步的历史参照，不伪称本轮重新训练。','','## 查询与训练片段范围','','|范围|G/P|G/S|G/原Render|G/原Local|','|---|---|---|---|']
    for scope in ['development/heldout','development/pooled','train_all/original','train_all/heldout','train_original/heldout','train_added/heldout']:
        lines.append('|'+scope+'|'+'|'.join(brief(comparison(s,scope,pair)) for pair in ['G-P','G-S','G-Previous_R','G-Previous_L'])+'|')
    lines+=['','这些相关查询与训练片段结果不能改名为fresh确认；训练内改善不自动建立开发用途，训练内退步也不自动否决限定开发范围的候选。历史Render片段内正证据和此前输入attention阴性保留。','','## 预定经验条件与解释边界','']
    for kind in ['S','G']:
        v=d['empirical_prespecified_flags'][kind];lines.append(f"- {kind}：主开发筛选={v['primary_development_screen']}；存在预定材料性开发损伤={v['any_supported_material_development_harm']}；完整暖查询相对P增加{v['full_warm_query_increase_over_P_percent']:+.3f}%，≤10%条件={v['within_prespecified_10percent_time_condition']}。")
    lines+=['',f"G/S区间支持完整形状增量={d['empirical_prespecified_flags']['G']['G_over_S_interval_supported']}。这些是按事前规则复述的观测条件；方法采用、下一方案及是否值得独立确认交回Astra。均值很小或区间跨零不称严格等效，也不把单一接法阴性外推到所有高斯参数编码。原r/a未变隔离了新增参数PE，但不能唯一识别任何效果来自局部attention、优化、监督或几何中的哪一种细机制。",'', '## 正确性、恢复与成本','',f"两训练输入独立NumPy字段抽样差0；零PE与原R一致；G/S共6个丢弃pilot更新验证PE/native梯度，完整model/Adam/step/RNG/config/schedule恢复通过。V不直接变化由接线结构核验，不将结构trace的0伪称独立实测。九条最终日志连续1–2000，实际样本、初值及源码身份核验见[匹配检查](matched_checks.json)。6336新预测NumPy float64独立评分最大差{a['new_max_abs_metric_difference']:.3g}。",'', '普通预检修复两项：Adam恢复时旧状态重叠造成OOM，随后跳过已完成G阶段的清理变量错误；均保留版本并复用已完成阶段，没有增加pilot更新。预检实际17/计费18次分为pilot训练6、继承R回放4、零PE4、恢复实际3/计费4（一次OOM已计费未执行）；必要非训练预检实际7/计费8，不混同pilot或24回放上限。', '',f"正式阶段资源观测JSON误用list根导致控制器退出，原P/S之后意外退出，原因未独立确认。两份step1900完整状态安全，host身份/Adam/RNG核验后按相同schedule续接。最终保留{attempt['formal_retained_updates']}更新，实际尝试{attempt['formal_actual_attempts']}，丢弃{attempt['formal_discarded_updates']}（P6/S3），对应样本实际{attempt['actual_sample_presentations']}、保留36000；旧budget原件、9步日志及对账永久保留。全部恢复计入原8worker小时，未增加有效步数/fit或改变算法。",'',f"保守nontraining counter={c['nontraining_budget_counter_including_pilot_training']}（含pilot6）；语义非训练计费{c['semantic_nontraining_charged']}、实际{c['semantic_nontraining_executed_if_no_other_failure']}。其中固定新预测6336、成本200，继承4224只重评分。GPU总{c['GPU_worker_hours']:.6f} worker小时，含准备/预检/加载/训练/恢复/评价/成本；上限8。最多2worker、最多1外部共享卡，未操作他人进程。最终权重已永久SHA核验，本轮RAM冗余退休；数据盘新增{c['data_bytes_at_check']/2**30:.4f}GiB、项目{c['project_bytes_at_check']/2**30:.4f}GiB，hard26/2GiB，没有删除旧成果、下载或场景重建。检查点I/O已包含相应elapsed，不再次全部相加。",'', '|完整暖查询|中位ms|峰值GiB|','|---|---:|---:|']
    for kind in ['Render','P','S','G']:
        v=cost['summary'][kind];lines.append(f"|{kind}|{v['median_ms']:.3f}|{v['peak_GiB']:.3f}|")
    lines+=['','两固定query×每组5热身/20保留，4组总200次原生前向。已预处理RGB与保存scene驻GPU；包含冻结encoder、实际原内容渲染、所需贡献权重/FP64 PE字段/投影、原生输出；不以特征或PE缓存替代。排除磁盘/JPEG、相机估计、来源构建，原来源成本沿用父轮，不能写成零成本纯RGB。共享服务器测量不是独占或跨硬件性能结论。[原始完整成本](cost.json) · [全部成本](cost_description.json)。','','## 图像与大纲范围','','[结果图](figures/outcomes.png)和[深度案例](figures/depth_cases.png)均已实际审阅；具体可见现象、反例、色阶/掩码及局限见[实际审阅记录](figures/visual_review.json)。固定案例加最大三种子开发G−P反例，图像不能替代全量指标或宣称单例因果。','','本轮直接服务O6并联系O4/O5，复用O2/O3；C1/C2/C5/C6保留，C3为原Render，改变C4为参数Q/K编码。sensor/joint-validity、给定相机/尺度、静态四来源及depth任务是临时范围，非纯RGB或全writer/encoder联合训练。所有历史阴性与正线索保留，总体精炼有效方法及适当独立确认尚未达成；没有已核验fresh reserve，本轮不自动启动额外训练。']
    guard=c['resource_sidecar']['last']
    lines+=['','## 资源附录执行','',f"原10GiB工程余量因盘可用量下降而触发安全停止，第三种子P19/S22完整保存、零新增丢弃。Astra独立资源审查后，仅用resource_addendum_v1修改工程预留，不改原science/runtime plan、lock或模型身份。新的独立sidecar及互监CPU watchdog在两次相隔至少30秒H≥6.5GiB准入后，只恢复原控制器。H=实时available−尚需永久产物未分配量−0.5GiB暂存额度未占部分；按唯一allocated出版文件入账，partial消耗暂存，避免rename及正常写入重复扣空间。H<6GiB、实际free<5GiB、30秒无解释下降≥0.5GiB或监控/账本失败会先停本人队列，再完整保存退出本人worker。",'',f"原所有硬预算、永久权重路径与精度保持。守护持续至9fit/6336预测及200cost调用结束，再正常退出；最终数据盘可用{c['final_data_available_bytes']/2**30:.4f}GiB，末次H={guard['H']/2**30:.4f}GiB。资源等待不记GPU active时间，活动恢复及保存照计；没有删除旧成果/其他用户资产、复制旧权重或把tmpfs当永久final。资源调整不是新实验或基于不齐结果的方法选择。[资源附录](../../../results/gaussian_memory/autonomous_research_loop/round_002/resource_addendum_v1.json) · [侧账本](resource_sidecar/ledger.jsonl)。"]
    raw=comparison(s,'development/original','G-P','missing/roi/log_mse');full=comparison(s,'development/original','G-P','full/whole/absrel')
    bias={k:s['summaries']['development/original/missing/roi/bias_squared/'+k]['mean'] for k in ['P','G']}
    pos=lines.index('## 查询与训练片段范围')
    lines[pos:pos]=['## 原始误差、偏差与完整输出','',f"开发原查询G/P原始ROI降误差为{brief(raw)}；bias²均值P/G为{bias['P']:.9f}/{bias['G']:.9f}，与原始误差改善同时下降。主空间G/P均值方向则为误差增加2.935%，三种子均未正，区间跨零。偏差/共同标定改善是与这些结果相容的竞争解释；原始误差正结果予以保留，预定主空间终点及不确定性保持。",'',f"完整全图AbsRel的G/P降误差{brief(full)}，区间仍跨零。主空间G对继承R均值误差高2.592%，区间跨零，属于未决用途代价；没有材料性损伤的区间支持也不构成非劣性或用途保持证明。训练其他查询G对继承R低2.609%[0.812,4.902]%，对继承Local低43.460%[36.676,49.234]%，是已训练片段的有限用途证据。",'']
    lines+=['',f"完整PE G暖查询中位相对原Render增加{100*(cost['summary']['G']['median_ms']/cost['summary']['Render']['median_ms']-1):.3f}%；预定G/P≤10%条件仅描述新增形状相对同中心接口的成本，公共参数PE开销另行保留。",'', '附录首次运行又因30秒H下降约0.715GiB触发完整保存，P1975/S1993全部保留。退出阶段空cmdline/并发停止竞态及非75外层句柄已归档，CPU-only修复将观察者停止串行化、每身份只发一次SIGTERM，空cmdline不再发信号但等待真实退出；没有额外GPU预检或丢弃更新。容量恢复后原34546仅reap失败句柄并退出，锁释放后唯一资源启动包装器47778调用未修改原queue.main，双准入及修复守护持续到成本结束并正常退出。完整证据见[资源执行汇总](resource_execution_summary.json)和[恢复交接](resource_yield_handoff_v1.json)。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('Finite report written; no successor selected.')


def handoff():
    import shutil
    c=read(OUT/'cost_description.json');d=read(OUT/'method_decision.json');attempt=read(OUT/'training_attempt_accounting.json')
    assert read(OUT/'queue_status.json')['state']=='complete'
    assert read(OUT/'figures/visual_review.json')['complete']
    assert (OUT/'report.md').exists()
    docs=['docs/research_current.md','docs/research_outline_alignment.md','docs/gaussian_memory_evidence_scope.md','docs/native_render_parameter_pe_execution_plan.md']
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','implementation_handoff.json'] and not p.name.endswith('.lock')]
    metadata={str(p):dict(sha256=sha256(p),bytes=p.stat().st_size) for p in files}
    metadata.update({str(ROOT/p):dict(sha256=sha256(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in docs})
    source={str(p):sha256(p) for p in (ROOT/'experiments').glob('native_render_parameter_*.py')}
    source[str(ROOT/'models/gaussian_scene_parameter_pe.py')]=sha256(ROOT/'models/gaussian_scene_parameter_pe.py')
    weights={}
    for seed in SEEDS:
        for kind in KINDS:
            r=read(OUT/f'{kind}_{seed}/status.json');weights[str(checkpoint_path(kind,seed))]=dict(sha256=r['checkpoint_sha256'],bytes=r['bytes'],steps=2000,persistence_verified=True)
    write(OUT/'manifest.json',dict(complete=True,created_utc=now(),artifacts=metadata,source_sha256=source,final_weights=weights,identity_lock=str(OUT/'configuration_lock.json'),external_cache_identity='Geometry identities in configuration_lock and prediction identities in evaluation markers; no old asset copies.'))
    h=dict(phase='implementation',status='complete',experiment='native_render_parameter_pe1',round='round_002',task_path='/root/research_implementation_sol',created_utc=now(),creation_config=dict(model='gpt-6.1-sol',reasoning_effort='high',source='Explicit original spawn accepted; parent-provided request record'),followup_parameter_support=False,effective_model=None,effective_reasoning_effort=None,independently_verified_effective=False,desktop_session_created=False,input_analysis_handoff=dict(path=str(ROOT/'results/gaussian_memory/autonomous_research_loop/round_002/analysis_handoff.json'),sha256=sha256(ROOT/'results/gaussian_memory/autonomous_research_loop/round_002/analysis_handoff.json')),fixed_scientific_plan=dict(path=str(SCIENCE_PLAN),sha256=sha256(SCIENCE_PLAN)),configuration_lock=dict(path=str(OUT/'configuration_lock.json'),sha256=sha256(OUT/'configuration_lock.json')),manifest=dict(path=str(OUT/'manifest.json'),sha256=sha256(OUT/'manifest.json')),report=str(OUT/'report.md'),tables=str(OUT/'report_tables.md'),summary=str(OUT/'scoring/summary.json'),independent_metrics=str(OUT/'scoring/independent_recomputation.json'),figures=[str(OUT/'figures/outcomes.png'),str(OUT/'figures/depth_cases.png')],visual_review=str(OUT/'figures/visual_review.json'),cost=str(OUT/'cost.json'),cost_description=str(OUT/'cost_description.json'),training_attempt_accounting=attempt,budget_actual=dict(formal_fits=9,formal_retained_updates=18000,formal_attempts=attempt['formal_actual_attempts'],discarded_formal_updates=attempt['formal_discarded_updates'],pilot_discarded_updates=6,new_predictions=6336,inherited_scored_predictions=4224,nontraining_charged=c['semantic_nontraining_charged'],nontraining_executed=c['semantic_nontraining_executed_if_no_other_failure'],conservative_counter_including_pilot=c['nontraining_budget_counter_including_pilot_training'],GPU_worker_hours=c['GPU_worker_hours'],hard_GPU_worker_hours=8,data_bytes=c['data_bytes_at_check'],project_bytes=c['project_bytes_at_check'],hard_data_bytes=26*2**30,hard_project_bytes=2*2**30,new_downloads=0,old_assets_deleted=0),process_exit=dict(all_current_workers_exited=True,original_interrupted_child_exit_code_unknown=True,original_launch_controller_exitcode=1,original_child_unknown_details=str(OUT/'failures/runtime_observation_v3/record.json'),RAM_redundancy_retired=True),method_empirical_flags=d['empirical_prespecified_flags'],scientific_successor_selected=False,method_adopted=False,overall_research_goal_achieved=False,new_confirmation=False,remaining_required_implementation=[],remaining_analysis=['Return results to same Astra task for utility/shape/scope decision; no successor selected here.','Competing explanations: strong-center/original Render sufficiency, spherical range versus full shape, fixed Q/K-interface limitations, content/optimization/supervision and given-camera/sensor/used-data scope.','Any independent confirmation requires genuinely unused identity audit and separately fixed finite plan.'],ordinary_engineering_records=[str(OUT/'failures/preflight_restore_memory_v1'),str(OUT/'failures/preflight_cleanup_v2'),str(OUT/'failures/runtime_observation_v3'),str(OUT/'reporting_entry_note.json')],scientific_write_lease_released_when_turn_ends=True,coordinator_state_or_goal_modified=False)
    h.update(resource_addendum=c['resource_addendum'],resource_sidecar=c['resource_sidecar'],final_data_available_bytes=c['final_data_available_bytes'],final_project_available_bytes=c['final_project_available_bytes'],all_actual_native_forward_calls=c['all_actual_native_forward_calls'])
    h['ordinary_engineering_records'] += [str(OUT/'resource_yield_v1.json'),str(OUT/'resource_sidecar/yield_001/cpu_fix_validation.json'),str(OUT/'resource_sidecar/proc_backstop/startup_fix_v1.json'),str(OUT/'resource_sidecar/capacity_resume_v1/old_controller_retirement.json')]
    h['process_exit']['resource_guards_completed']=str(OUT/'resource_sidecar/complete.json')
    h['process_exit']['proc_backstop_completed']=str(OUT/'resource_sidecar/proc_backstop/complete.json')
    h['resource_execution_summary']=str(OUT/'resource_execution_summary.json')
    h['goal_service_status_last_reported_by_coordinator']='blocked; implementation did not call goal tools or claim backend resumed'
    write(OUT/'implementation_handoff.json',h)
    # Count the final handoff itself without creating a self-hash dependency.
    for _ in range(4):
        actual_project_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        assert actual_project_bytes<2*2**30
        if h['budget_actual']['project_bytes']==actual_project_bytes:break
        h['budget_actual']['project_bytes']=actual_project_bytes
        h['budget_actual']['project_bytes_note']='Final own artifact bytes including handoff; cost_description retains earlier separately labeled check.'
        write(OUT/'implementation_handoff.json',h)
    write(ROOT/'results/gaussian_memory/autonomous_research_loop/round_002/implementation_handoff_reference.json',dict(path=str(OUT/'implementation_handoff.json'),sha256=sha256(OUT/'implementation_handoff.json'),status='complete'))
    print(dict(handoff=str(OUT/'implementation_handoff.json'),sha256=sha256(OUT/'implementation_handoff.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--handoff',action='store_true');args=p.parse_args()
    handoff() if args.handoff else report()
