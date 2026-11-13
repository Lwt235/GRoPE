# 双模型顺序研究循环

协调主任务：01a187a1-01bb-77c0-baeb-82fb3553f2ac，host remote-ssh-discovered:GPU；项目46195a45-9866-44a5-a0ae-adfeb022ca66；共享目录/home/liwt/GRoPE。

1. coordinator独占维护state.json和events.jsonl；角色只写自己的阶段handoff及获租约的研究文档/代码。创建/发信前记录intent，成功后记录返回ID；不因超时重复创建，先查任务和状态。
2. Sol首次任务仅只读确认待命，无代码/文档写入，无GPU任务。确认完成再创建Astra分析阶段。后续只复用这两个任务，按每次明确交接参数指定模型和推理强度。
3. analysis阶段只有Astra可更新研究分析、方案和living docs；implementation阶段只有Sol可更新实验代码、运行/结果和living docs。主任务仅读核查及写协调状态；handoff后释放研究写租约。
4. 第一份分析输出：round_001/analysis.md、round_001/plan.json、round_001/analysis_handoff.json。后续轮次用递增round_NNN。handoff包含phase、status、input_reports及身份、配置核验来源/边界、产物路径、有限预算、决定分支、需要返回分析的事项。不要伪称已验证后台模型。
5. 计划应分清O项核心、C项实现选择和临时范围；读相关历史，明确与旧失败尝试的实质区别。旧数据不能称fresh；独立确认要在候选明确后预留并锁定未用于开发的身份。没有值得执行的新假设可给出有依据的收束/阻塞建议，而不是无限维修或宣布科学目标已达到。
6. 实现阶段先检查已有产物和完成标记，按计划固定终点续跑，不重复拟合。常规修复版本留档；实质科学变更停止依赖工作并交回Astra。资源占用实时核验，保持其他任务；无权限自行修改其他用户进程。
7. 完成状态与完整结论、所有匹配对照、成本和适当核验一致后才进入下一阶段。分析角色不给实现角色直接抢先开工，统一经主任务记录和派发；双方获人类授权可向主任务发送结果和需判断事项。
8. 任务接口若只确认请求接收而不提供实际生效配置，记录requested/accepted/unverified，不擅自等同已生效。接口拒绝指定组合即停下该调度并保存错误，不换模型。

所有阶段必须遵守[人类授权](authorization.md)。协调记录不是新增数值门禁或用户审批流程。

## 实际调度接口

2026-11-13：桌面跨会话create_thread连续两次因本地MCP传输失败，无返回ID；改用可复用collaboration任务接口，模型和推理强度保持明确指定。新任务以task_path/agent_id标识，不伪称已创建独立桌面会话。`spawn_agent`采用fork_turns=none以实际传入模型/推理强度；后继复用followup_task，不创建同角色重复任务。主任务与角色之间用collaboration消息和结果读取；人类双向授权同样适用。可恢复记录位于state.json；恢复时先list_agents核对存活任务，不盲目重派。模型若不受支持或角色无法继续，则保留现场并报告，绝不替换模型。

配置记录边界：collaboration.followup_task不接受model/effort参数，后续复用以本任务创建时实际传入的固定配置为依据；每条交接同时记录creation_config与followup_parameter_support=false，不伪称每次followup都再次传入模型参数。当前接口也未返回后台effective配置；该项保持unverified。如果任务恢复会丢失配置且接口无法保证继续，先保存并报告，不能换模型。


## 后继实现角色Fast切换（2026-11-13新授权）

当前collaboration Sol/High先完成native_memory_context_pe1并实际结束turn。主任务在释放写租约后可用独立Codex CLI逐调用model/effort/service_tier覆盖，尝试创建新的Sol/High/Fast实现任务，捕获真实thread UUID及进程handle。先做有限只读bootstrap，再按Astra下一方案实施；不同时持两个实现写租约，不重做旧结果，不修改全局config。原Astra任务保留，主用文件/CLI JSON结果与resume提示完成授权双向交接。新Fast请求被接受不自动证明实际tier；优先核对接口暴露的metadata，无法核验则明确未知；组合不支持则保全现场如实报告。pending记录为sol_fast_replacement_pending.json，启动前禁止把它标成已开启。


## 持续总目标与每轮有限边界（2026-11-13）

用户要求继续建立循环目标；主任务发现goal服务当时返回null（原因未知），现已实际建立active持续研究总目标。每轮有限计划/预算/完成状态保存在对应round及state，不把单轮结项当作总体目标完成。当前唯一queue104639与postqueue113121继续，原Astra子智能体保留；当前Sol完整结束后再尝试显式Sol/high/fast独立任务，见sol_fast_replacement_pending.json。公开子任务进度/最终文本可由主转发并标来源，不能把主任务归纳称为模型原文。
