# 当前有限目标续接记录

**当前已完成。** 六拟合、4224新预测、独立评价、实际审阅与文档均完成，见report.md及method_decision.json。下面仅保留执行中的历史续接信息；不要重复启动。

2026-11-12。目标`native_training_coverage1`仍active；不要重建目标、重启已运行队列或把训练开始当成目标完成。

- 工作目录`/home/liwt/GRoPE`，Python`/home/liwt/miniColmap/.venv/bin/python`；主机CUDA/进程检查需实际host权限，默认sandbox中的nvidia-smi会失败。
- 固定方案见`docs/native_training_coverage_execution_plan.md`及本目录`plan.json`，科学源码锁在`configuration_lock.json`。32训练/12开发地点，6次Local/Render配对2000步，结构不改；旧12地点结果取自view_transfer1的A行，不重训。
- 新20份来源/高斯/编码/八查询缓存及预检完成。单位/来源深度配方旧样例回放差0，前后向预检峰值9.615GiB；正式峰值约10.42GiB。
- 队列：`experiments.native_coverage_queue`（exec session47870），实时状态`queue_status.json`，日志`logs/queue.log`；前三拟合Local10/Render10/Local11已完成2000步和导出，余Render11/Local12/Render12在执行。GPU实时安排可能变动，以队列为准。
- 后处理：`experiments.native_coverage_finish`（exec session18048），等6fits及全部导出后自动跑新20地点GT来源支持、独立评分、配对身份/持久化检查、两张图和固定数表/成本；状态`finish_status.json`。该进程**不会**代替实际图片审阅、科学取舍或最终文档更新。
- 所有原始/派生新数据位于数据盘`gaussian_memory/native_training_coverage_v1`；全精度Adam续训在`/dev/shm/liwt_native_training_coverage1_<uid>`，6份最终权重自动持久化到数据盘，验证后删除各自RAM冗余。
- 新预测目标4224：6模型×(32训练+12开发)×8查询×2条件；继承2304行。主终点固定开发原四查询缺失ROI空间误差，交互`(R−L)32−(R−L)12`，同时看32净用途及Render/Local自身绝对变化。新四查询/合并/原始误差/偏差/完整输入及支持条件保留。
- 后处理完成后：读取完整summary/数表；实际查看`figures/coverage.png`与`depth_cases.png`，写visual_review；重写当前工作report并写method_decision、completion_audit；同步research_current/outline/core/evidence_scope/autonomous/execution_plan；无剩余工作后才update_goal complete，报告实际用时。
- 当前report明确未完成。不得根据训练损失或部分种子改预算/选样/结构。旧coverage1阴性与view_transfer片段内55.59%而跨地点无可靠优势保留。O4/O5检验、O6未测；旧数据/sensor/给定相机/静态条件，非新确认。
- `metadata_clarifications_v1.json`仅澄清继承计划中的未用旧描述字段，实际32地点采样/预算/计数已正确锁定，未改配置或重启训练。
