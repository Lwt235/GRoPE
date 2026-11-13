# native_input_pe1执行接续

**完成（2026-11-13）：** 所有12训练/8448预测/检查/成本/图像审阅/报告和文档已完成。队列及收尾进程退出，所有权重持久化，专用RAM退休；不要重启下方旧运行步骤。有限目标可标complete，总体研究目标未达成。

用户已授权“设为目标并执行”；当前有限目标active。完成12次四组配对训练、8448新预测、成本/评分/匹配检查/两图实际审阅/科学报告及文档后才complete，不自动接续下一项。

更新2026-11-13：9/12拟合和6336预测完成，最后P/S/G第三种子在GPU1/3/4运行。前两配对种子描述`two_seed_description.json`：L/P/S/G主空间误差.02219917/.02249602/.02246308/.02242489；G/P低0.316%，G/L高1.017%，其他查询G/P低1.048%、G/L低3.918%，完整AbsRel G/L高11.79%。无区间、采用或新确认。S第二种子在661步完整保存后从GPU1转GPU6续训已完成2000连续步，记录`runtime_relocation_v1/record.json`；最后GPU1系重新空闲后按原规则准入。成本/最终评分仍待全部完成。

图的逐地点PE面板保持至少±1%的显示跨度以免夸大小差异，这不是实用/等效门槛。最终分析、成本、图和检查源码哈希另存`analysis_source_identity.json`，与正式训练锁区分，尚未出最终图。两个首种子描述属探索性中间记录。所有正式训练源码/plan继续与原锁一致。

2026-11-13接续：已完成6/12拟合、4224份预测；S/G第二种子及L第三种子在GPU1/3/4运行（GPU1新空闲，均按原准入规则）。首完整种子描述`first_seed_description.json`：L/P/S/G主空间误差0.02234237/.02223455/.02223519/.02223574，G/P高0.0054%，G/L低0.48%，其他查询G/L低6.38%，完整AbsRel高14.72%。无区间或最终取舍，不为微小差异改公式、温度、精度或预算。全部三种子仍须完成。

- 计划：`/home/liwt/GRoPE/docs/native_input_pe_execution_plan.md`。
- 数据/权重：数据根目录`GRoPE/gaussian_memory/native_input_pe_v1`，续训在`/dev/shm/liwt_native_input_pe1_1020`（实际UID以data.RAM为准）。
- Python：`/home/liwt/miniColmap/.venv/bin/python`，先import numpy再torch，host GPU操作require_escalated。系统GLIBC不可改。
- 44几何缓存、8项预检完成；峰值10.063GiB，零残差/新reader-Adam恢复差0，全48交互块/DPT和新增PE列有梯度。预检参数丢弃。
- 配置遗漏conditions在正式训练前补齐，记录`failures/pre_run_conditions_v1`。已生成正式配置锁；不要无记录修改锁定科学源码/plan。
- 队列：`experiments.native_input_pe_queue`，stdout `queue.log`，首波L/P/S种子20261110在GPU3/4/7，G和后续种子排队；最多3worker、2共享卡、14GiBcap/20GiB准入/6GiB余量。完整原生/reader/双AdamRAM续训，最终状态持久化SHA确认后才删除RAM冗余。其他用户进程不动。
- 收尾：`experiments.native_input_pe_finish`在等待12拟合/导出，后运行cost、score、checks、figures、cost_tables。完成数值产物仍需根agent实际view_image两图、写科学取舍/报告并同步living docs。
- 关键状态：`queue_status.json`、各`{L,P,S,G}_{seed}/progress.json/status.json`、`finish_status.json`，所有正式步数2000。
- 本轮新L为当前patch注意力，P/S/G同一保存内容，主开发原查询G/P；球形控制和对L净用途分开。Previous_L/R仅不同接口历史背景，不是匹配PE对照。
- 8448新评分独立NumPy复算，4224旧32地点行保留来源。旧开发不称新确认；sensor/jointvalidity/给定相机/静态来源边界保留。正结果不等于一般研究目标达成。
- 不使用subagents，未获授权；不额外下载、清历史数据或调参。
