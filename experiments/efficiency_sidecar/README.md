# Efficiency sidecar

这是用户授权的独立效率探索。主会话的研究入口和冻结训练文件保持不变。
本目录不会被主训练自动导入；不得把这里的结果直接覆盖主实验。

**2026-11-09当前状态：** 单项独占复测、完整旧val评价和融合＋编译组合测试均已完成，
见 [当前采用建议](ADOPTION_CURRENT.md) 与 [本轮完整结果](../../results/efficiency_sidecar/deployment1/report.md)。
组合训练步节时约10%，完整训练终点效果尚待下一项确需训练的任务配对；没有后台任务。
下方“等待独占”等描述为历史执行记录，不代表当前待办。

- `PLAN.md`：假设、对照、资源与科学边界。
- `adapters.py`：只修改传入的私有模型实例，参数名与结构不变。
- `check_cpu.py`：单线程、旧验证输入、描述子/全参数梯度/旧权重输出/CPU 微基准。
- `benchmark_gpu.py`：显式启动的 GPU 基准，主训练或评价未结束时拒绝启动。
- `write_report.py`：从完成的 CPU JSON 生成说明。

首批结果位于 `results/efficiency_sidecar/cpu1/`，脚本拒绝覆盖已有结果。
**没有后台 GPU 等待器或自动调度任务。** 不使用新确认数据，不保存新的科学模型。

第二步计划也已完成：`followup_gpu.py` 执行重复噪声复核及分段计时，
`cache_geometry.py` / `check_cache_cpu.py` / `cache_gpu.py` 验证单基元缓存。
结果见 `results/efficiency_sidecar/followup1/report.md`。当前决定保留几何重排候选，
缓存虽通过数值核对，但包含传输后的净收益不稳定，不默认接入主训练。

## 数学与实现

已有优化：末层无偏置时，将 `qᵀWz` 改为 `(Wᵀq)ᵀz`。
新几何原型：平移后的中心记为 x、精度矩阵记为 P，则

`(xi-xj)ᵀPi(xi-xj) = xiᵀPi xi - xiᵀ(Pi+Piᵀ)xj + xjᵀPi xj`。

将 query 的常数、线性和六个二次系数，与 key 的
`[1, x, y, z, x², y², z², xy, xz, yz]` 作矩阵乘法。
共同平移减少抵消误差；距离仍用原直接位移计算，完全重合中心的二次型显式置零。
保留 FP64 运算、原始求逆和原有 floor；不宣称浮点逐位等价或降低 O(N²) 复杂度。

额外内存候选：只重算首层 Linear+GELU；隐藏关系仍保存，并非完整 attention 重算。
固定球形控制使用相同 FP64 距离→原网络 dtype→径向运算顺序，直接构造最终输入。
适配器不支持需要几何输入梯度的用途；不能自动用于 RGB 或可学习高斯。

## GPU 阶段

首轮之后，用户明确授权与已有项目任务共享 GPU。受限共享初测使用
`--share-project-gpu`，执行约束见 `SHARED_PILOT.md`，结果单独保存在
`results/efficiency_sidecar/shared_gpu1/`。共享计时不能替代独占卡的正式效率证据。
以下为默认独占模式；它仍会在主队列未结束时拒绝启动。

主训练/自动评价结束且一整张 GPU 空闲后，由本会话显式启动；环境需设置
`CUDA_VISIBLE_DEVICES` 为该卡 UUID、`CUBLAS_WORKSPACE_CONFIG=:4096:8`，
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`，使用兼容 Python：

```bash
nice -n 15 /home/liwt/miniColmap/.venv/bin/python -u -m experiments.efficiency_sidecar.benchmark_gpu
```

脚本不选择或抢占卡，不修改主队列，要求至少 24 GiB 空闲并设置自身 20 GiB 上限；
运行时反复检查其他 GPU 进程，出现竞争时只停止自身并保留部分结果。
限定 10 分钟；3 个角色、4 个实现、平衡次序，分别测推理、前后向、含 AdamW 的步骤。
含 AdamW 的步骤仅改变内存副本；不更新源 checkpoint。
默认独占模式的大批次配置尚未完整执行；共享模式只覆盖较小批次。

CPU 结果不作为 GPU 性能结论。旧输入微基准也不替代完整训练吞吐、长期精度验证或新数据确认。

## 反向累加融合（第三步，已完成）

`BACKWARD_PLAN.md` 对应的算子定位、单项实现、数值检查、完整步骤交错计时和64步私有训练均已完成。
`fused_hidden_grad.py` 将四层共享位置隐藏特征的梯度在一个Triton算子中合并；只供显式适配私有模型。
`hidden_gpu.py` 提供 correctness / timing / stability 阶段，`report_backward.py` 汇总全部结果。
结果见 `results/efficiency_sidecar/backward1/report.md`。
共享GPU、B4/N512下，五维与Distance整步配对耗时下降中位分别为4.67%与4.00%，均8/8块正向；
峰值显存约多0.34GiB。64步loss有限且配对差异小，但不能替代完整精度确认。
保留为可选训练优化，不自动接入主任务，不声称新增推理加速或五维独有优势。
下一步为稳定资源下实际批次的速度/显存复测；首次Triton编译失败及修正身份已保留。

## 正式批次复测（第四步）

`FULLBATCH_PLAN.md` 固定 B16/N512 的显存、数值核对和完整步骤比较，沿用同一个融合核。
执行入口为 `fullbatch_gpu.py`，结果按 `fullbatch1/shared` 和 `fullbatch1/exclusive` 分开保存。
本次没有空闲卡，已使用用户授权的共享资源；独占复测不能由共享结果代替。
正式批次需要约9.3GiB张量显存，因此自身allocator上限为12GiB；不修改主训练。

独占卡可用时，设置 `CUDA_VISIBLE_DEVICES` 为该卡UUID以及前述确定性/单线程环境，运行：

```bash
nice -n 15 /home/liwt/miniColmap/.venv/bin/python -m experiments.efficiency_sidecar.fullbatch_gpu correctness --mode exclusive
nice -n 15 /home/liwt/miniColmap/.venv/bin/python -m experiments.efficiency_sidecar.fullbatch_gpu timing --mode exclusive
/home/liwt/miniColmap/.venv/bin/python -m experiments.efficiency_sidecar.report_fullbatch --mode exclusive
```

脚本不自动等待或抢占卡，存在其他GPU任务则退出；不覆盖已有阶段。
共享模式明确使用 `--mode shared`，只允许队列中可核对PID身份的本项目同卡任务。
完整实测与未完成项见 `results/efficiency_sidecar/fullbatch1/shared/report.md`。

## 两项新的前后向优化尝试（未采用）

用户允许继续在共享GPU上探索后，测试了两头专用contraction与一次共享hidden布局重排。
实现分别为 `relation_contraction.py`、`relation_layout.py`；固定计划分别为
`CONTRACTION_PLAN.md`、`LAYOUT_PLAN.md`，完整报告见
`results/efficiency_sidecar/relation_trials1/report.md`。

B4/N512、五维/Distance、推理/完整训练步骤、各8块平衡比较，两个候选各条件均0/8块加速。
因此不扩大到B16、不做组合或64步扩展，也不接入主训练。
专用核节省约0.37GiB训练显存，但训练和推理更慢；布局重排未获速度收益。
原输出及全参数梯度检查通过，但逐参数一步AdamW更新门槛存在失败，全部失败保留。
`correctness.json` 此处是后续诊断记录，不能解读为原始所有门槛均通过；
其中明确记录原失败身份、全部更新例外及更新后输出差异。
尚无新可采用加速；此前几何重排和融合梯度结果保持不变。

## 局部编译（可选推理/显存候选）

`compiled_runtime.py` 编译位置Linear+GELU与attention block，保留几何eager计算、FP32/FP64精度。
CUDA Graph与max-autotune关闭；未叠加此前梯度融合。计划见 `COMPILE_PLAN.md`。
`compile_gpu.py` 完成B4和B16的数值/8块配对计时，`compile_stability.py`完成64步诊断。
完整报告：`results/efficiency_sidecar/compile1/report.md`。

B16共享计时：五维/Distance推理耗时下降3.79%/19.80%，训练下降3.73%/3.29%，
两者训练峰值张量显存各少约1.52GiB。首次调用包含约十几秒编译与执行开销。
五维64步轨迹明显分叉，Distance很接近；`compile_trajectory_diagnostic.py`在固定相同权重后
验证输出/梯度仍接近，不能把轨迹差直接判为编译算错或准确率下降。
仅保留可选推理/省显存候选，训练终点效果待验证，不修改主任务。
最后报告附加了固定权重诊断；重生成时应保留该诊断和采用边界，不能只看计时表。
