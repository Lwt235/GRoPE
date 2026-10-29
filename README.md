# GRoPE：高斯几何与多视角 Attention 的初步验证

**最新：第二轮计划已执行。** [第二轮报告](round2_report.md) 包含共享随机身份对照、真实投影关系的渲染 RGB 匹配、独立误差消融、条件门控以及共用 RGB 几何预测器。具体协议见 [ROUND2_PROTOCOL.md](ROUND2_PROTOCOL.md)。第一轮报告仍保留，结论应结合第二轮更新阅读。

本仓库验证高斯中心、方向、尺度与位置不确定性对匹配的作用。现有实现是**数学检查与合成场景机制实验**，不是 WorldRoPE / HRoPE 的完整复现，也没有真实图像或新视角合成结果。

实验结论与方案调整见 [research_report.md](research_report.md)。原始逐场景结果、验证集选择、训练曲线与运行环境保存在 `results/`；每轮包含当时源代码快照及 SHA-256。

## 第二轮复现

```bash
python -m unittest discover -s tests -v
python -m experiments.launch_gpu --module experiments.identity_probe --out results/reproduced_identity
python -m experiments.launch_gpu --module experiments.rendered_probe --out results/reproduced_rendered
python -m experiments.launch_gpu --module experiments.rgb_geometry_probe --train-only --out results/reproduced_rgb_geometry
python -m experiments.rgb_geometry_probe --eval-only --out results/reproduced_rgb_geometry --matching results/reproduced_rendered
```

GPU 启动器会重新检查空闲设备；CPU 运行可直接调用各模块并传 `--device cpu`。完整默认配置为五种子、每匹配器 600 步，门控按验证集规则有条件运行。第一阶段身份探针沿用每模型 400 步。

现有结果的报告与审计命令：

```bash
python -m experiments.uncertainty_calibration
python -m experiments.round2_audit
python -m experiments.round2_summary
```

校准、审计及报告模块默认读取交付时的 `results/round2_*` 目录。重放与重新生成审计需要本地 checkpoint；这些 checkpoint 和大体积概率数组保留在本机并由 `.gitignore` 排除。逐场景指标、CSV、manifest、源代码快照及预览图已保存在结果目录。

本轮训练了一个**小型合成域 RGB 几何预测器**，不是预训练几何模型。HRoPE/WorldRoPE 只实现了注明范围的 Q/K 分数算子，未复现 V/输出处理、完整预测头或 NVS 训练。

## 环境与运行

当前已验证环境：Python 3.11.7、PyTorch 2.6.0+cu124、NumPy 1.26.4。无需下载模型或数据，不安装 CUDA 扩展，也不修改系统 GLIBC。

```bash
python -m unittest discover -s tests -v
python -m experiments.math_probe
python -m experiments.stability_probe
```

第一轮：13 个变体、5 个种子、每个模型 400 步。可以直接使用 CPU：

```bash
python -m experiments.matching_probe --device cpu --out results/reproduced_pilot
```

也可以由启动器检查 GPU 占用，按 7 → 0 顺序选择无计算任务且显存占用低于 128 MiB 的卡。没有空闲卡时停止，不抢占其他进程。Codex 沙箱可能隐藏设备，此时 GPU 命令需要经执行环境授权后运行。

```bash
python -m experiments.launch_gpu --out results/reproduced_pilot
```

第二轮按首轮发现加入强基线消融、平滑矩描述和支撑区域变化：

```bash
python -m experiments.launch_gpu --out results/reproduced_revision \
  --methods center,rope3d_axis,rope3d_oblique,expected_oblique_support,rope3d_axis_normal,rope3d_axis_scale,rope3d_axis_logcov,rope3d_axis_projected_logcov,rope3d_axis_bounded_logcov,rope3d_axis_moments,rope3d_axis_random_control,expected_axis_uncertainty_oracle_logcov
```

对已训练模型仅在验证集校准几何权重，再用全新测试场景评估：

```bash
python -m experiments.calibrate_probe
python -m experiments.summarize
```

`calibrate_probe` 默认读取 `results/revision/checkpoints/`。权重保留在本地，但 `.gitignore` 排除这些可重新生成的 checkpoint。若重新训练到其他目录，使用 `--checkpoints` 指定。

严格重现某轮历史代码时，从对应的 `results/<轮次>/source/` 中运行其模块；最终代码新增了 `view_support` 对照，首轮源代码快照保留原始 8 种场景。

结果审计修复了极端噪声下尺度特征取 log 的正数下限。训练过程未改动；前两轮指标由已有 checkpoint 重新评估，修复前指标保存在 `pre_numerical_audit/`，修复后评估代码在 `evaluation_source/`，记录见 `results/numerical_audit.json`。需要重建此审计时运行 `python -m experiments.audit_results`。

## 实验边界

- 每个模型只有 2,387 个登记参数；外观网络和初始化相同，几何距离只增加有效标量权重。所有模型登记同样数量的参数，部分参数在基线中不参与计算。
- 两视角共享潜在支撑的合成设定有利于形状匹配；协方差没有从真实 RGB 中预测。它用于验证信息是否可能有用，不能证明可学习性或实际性能。
- `expected_*` 是直接三维期望 RoPE 消融；`reprojection` 是重投影距离核。二者都不应标记为正式 WorldRoPE 复现。`*_uncertainty_oracle*` 使用已知注入误差协方差，只是诊断参照。
- `center_projected_logcov` / `rope3d_axis_projected_logcov` 比较共同查询视角中的投影协方差，不包含完整的局部单应 Attention。
- 特征分解描述子在本轮预计算，未通过协方差预测器反向传播。端到端接入前需要修复重复特征值处的梯度实现；`covariance_moments` 提供平滑对照。
