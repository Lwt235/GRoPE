# 完整评价与组合优化验证

旧seed11模型、旧验证集765对象，单张RTX A6000独占采样，CPU1、网络FP32/几何FP64、TF32关闭。
数据/权重身份核对、输入准备之外的模型读取和Python启动不计入下表；评价耗时包括缓存映射、gather/H2D、完整前向、输出回传、指标和预测写盘/fsync。首次总耗时另计私有模型适配和上卡。没有清空操作系统页缓存，不称冷磁盘性能。

## 完整评价（B4/N512，末批B1）

| 模型 | 实现 | 首次适配＋评价(s) | 暖评价中位(s) | 峰值分配(GiB) | OA / mCA |
| --- | --- | ---: | ---: | ---: | --- |
| relative5 | original | 4.481 | 4.258 | 0.803 | 86.1438% / 81.7059% |
| relative5 | geometry_factored | 2.023 | 1.931 | 0.803 | 86.1438% / 81.7059% |
| relative5 | compiled | 9.077 | 1.875 | 1.054 | 86.1438% / 81.7059% |
| distance_mlp | compiled | 8.734 | 1.461 | 1.054 | 82.8758% / 78.3441% |
| distance_mlp | geometry_factored | 1.625 | 1.620 | 0.787 | 82.8758% / 78.3441% |
| distance_mlp | original | 3.369 | 3.355 | 0.802 | 82.8758% / 78.3441% |

relative5：编译相对几何重排的暖评价配对节时 3.21%，两块结果 [4.539188516543325, 1.8777644444369024]；同实现重复绝对漂移中位 0.79%。预测改变 0/765，最大logit差 3.81e-06。首评额外开销约需 126.68385283564864 次暖评价摊销（描述性估计）。

distance_mlp：编译相对几何重排的暖评价配对节时 8.70%，两块结果 [9.690316350021211, 7.708078546274844]；同实现重复绝对漂移中位 0.53%。预测改变 0/765，最大logit差 4.77e-06。首评额外开销约需 44.55858091332567 次暖评价摊销（描述性估计）。

## 四臂完整训练步（B16/N512）

| 模型 | 实现 | 中位ms | 峰值GiB |
| --- | --- | ---: | ---: |
| relative5 | geometry_factored | 107.300 | 7.880 |
| relative5 | fused | 100.123 | 9.254 |
| relative5 | compiled | 103.756 | 6.360 |
| relative5 | combined | 96.465 | 9.156 |
| distance_mlp | geometry_factored | 103.673 | 7.816 |
| distance_mlp | fused | 96.352 | 9.191 |
| distance_mlp | compiled | 100.525 | 6.295 |
| distance_mlp | combined | 93.046 | 9.093 |

- relative5 geometry_factored->fused：配对节时 6.82%，8/8块正向，重复绝对漂移中位 0.70%。
- relative5 geometry_factored->compiled：配对节时 3.22%，8/8块正向，重复绝对漂移中位 0.70%。
- relative5 geometry_factored->combined：配对节时 10.16%，8/8块正向，重复绝对漂移中位 0.85%。
- relative5 fused->combined：配对节时 3.57%，8/8块正向，重复绝对漂移中位 0.71%。
- relative5 compiled->combined：配对节时 6.92%，8/8块正向，重复绝对漂移中位 0.61%。

- distance_mlp geometry_factored->fused：配对节时 7.28%，8/8块正向，重复绝对漂移中位 0.36%。
- distance_mlp geometry_factored->compiled：配对节时 2.99%，8/8块正向，重复绝对漂移中位 0.55%。
- distance_mlp geometry_factored->combined：配对节时 9.98%，8/8块正向，重复绝对漂移中位 0.32%。
- distance_mlp fused->combined：配对节时 3.47%，8/8块正向，重复绝对漂移中位 0.66%。
- distance_mlp compiled->combined：配对节时 7.39%，8/8块正向，重复绝对漂移中位 0.93%。

## 解释边界

核对 224 次资源快照，无其他同卡计算进程；其他卡仍有工作。完整评价各臂重复输出相同，所有指标从保存预测重算。训练每块同臂重复末步损失完全一致。
组合通过同权重前向、全参数梯度及一步后输出核对；AdamW参数更新例外在review.json保留。该检查不替代完整训练效果验证，也不推翻此前编译64步轨迹分叉。
本轮未训练新科学模型、未读取新确认集、未改变主任务。完整训练终点配对随下一项确需训练的科学任务安排。原模型无几何输入梯度的适用限制保留。
