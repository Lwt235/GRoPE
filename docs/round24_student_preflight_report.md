# R24：固定 Student-t4 合成数值预检报告

2026-10-31。**R24 的固定合成数值预检通过**：42 次生产调用、72 次独立参考调用和 6 项恒等检查全部通过预先冻结的门槛。生产与最高阶独立参考的最大概率绝对误差为 **1.03517×10⁻¹²**，生产阶段耗时 **0.327999 秒**。这支持当前实现对这组固定合成输入的数值可用性；没有检验真实任务效果、一般参数域精度或最终研究目标。[root 验收](/home/liwt/GRoPE/results/round24_student_root_acceptance_20261031.json)、[独立审计汇总](/home/liwt/GRoPE/results/round24_student_independent_audit/summary.json)。

本报告在解析结果前重核 receipt 的 20 个绑定文件及生产 59 个、审计 56 个文件的完整库存，全部一致。仅使用封存 JSON/metadata 和文件字节哈希，未解码 NPZ、读取真实输入或标签、执行模型/参考重放。完整逐项数据见[36 个基础案例](/home/liwt/GRoPE/docs/round24_student_preflight_report_data/base_36_cases.csv)、[72 次参考调用](/home/liwt/GRoPE/docs/round24_student_preflight_report_data/reference_72_calls.csv)、[6 项恒等检查](/home/liwt/GRoPE/docs/round24_student_preflight_report_data/identity_6_checks.csv)。

本轮定义一个中心 a 和协方差 `S=C+h²I`，固定 `ν=4、h=.25`，Student 的 **scale matrix 为 S/2**。实际积分密度与最终概率为

\[
p(y)=\frac{(1+q/2)^{-3}}{\pi\sqrt{\det S}},\quad
q=(y-a)^TS^{-1}(y-a),\quad
P_j=\operatorname{softmax}_j\{\log[M(B_j)+10^{-30}|B_j|]+b_j\}.
\]

这里的全平面 covariance 是 S，不能把 S 直接当 Student scale；密度归一化常数也不能在固定绝对 floor 前省略。生产代码先积分 native cell，再加一次面积 floor，最后加内容偏置并 softmax。C 的微小负特征值只按冻结准入容差记录，不裁剪或修补；有限输入 S 必须通过严格 Cholesky。

固定人口是 **18 种几何 × 2 种偏置 = 36 个基础案例**，另有 6 个生产恒等案例。几何涵盖零/rank-one C、三个 patch 尺度、旋转和负相关、薄长/宽分布、FOV 边界/外部中心、低于 floor 的远中心、anchor 二阶矩，以及两个各含 432 个原生 cell 的合成几何；其余基础几何各含 16 个 cell。[完整冻结描述](/home/liwt/GRoPE/results/round24_student_preflight/fixtures.json)。36 个案例不是 36 种独立几何，72 次参考是同一 36 个案例的两个积分阶数，不是 72 个独立任务。

文件名中的 T0 表示 b=0；T10 表示固定名义温度 10，并使用

`b(k)=τ sin(.71 kx+.23) cos(.37 ky−.19)`，`τ=10h²/(h²+tr C)`。

所以实际 τ 随 C 变化，不总是 10。这只是确定性的 float64 native 偏置，用来检查积分、floor 与 softmax 顺序；没有在线 learned Q/K、训练或实际 V 验证。每种几何两次计算的 raw 质量相同，不能把内容重复当作额外几何覆盖。

生产方法是物理边长不超过 `4h` 的规则子矩形上做二维 tensor GL24，并单独积分包围 FOV。独立参考不调用生产密度、tiler 或积分器：先用 Student-t4 的 x 边缘密度，再用条件 t5 CDF 差积分 y，最后沿 x 采用物理 panel 不超过 `4h` 的 GL64/96。其条件 scale² 为 `D(2+u²)/5`，其中 `D=Syy−Syx²/Sxx、u=(x−ax)/√Sxx`；共享依赖是已披露的 NumPy/SciPy。两阶收敛是本组输入的数值证据，不是统一误差证明。[冻结协议快照](/home/liwt/GRoPE/results/round24_student_preflight/source_snapshot/ROUND24_STUDENT_PREFLIGHT_PROTOCOL.md)。

所有比较均使用封存的生产 P。raw FOV 相对误差以 native raw mass 之和比较，raw L1 以参考 FOV 质量为分母；这些检查没有被 floor 后的概率误差替代。

| 检查 | 全部案例最大值 | 冻结上限 | 达到最大值的案例 |
| --- | ---: | ---: | --- |
| 生产对 GL96：P 最大绝对误差 | 1.0351719482×10⁻¹² | 10⁻⁴ | point_center_t10 |
| 生产对 GL96：raw FOV 相对误差 | 3.9059592726×10⁻¹² | 10⁻⁶ | point_center_t0 / t10 |
| 生产对 GL96：raw L1 / 参考 FOV | 1.7335009460×10⁻¹¹ | 10⁻⁶ | tiny_far_t0 / t10 |
| GL64 对 GL96：P 最大绝对误差 | 1.1102230246×10⁻¹⁵ | 10⁻⁸ | point_center_t10 |
| GL64 对 GL96：raw FOV 相对误差 | 5.2685219940×10⁻¹³ | 10⁻⁸ | tiny_far_t0 / t10 |
| GL64 对 GL96：raw L1 / 参考 FOV | 6.4968503017×10⁻¹² | 10⁻⁸ | tiny_far_t0 / t10 |
| 42 次生产的概率归一化误差 | 2.2204460493×10⁻¹⁶ | 2×10⁻¹² | nominal8_t10 等 |
| 72 次参考的概率归一化误差 | 2.6645352591×10⁻¹⁵ | 2×10⁻¹² | tiny_far_t10，GL64/96 |
| 生产独立重组 floor/bias 的 P 误差 | 2.2204460493×10⁻¹⁶ | 2×10⁻¹² | point_center_t10 |
| 生产 native raw 和对单独 FOV 积分 | 7.7715611724×10⁻¹⁶ | 10⁻⁸ | native432_t0 / t10 |
| 参考 native raw 和对单独 FOV 积分 | 1.1102230246×10⁻¹⁶ | 10⁻⁸ | point_center_t0，GL64 等 |

tiny_far 的生产 raw FOV 为 **4.203556218073709×10⁻³⁵**，GL96 为 `4.203556218070057×10⁻³⁵`；总 floor 是 `1.6×10⁻²⁹`，raw/floor 约 **2.627223×10⁻⁶**。两种偏置下 raw FOV 相对误差均为 `8.6880382001×10⁻¹³`，raw L1/FOV 为 `1.7335009460×10⁻¹¹`。正 raw 质量被保留，没有提前置零；全部 42 次生产均无零 raw cell、无整行零质量，也未使用精确 area-prior 标记。floor 在此占主导，但 T10 仍会调制 floor；P 很准确本身不能替代这里确实执行的 raw 相对检查。

36 个基础案例中，生产对 GL96、GL64 对 GL96 的 argmax 变化数均为 0，**不等于没有并列**。T0 的 outside、boundary 有精确最大值索引 `[7,11]`，native432 为 `[203,228]`，corner_point 为 `[5,6,9,10]`。tiny_far_t0 的精确最大值索引为 3，但按预定 `10⁻¹²` 绝对差定义，近似最大值集合为 `[2,3,7]`。这些索引均为零起始原生 cell 顺序；它们是数值诊断，不是定位正确率。

6 项恒等检查只比较生产结果与其生产 parent。前五项 parent 为 rotated_t0，最后一项为 point_center_t0；全部使用零内容，门槛均为 raw/P 绝对误差不超过 `10⁻⁸`。

| 恒等检查 | raw 最大绝对误差 | P 最大绝对误差 |
| --- | ---: | ---: |
| 平移 | 2.775558×10⁻¹⁷ | 5.551115×10⁻¹⁷ |
| 屏幕单位 ×3.7，C/h/floor 密度同步换算 | 5.551115×10⁻¹⁷ | 5.551115×10⁻¹⁷ |
| key 逆序排列 | 0 | 0 |
| 非等面积 cell 合并 | 5.551115×10⁻¹⁷ | 1.110223×10⁻¹⁶ |
| 坐标轴交换 | 2.775558×10⁻¹⁷ | 5.551115×10⁻¹⁷ |
| 零 C 的完全相同输入重复 | 0 | 0 |

最后一项仅检查重复输入一致性，不是独立参考或有限 covariance 趋零序列；也不能据零内容合并成立推断任意内容偏置下的合并恒等式。

| 成本 | 封存值 | 预算或计时范围 |
| --- | ---: | --- |
| 全部 42 次生产阶段 | 0.3279985674 秒 | 120 秒；含逐条输出记录 |
| 生产阶段后整理 | 0.0613348912 秒 | 单列；总计至 completion 写入前 0.3893337697 秒 |
| 独立参考/审计阶段 | 1.8009582385 秒 | 120 秒；按审计记录的整阶段口径 |
| 参考阶段后整理 | 0.0028744247 秒 | 单列；总计至 completion 写入前 1.8038329724 秒 |
| 生产 density cell-node 总数 | 2,691,072 | 42 次合计，包含各自直接 FOV 积分 |
| 生产单次最多 cell-node | 497,664 | 每次上限 4,000,000 |
| 独立参考 x cell-node 总数 | 394,240 | 72 次合计上限 2,000,000，包含直接 FOV |
| 生产峰值 RSS | 87,376 KiB（85.328125 MiB） | 进程累计高水位，不是每个案例独占内存 |

四个 432-key 基础调用各耗时约 0.041645–0.043410 秒，其余输入大多只有 16 keys。这里的成功延迟只适用于这些合成调用；不能直接外推全训练、梯度或模型成本，也不能与 R23 不同分布和人口的失败计时组成公平加速比。没有积分域截断被引入，因此没有可借用的 Gaussian 截断尾证书；仍不存在一般求积和舍入误差的严格上界。

该形状的解释限制保持不变。`C=0` 给出 **Student-t4 点核**，不是旧 Gaussian 点核；把 `h²I` 加到 covariance 不等于卷积 Gaussian blur。t4 有有限二阶矩但四阶矩发散，较高中心密度和较重尾部不保证区域改善。既有同协方差反例仍成立：S=I 时，在 `q=2` 处 t4/Gaussian 密度比为 `e/4<1`；平方半径在 `2−ε` 与 `2+ε` 间的均匀薄圆环 covariance 恰为 I，其 KL 到 t4 比到 Gaussian 更差，差值在 ε→0 时趋于 `log 4−1>0`。此反例否定普遍优势，不预测本仓库任务的实际效果。[既有路线审查](/home/liwt/GRoPE/docs/student_t4_route_comparison_unselected_draft.md)、[R15 备用机制记录](/home/liwt/GRoPE/docs/round15_student_t_reserve.md)。

正式联合来源为 **10 个文件**，冻结前两次联合 toy 测试均为 **41 项通过**。第一次成功的 metadata-only 预检之后，只改动参考代码的最终无效 raw-mass 异常分支，使 `ReferenceFailure` 携带已耗工作量和 partial masses；数值分布、节点、案例和门槛均未变化。第二次 v2 联合预检重新通过，其来源绑定用于正式冻结；正式参考 SHA 为 `8c15fada1c34735181ffc4314fb4816b01f2885cb1697b6feaeef5992edbb76e`。第一次预检保留，期间没有提前执行正式 fixture，也没有效果后重试。[开发历史](/home/liwt/GRoPE/results/round24_preflight_development_history_20261031.json)、[正式 v2 预检](/home/liwt/GRoPE/results/round24_joint_preflight_20261031_v2.json)。

root 验收确认 current source 与双方快照一致，且全部正式调用晚于联合冻结。生产 59 个文件是 58 个 completion 绑定条目加 completion 自身；审计对应为 55+1。本报告重核所有绑定与库存，并逐项复核表格行数、案例顺序和数值来源。R24 没有读取真实数组/标签，`scientific_gate_passed=false、goal_complete=false`；旧区域、中心、扰动、实际 V、优化 GS 与未触碰联合确认门槛全部保留。任何后续真实验证都需要独立协议，数值通过不自动晋升方法或证明用户目标已经完成。

| 交付文件 | SHA256 |
| --- | --- |
| root acceptance receipt | `9fd1a57fa68c09bba39f9a474d91f538c43de3e0197504a986e4cb5080104503` |
| base_36_cases.csv | `d2248a5c0669aa2c374c7eaecd7acd8789be1d67e67681233174693ff8b3e188` |
| reference_72_calls.csv | `29c4f5f824f99470c8c7b5639cb68788070674abb37c3fa646f7a2a76189965e` |
| identity_6_checks.csv | `e8b1db95fa7c9cbea51d330f9e0321d815312df1debcf414af79e78a7cb11ceb` |
