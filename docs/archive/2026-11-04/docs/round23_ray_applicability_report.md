# R23：真实输入适用性与成本失败筛查报告

2026-10-31。**当前冻结实现未通过 R23：72 条输入全部通过独立输入审计，72 条前向均未完成概率输出，完整生产阶段耗时 600.337994 秒，超过预定 600 秒。**失败记录与交付完整性已经 root 验收；这表示失败证据可信，不表示算子通过。没有执行一般输入的独立概率参考，没有读取效果标签，也没有科学效果通过结论。[验收记录](/home/liwt/GRoPE/results/round23_ray_root_acceptance_20261031.json)、[独立审计汇总](/home/liwt/GRoPE/results/round23_ray_applicability_independent_audit/summary.json)。

本报告仅解析已封存 JSON/metadata，并按字节重核其绑定文件；未解析原始真实输入档案，未重放模型。72 条逐项状态和原始失败文本见[完整 CSV](/home/liwt/GRoPE/docs/round23_ray_applicability_report_data/all_72_statuses.csv)，场景与尺度汇总见[分布 CSV](/home/liwt/GRoPE/docs/round23_ray_applicability_report_data/scene_query_statuses.csv)。两张表只整理既有记录，不增加实验或门槛。

R23 检验的是 R22 原实现能否在固定成本内处理少量合法真实输入。人口在读取数组前固定：旧 TRAIN 的 c3、c5、c6 各自首对 `[30,45]`，每对分别取 q8、q16、q24 的 8 个固定序号，共 **72 个不同 query、3 个 pair**。它不是随机总体样本，也不是全部 TRAIN。每个 query 使用全部 432 个原生 g8 key，没有按前方/FOV、质量、标签、旧成功情况或运行异常筛除查询。[冻结协议快照](/home/liwt/GRoPE/results/round23_ray_applicability/source_snapshot/ROUND23_RAY_APPLICABILITY_PROTOCOL.md)、[固定人口](/home/liwt/GRoPE/results/round23_ray_applicability/query_specs.json)。

| Query 尺度 | 每个场景使用的零起始原生序号 | 三场景合计 |
| --- | --- | ---: |
| q8 | 0、61、123、184、246、307、369、431 | 24 |
| q16 | 0、15、30、45、61、76、91、107 | 24 |
| q24 | 0、6、13、20、26、33、40、47 | 24 |

原始 R10 档案仅允许读取六个数组字段：`q_points3`、`q_pixel_cov3`、`q_K`、`k_K`、`q_to_k_transform`、`q_predicted_depth`。来源是已有 RGB 预测深度和给定相机，没有传感器监督、R13 targets、RGB/features 或旧概率进入本轮前向。整档字节哈希用于来源完整性，不代表读取其中其他数组值。

输入由原生 query 内全部像素组成。源相机点为 xᵢ，均值为 m，合法像素内因子为 `Wᵢ=zᵢ inv(Kq)[:,:2]/√12`。将中心化点与 Wᵢ 按 `1/√n` 拼成 `F[3,3n]`，对 Fᵀ 做 reduced QR，取 `Lsource=RQRᵀ`；目标相机输入为 `μ=Rcamera m+t`、`L=Rcamera Lsource`。独立审计不复用 QR，而是直接求原始像素内 covariance 加像素间 covariance，再核对 `LLᵀ` 及刚体变换。没有 jitter、特征值裁剪、秩阈值或中心修正。

这里必须区分两种深度处理：源预测深度沿用旧档案的 `NaN→10、+∞→20、−∞→0.1` 及 `[0.1,20]` 清理；变换到目标相机后，**不使用旧投影中的 Z≥0.1 m 截断**，而是严格 `Z>0` 的原投影测度。给定相机矩阵原样使用，正交性和行列式残差只记录，不修补、不另设选择门槛。

图像保持 144×192，参考坐标为像素坐标除以 8：`H=Kk[:2,:2]/8`、`c=Kk[:2,2]/8`，cell 边界为 `(整数像素边界−0.5)/8`。432 个 g8 cell 各面积 1，整个 FOV 面积 432；冻结面积 floor 密度为 `10⁻³⁰`，对应总 floor `4.32×10⁻²⁸`。沿用 `h=.25`（屏幕参考单位，即 2 像素）、T=0。这是冻结算子定义，不能误写成本轮已经生成了面积先验概率。

72 条输入审计全部通过：归一化误差 `max |actual−reference|/(1+|reference|)` 的全局最大值为 **2.054591693230416×10⁻¹⁵**，低于事先规定的 `10⁻¹⁰`。源点独立重建最大误差为 `7.805868340394377×10⁻¹⁶`，像素内 covariance 重建误差为 0；H、offset、bounds、身份与哈希均按合同核对。全部输入先封存，独立输入审计完成后才启动前向。[输入审计](/home/liwt/GRoPE/results/round23_ray_applicability_independent_audit/input_verification.json)、[72 条输入核对](/home/liwt/GRoPE/results/round23_ray_applicability_independent_audit/input_audit/checks.json)。这些结果支持输入构造正确，不验证透视概率积分精度，也不证明区域的单 Gaussian 近似有效。

所有前向失败均保留原样。表中“下溢”是 `inner marginal scale underflow`，“内部预算”是 `inner_evaluations budget exceeded`，“共享截止”是 `production phase wall budget exceeded`；各行输入通过数均为 8，完成数均为 0。

| 场景 | Query | 固定数 | 下溢 | 内部预算 | 共享截止 | 完成 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| c3 | q8 | 8 | 6 | 2 | 0 | 0 |
| c3 | q16 | 8 | 4 | 4 | 0 | 0 |
| c3 | q24 | 8 | 3 | 5 | 0 | 0 |
| c5 | q8 | 8 | 5 | 3 | 0 | 0 |
| c5 | q16 | 8 | 3 | 5 | 0 | 0 |
| c5 | q24 | 8 | 1 | 6 | 1 | 0 |
| c6 | q8 | 8 | 0 | 0 | 8 | 0 |
| c6 | q16 | 8 | 0 | 0 | 8 | 0 |
| c6 | q24 | 8 | 0 | 0 | 8 | 0 |
| **合计** | | **72** | **22** | **25** | **25** | **0** |

22 条下溢发生在内部一维边缘积分的缩放密度计算；冻结实现将无法表示的尺度作为失败，没有把它替换成零质量。它不等于整条 query 的真实 FOV 质量为零，也不能由此判断输入 covariance 非法。25 条内部预算失败各自达到 **4,000,000 次 inner evaluation** 上限，仍未完成输出；没有重试、提高节点数、换内核或降低精度要求。[完整封存记录](/home/liwt/GRoPE/results/round23_ray_applicability/forward/records.json)。

共享截止有明确的顺序效应：前 47 条是 22 条下溢和 25 条内部预算失败。第 **48** 条 `c5/q24/query 47` 在已做 85 次 depth evaluation、845,434 次 inner evaluation 后被共享截止中断；其调用连同输入加载耗时 4.656863 秒。之后 c6 的 24 条在截止已到的情况下短路，所有工作计数为 0，仍逐项保留状态。因此不能把这 24 条归因成各自几何的数值失败，更不能据表宣称 c6 比其他场景难。

| 成本或工作量 | 封存结果 | 含义 |
| --- | ---: | --- |
| 全部 72 条生产阶段 | **600.337994 秒** | 包含输入装配和逐条输出记录；超过 600 秒，门槛失败 |
| 各行 call + input load 求和 | 600.209323 秒 | 未包含完整逐条记录开销，不能替代阶段计时 |
| 输入准备阶段 | 0.397190 秒 | 前向之前单独计时 |
| 最终生产封存前的阶段后整理 | 0.148998 秒 | 另报；没有从已超限的生产阶段中扣除 |
| 下溢组调用与加载时间合计 | 61.391064 秒 | 22 条失败，单条约 0.011980–15.724609 秒 |
| 内部预算组调用与加载时间合计 | 533.876180 秒 | 25 条失败，单条约 18.448054–22.508265 秒 |
| 共享截止组调用与加载时间合计 | 4.942080 秒 | 含第 48 条中途停止及后 24 条短路 |
| depth evaluations 总数 | 9,769 | 工作计数，不是独立 query 数 |
| inner evaluations 总数 | 112,346,714 | 已做工作仍未产生完整结果 |
| rectangle calls 总数 | 4,229,977 | 包括未完成调用；不是独立验证样本 |
| 峰值 RSS | 94,748 KiB（92.527344 MiB） | 整个生产进程累计高水位，非单条增量 |

这些计时不提供成功 query 的完成延迟；不能把失败均值外推成“每行约 17 秒”，也不能估计全 TRAIN 的有效吞吐。所有 `outer_intervals=0` 表示外层积分没有正常返回该统计，不能解释为没有执行外层积分。已有 R22 合成例的成本与通过结论没有覆盖这次一般均值、真实协方差、432-key 条件。

72 条前向记录的 `array_sha256` 全部为空，没有成功概率或可用 partial 数组。空输出档案和失败原因都在完整库存内；它们不是 72 个零概率结果，也不是 72 个面积 prior 结果。独立最终审计确认全部失败记录合法保留，输出算术/metadata 违规数为 0；在没有成功数组的前提下，这不能表述为概率归一化、raw FOV 可加性、tail 或一般数值精度通过。[最终逐项审计](/home/liwt/GRoPE/results/round23_ray_applicability_independent_audit/final_audit/checks.json)。

联合预检有 **52 个纯 toy 测试通过**，联合来源闭包为 **18 个文件**，运行环境为固定 Python 3.11.7、NumPy 1.26.4、SciPy 1.13.1，OMP/MKL/OpenBLAS 均为 4 线程。预检还用禁止 `np.load` 和概率积分的替身核验 metadata 冻结路径。真实管线六阶段进程返回码均为 0，表示冻结、封存和审计流程完整执行；模型失败按协议作为数据保存，因此不能把进程退出成功当作适用性通过。[联合预检](/home/liwt/GRoPE/results/round23_joint_preflight_20261031.json)、[执行记录](/home/liwt/GRoPE/results/round23_ray_applicability_20261031_execution.json)。

root 验收检查了 18 项 current/source snapshot 绑定、严格冻结及阶段时间序和全部库存。本报告另行重核 receipt 中 15 个绑定文件的 SHA，以及生产 **174** 个文件、审计 **31** 个文件的完整库存：分别是 completion 内 173/30 个文件哈希，加各自 completion 本身。全部一致。没有删除失败、补做替代查询或修改冻结来源。

另有一次**执行结束后的 root 验收控制器修正**：旧控制器读取过时草稿字段 `all_calls_completed`，触发 KeyError；v2 改为冻结合同字段 `applicability_calls_completed`。修正仅涉及验收 metadata 控制器，原错误、前后控制器哈希及说明已保留；不是实验管线失败、数值修复或前向重跑。[修正记录](/home/liwt/GRoPE/results/round23_root_acceptance_controller_repair_20261031.json)。

本轮支持的决定是停止把**当前冻结实现**推进效果实验。它没有建立一般概率积分精度，也没有检验 KL、PCK、V、GS、VAL 或 fresh 泛化；现有数值失败不直接否定投影测度的数学定义，却足以否定当前实现按既定预算进入效果验证的资格。R22 的固定合成通过、R20 的固定候选筛查失败和 R21 的诊断边界均不改变。任何下一步实现修订或独立参考都须另行说明、冻结和验证；本报告不自动晋升候选，不选择新的研究路线，也不开放标签。

交付文件哈希如下；CSV 保留逐项原始文本、顺序、工作计数、时间、输入审计和档案哈希，可直接定位每一条失败。

| 文件 | SHA256 |
| --- | --- |
| root acceptance receipt | `faf571ce98182e45635059b3558bde6d28465b8bdf25e63f05374da575f023f7` |
| all_72_statuses.csv | `9fec2668e67539dedde8a5918ecd49665b3309919f413bf643781e1d2e815e67` |
| scene_query_statuses.csv | `335a03833bebcf472188d274b9abe27cf53ff754446e4eb506c68add1eb4a4cd` |
