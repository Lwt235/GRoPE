# Round14：固定内容温度与尺度的 TRAIN-only 机制诊断

2026-10-30。本协议须经审阅并冻结后执行。动机是已审计R12中full4的24→8中心PCK损失，以及R13表明粗尺度区域模式与中心有较多但低KL成本的冲突。本轮只问：父区域内容T=5是否推动了中心偏移；同带宽下单高斯与四高斯的差别是否仍在。**不新增模块、不调整ρ、不扫描新参数、不选择赢家，也不把训练结果算作验证集门槛。**

## 固定样本与七个角色

48个旧R10训练pair，Harvard c3/c5/c6各16个；q8/q16/q24均对原生g8 key。所有query前向保留。输入只使用同一冻结RGB预测几何、特征与相机。角色固定如下；这是配对的机制对比，不是选择网格：

| role | 几何 | h（8px单位） | 内容T |
|---|---|---:|---:|
| full4_t0 | 原full4四个局部完整矩 | .5 | 0 |
| full4_t5 | 同上，完全同一geometry logits | .5 | 5 |
| full1_t0 | 单个完整global矩 | .5 | 0 |
| full1_t5 | 同上，完全同一geometry logits | .5 | 5 |
| point_pck | 冻结point_pck控制 | .5 | 5 |
| quad16 | 冻结quad16控制 | .5 | 5 |
| full1_frozen | 旧full1的锁定参考 | .25 | 0 |

最后一项仅增加一次单高斯积分，global矩共用，可分离旧full1的带宽与内容混淆。每条件仅计算五组不同的geometry logits，再产生七组概率；不引入新可选超参数。冻结R10/R11/R12模型、key矩形积分、feature pooling、background floor、单位与有限fallback全部原样复用，内容始终是相同query父区域与key内容余弦。T0/T5来自同一个geometry张量和相同content张量。

## 标签与“预测中心”的定义

复用已经独立审计的R13每pair目标档案：q8/q16/q24→g8 direct sensor计数、独立原生中心投影/valid/status。须绑定R13 lock、completion、通过的独立audit哈希，并核对完全相同的48pair身份。源冻结/forward阶段仅读取manifest与预期标签哈希，不解码这些标签数组。三个训练场景的全部forward完成并写入哈希后，任何evaluate才允许读取标签。

本轮偏移统计中的“预测中心”明确是**同一冻结point控制使用的全区域像素投影均值μ**，不是传感器中心，不是独立预测的单个原生中心像素。保存实际用于point的μ。所有角色相对同一个μ报告偏移，并额外报告μ到独立sensor中心的误差，避免把区域质心与中心像素误差混在一起。

## 统计与固定对比

每个角色、query尺度、scene/pair保存direct概率和原生argmax。direct KL继续真实可见像素质量加权；PCK@8使用R13独立sensor-valid中心。各尺度内先合并pair的原始分子/分母，再等权平均三个场景。三个尺度的等权宏平均仅作附加汇总，不能掩盖q24结果。不同query尺度的KL不解释为相同任务难度。

报告：direct KL、PCK@8；winner到μ的距离与概率期望key中心到μ的距离；μ周围8px内的概率质量。偏移与中心概率质量均报告全query及sensor-center-valid两种分母，全部query不会因为有效性或偏移过大被删除。μ到sensor中心距离只在center-valid群体定义。

三个预声明配对对比为：full4_t0−full4_t5，full1_t0−full1_t5，full1_frozen−full1_t0。报告各scene/尺度的KL相对变化、PCK百分点差、winner相对μ的偏移差；另报winner改变比例与位移，以及左正确右错误/右正确左错误的中心有效比例。最后两类仅事后解释，不能进入前向规则或mask。

相对point_pck报告full4 T5与T0的**训练集**PCK缺口，且仅当T5确有正缺口时报告训练缺口回补比例。不能把这称为补回R12验证集的11.20pp，也不预设关闭内容必然有益。“未明显损害区域KL”以实际相对变化透明展示，**本诊断不创建新的pass/fail门槛**。

## 阶段与产物

新runner `experiments/round14_temperature_probe.py`；新测试 `tests/test_round14_temperature_probe.py`。四个阶段：`freeze`、`forward --scene`、`evaluate --scene`、`finalize`，默认目录 `results/round14_temperature`。freeze之前必须完成源码与合成测试审阅；freeze仅锁定协议，不产生预测；forward验证冻结锁后才能运行。未冻结时禁止真实输入前向或标签评价。

freeze绑定源/协议/测试、48个训练input和manifest、R13目标及独立audit、原R12算子源锁、固定配置和分区。forward保存48pair×3query尺度×7role=1,008个float64概率矩阵及argmax、每尺度共享μ、720组几何数值诊断，不读sensor或标签。预计原始概率约0.68GB；不重复保存公共网格概率或dense几何。

evaluate必须先验证全部三个场景forward completion。随后逐pair解码R13目标，核对概率/argmax、共同群体与所有尺度相同visible mass，保存1,008条原始指标和432条预声明配对指标。finalize复核所有哈希、时间顺序、群体和场景等权汇总。无覆盖或静默续跑。

合成测试验证：固定五组geometry/七组角色、零内容不随feature变化、同矩同分量退化时同h/T概率一致、h参考确实不同、独立手算KL/PCK/偏移、配对人口、三个场景全部forward完成前的标签阶段保护。所有测试不访问真实目标或新增效果分数。

无论结果如何，R12失败、真实深度/GS/NVS未通过项与所有旧门槛保留。如结果支持内容温度机制，只允许随后另行预注册一个精炼候选并完整验证，不在本轮直接选择T0替换主方案。
