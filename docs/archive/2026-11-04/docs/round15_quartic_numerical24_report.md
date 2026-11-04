# Round15 固定 GL24 修复：合成数值与 CPU 成本

**固定 GL24 的合成数值复验通过，但速度优势明显小于原 GL16。** 本报告不宣称真实任务效果或旧门槛通过；效果标签仍未读取。真实 TRAIN 的完整576 geometry /1728 probability行GL24复验由独立预检任务另行记录，不能由本次合成通过替代。

## 为什么必须修复

原GL16虽通过有限合成域，却在真实 TRAIN 前向数值检查中失败：Harvard c6 第一pair、q8 index0、quartic_point、h=.25 的T0/5/10三行超过1e−4；实际最大概率误差约0.0030897，使用独立高精度支持的hybrid积分误差约0.0030937。参考GL64→96差低于6.7e−14，因而不是参考未收敛，也不是float32准备主导。其他1725行通过和argmax不变均未用于豁免。

旧 `results/round15_train_numerical` 明确保留为失败；原GL16模型、所有旧源和数值/成本快照未修改。当前修复仅在已有冻结API显式传 `quadrature_order=24`，仍按≤1参考单位子矩形积分并合回原key。密度形状、支持/内容、h/T范围、floor、数据和所有数值/科学阈值保持不变，没有按效果调阶数或新增模型模块。

## 同域独立合成复验

完全重跑120个原始矩形、240个拆分、12组完整FOV和4类梯度，使用相同冻结SciPy参考和同一门槛：

| 指标 | GL24结果 | 原门槛 |
|---|---:|---:|
| 最大原始质量绝对误差 | 9.99e−16 | 1e−8 |
| 最大拆分可加性误差 | 6.66e−16 | 1e−8 |
| 最大FOV归一化概率误差 | 1.318e−9 | 1e−4 |

全部通过；内部仍没有floor或额外attention keys。
报告：`results/round15_quartic_tiled_independent_order24_v1/verification.json`。
最终审计锁：`results/round15_quartic_tiled24_numerical_audit_frozen/lock.json`，SHA256 `60f9d8d16a8eb1530e6fa6dcd7af6cfd49851baeb083bded1c9a15487814b7f1`。

## CPU成本全部重新配对测量

新成本协议 `ROUND15_NUMERICAL24_PROTOCOL.md` 与入口 `experiments/round15_quartic_preflight24.py` 复用冻结旧输入生成器/计时循环，只扩展新协议与source provenance并强制order24。3项guard测试通过；旧源未编辑。新进程中重测所有角色，不把不同进程的GL16结果直接作为主要比率。

仍为纯合成q8/q16/q24→432个g8 keys、固定仿射、h=.5、float64、CPU4线程、1 warmup＋3次测量中位数，仅位置积分和节点求和：

| query尺度 | quartic GL24 | Gaussian full1 | Gaussian full4 | Gaussian quad16 | 原quartic GL48 |
|---|---:|---:|---:|---:|---:|
| 8 px | 1.463 s | 0.184 s | 0.781 s | 2.817 s | 3.838 s |
| 16 px | 0.246 s | 0.0315 s | 0.173 s | 0.652 s | 0.894 s |
| 24 px | 0.0945 s | 0.0123 s | 0.0568 s | 0.409 s | 0.314 s |

相对quad16的时间为0.519/0.377/0.231，即1.93/2.65/4.33倍加速；q8甚至未达到2倍。相对full4为1.87/1.42/1.66倍时间，相对full1为7.94/7.80/7.68倍时间。因此不能沿用旧GL16“快于full4”的结论。相对原GL48仍快2.62/3.64/3.33倍。

服务器上存在其他作业，三个重复的原始时间全部保留；这些是有限的CPU预检结果，不能当作GPU性能，也不是包括support准备、content/softmax和common aggregation的旧正式gate计时。当前没有GPU计时。

新结果：`results/round15_quartic_preflight24/cost_cpu.json`，SHA256 `2398ac0108868389319e5179dcea065c032dd972a8320a06616cb6f0a2423811`。
新source/input锁：`results/round15_quartic_preflight24/lock.json`，SHA256 `ec0952139cd8b102ee39615ea76e78d42218a3084e071b7f272505f0e6292332`。
最终独立数值锁通过 `numerical_audit_link.json` 与成本锁已绑定的同一report/completion显式相连。

精度修复是否可用于后续候选效果实验，仍需完整真实TRAIN数值复验和主任务审查；不得凭本报告继续读取标签或降低任何门槛。
