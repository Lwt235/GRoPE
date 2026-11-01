# R20 正式协议静态设计审查

2026-10-31。已审查 [正式协议](/home/liwt/GRoPE/ROUND20_ANCHOR_FIXED_PROTOCOL.md)。
**设计 PASS，无阻塞设计问题；支持先固定配置否证，不直接做新 12-grid。**
这是静态设计结论，不是实现/数值验收或执行许可。
只读正式协议、旧 R17 协议、源码及选择元数据，没有运行新 forward 或读取 R19 新数组。
R19 的通过与数值来自 root 已验收后的通知，并非本稿重新审计。

R19 报告 q24 的 nearest-a 比 nearest-mu 高 2.403758pp，三个场景均正；
q8/q16 为 +0.513677/+0.625717pp，nearest-a 也高于固定 R17 主选 q24
1.474554pp。这使“中心身份值得一次 Gaussian 构造否证”比先前有了更直接的
依据，因此支持先完成当前机制，而非立即转入新 V 表示分支。它仍不证明 Ca、
内容耦合、完整区域质量或真实下游通过。

## 选择一个固定点，明确它只能回答什么

协议唯一主选固定 `h=.25,T=10, mean=a, Ca=C+(mu-a)(mu-a)^T`，仍用
`tau=T*h²/(h²+tr(Ca))`，保留旧 native Gaussian 积分、一次面积 floor、内容
池化、格与数值顺序。a 与 mu 先统一为 reference-8；Ca 不能误加像素单位的
偏移，也不能重复加入已有 pixel footprint。a 沿用已冻结的合法采样/投影
约定；mixed query 必须按自己的原 bounds 生成，不借标签或附近更好的中心。
协议已明确 a/Ca 保留 float64，旧 float32 mu/C 先提升 double；tau32 乘
content32 后提升 double 加 geometry。旧 C=0 只在 a=mu 时使 Ca=0 并退化
point；C=0、a不等于mu时仍有秩一支撑，合成测试须覆盖两者。

这个点继承旧 R17 的固定主配置，节约在无望构造上的校准成本；它**不是**
新模型的最优配置。Ca 同时增宽几何并降低 tau；若无匹配控制，只能说整个
参数构造发生了变化，不能分别归因于均值、方向、宽度或内容。协议已经增加一个
同 a、旧 C 几何、但使用主选同一 tau(Ca) 的固定归因控制；只作归因，不在
主选失败后晋升。它直接检查秩一增宽是否有用，不引入新搜索家族。

固定阶段失败即停止**本次固定构造**，保留失败；不在看效果后原地启动
12-grid 搜救。只有预先声明的固定阶段完整通过，才考虑另冻同信息控制校准；
更宽 candidate grid 既不由此授权，也不被要求。
这不排除所有带中心的 Gaussian，也不让一次固定点等同于完成校准或 VAL。

## same-a 对照不能只配一个可能偏弱的 T10

所有新 a 对照共享同一个 a、合法输入、query/key 内容、格、floor 与人口；
point 只去掉支撑 C，fixed 保留旧按面积定义的固定范围，不得暗借主选 Ca。
不要仅比较同 h/T 的尖 point 与被 trace 压低内容的主选，然后宣布形状有效。
从旧锁已选配置**事前**搬用以下 a 版本即可，不必第一步搜索新网格：

- 区域 Gaussian/quartic point：旧区域选择 `h=1,T=5`；fixed：`h=.5,T=5`，全 14 条件。
- 中心 Gaussian / quartic point：各自旧定位选择 `h=.25,T=5`，以及 legacy
  Gaussian point `h=.5,T=5`，覆盖 q8/q16/q24→g8。它们不能用 nearest-a 代替。

这些是同信息、旧固定配置的新 forward，不能复用旧 mu 版本的概率或称为新
校准最强 a 控制。协议十二完整算子还包括 matched point/fixed、constant-T
和 matched isotropic；quartic保留旧GL24，fixed精确沿用area/(64*12)*I，
point/fixed无需准备或借用full统计。若以后进入新校准，a-point/fixed 及定位控制也须获得同等
校准预算、固定目标与 tie-break。本轮不选择新主选；只有将来另有候选校准
协议时，才适用先满足原中心保护、再按原 J 选择一次的约束，当前协议没有
授权候选网格，固定主选保持不变。
旧 mu 的 strong/legacy point、fixed、full1/full4、quad4/16、旧主选等原参照
继续完整保留；可复用的只是完全同身份且已接受的旧结果，不能因新中心更好
删除原强控制，或把同 alias 重复计为独立证据。

## 全人口、数值与推进规则必须在效果前落成机器合同

固定旧 TRAIN 48 对，全部原九 common 条件和五 native scale 条件，沿旧
common 投影、mass 加权 KL 与等场景/等条件口径；原
`J=.5*common9+.5*direct5` 不改。中心仍为 q8/q16/q24→g8 原合法中心人口、
native argmax、first-index ties、整数计数及精确 Fraction，损失边界恰为
`1/200`。全部行保存，front/FOV/区域有效性不能缩小中心人口。

正式协议已将此前“不恶化才继续”的歧义消除：原八项 TRAIN 宏规则和九项
中心保护不减；增加全14条件各对旧strong/legacy quad的2%保护；common9
和direct5各对same-a Gaussian point/quartic point/fixed改善至少2%；再加
same-a三localizer的九项.5pp中心保护。所有条件合取，Fraction无epsilon。
这明确是现在新增的保守TRAIN资源筛选，没有伪称旧预注册，也没有另加
未声明的逐场景全正或旧主选逐项不退步规则。
原 TRAIN 宏门槛不能悄悄替换原 VAL 的逐条件门槛；全 14 条件报告也不等于
这次已经做了六扰动或全部 VAL 门槛。

新均值/Ca 不继承旧数值通过资格。冻结完整来源、独立 auditor 与参考人口，
先检查 Ca 恒等式、PSD、单位、a=mu 退化、同一 tau 的 matched 控制、极小
FOV/外部 a、24px child 求和及一次 floor。旧数值误差/参考收敛/概率有效性
标准不放宽；来源错配、参考不收敛与科学失败分开。新 a 的合法来源和所有
新 forward 要全局封存并先通过独立前向审计，再读取实际标签；重用 TRAIN
不恢复统计盲法。来源及掩码身份不能以 allclose 替代。

## 后续仍然受原限制

如果固定阶段通过，只允许考虑另冻同信息控制校准与后续旧 VAL：保留全部 14 区域、
原中心、六扰动、强对照、原 accuracy/cost 路径及完整成本，不因为 fixed
TRAIN 良好而跳过。若只改善 a-point 的定位、Ca 无区域优势或损伤原保护，
应如实停止；不增加中心插值、协方差系数或新带宽来救本次候选。

R8 的 oracle-route 配原 V 仅比 singleview 改善 1.031%、R9 真 GS 形状读取
仅改善 0.1460%，仍使真实下游收益链不确定；这些不是理论上限。当前最小
阶段是合理的机制投入，不是对值瓶颈的解决，也不启动大规模训练。Ca 作为
patch 的中心语义修正不构成数学新颖性；原 actual-V、真实 optimized-GS 与
最后未触碰确认的门槛全部保留。

协议还明确全量保存8064个矩阵，独立forward/numeric通过后才实际读标签；
不授权事后切换存储合同。本人已承担新模型及合成测试实现，34项纯合成
测试通过（NumPy先导入、Torch intra/inter-op及三线程环境变量均为4）；
这不是实数据数值通过。本人不再担任该模型的独立数值审计者；联合冻结与
真实数值状态另行记录。
