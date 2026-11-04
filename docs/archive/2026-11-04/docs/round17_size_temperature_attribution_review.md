# R17：已知 query 尺寸是否足以解释温度抑制？

外置归因审查，尚未采用、尚未执行。只读既有 TRAIN 输入描述与源码，没有
读取当前 VAL 效果、生成新概率或修改冻结文件。这里不增加本轮门槛、不选择
新 primary，也不改变本轮完整执行和判定；任何后续比较须另冻归因协议。

## 现有对照没有回答这个问题

primary 的几何和温度都使用观测投影区域的完整协方差 C_obs。
matched isotropic 保留原 trace/tau，只改变几何方向性；constant-T full
保留原几何，只移除温度抑制。两者均不能区分 observed trace 的逐 query
信息与仅由已知 token 尺寸产生的缩放规律。

[既有 TRAIN 描述](../results/round17_train_temperature_description/summary.json)
仅含三个固定首 pair ×四 query 分区 ×八固定 query，共96个去重样本，每个
分区24个；不是完整 TRAIN 或 VAL 统计。h=.25、T=10 下：

| query | 观测 tau 中位数 | 理想原生方形的 tau |
|---|---:|---:|
| g8 | 2.697547 | 30/11 ≈ 2.727273 |
| g16 | 0.844760 | 6/7 ≈ 0.857143 |
| g24 | 0.395670 | 0.4 |

中位数接近支持检查这一解释，但不是效果等价证据。同一小样本中 g24 的
观测 tau 范围仍为0.210325–0.464259，不能用中位数替代逐 query 差异；
即使温度变化小，原生概率的近平局也可能改变 argmax。没有标签或新预测就
不能推断哪个温度对区域 KL、中心 PCK 或实际 V 更有用。

## 最小对照：只替换温度的支撑来源

令 query 整数半开 bounds 给出宽 w、高 b，单位仍为原192×144图像像素。
连续像素支撑（包含像素 footprint，而非只有像素中心）的原生协方差为：

```text
C_native = diag(w², b²) / (64*12)
trace_native = (w²+b²) / (64*12)
tau_size = T*h² / (h²+trace_native)
```

只把 `tau_obs` 换为 `tau_size`。完整 Gaussian 几何仍为
`N(mu_obs, C_obs+h²I)` 的同一 native-key 积分；原 mean、内容、已选 h/T、
floor、softmax、输出及评价人口全部不变。它没有逐场景/逐尺度可调系数。
这是 known-native-size 温度对照，不是把预测 C 换成 fixed 几何。

必须按 **bounds trace** 定义：`2*area/(64*12)` 只对方形相等。mixed 含
16×8和8×16矩形，其 bounds trace 为5/12，area-based值为1/3；h=.25/T=10
时对应 tau 为30/23≈1.304348与30/19≈1.578947。混用两者会引入额外差异。
**原有 area-based fixed 基线保持原定义和名称**；此新对照既不修补旧实验，
也不能被重命名为旧 fixed。原有比较及负结果继续保留。

| 角色 | Gaussian 几何 C | 内容系数 |
|---|---|---|
| 固定 primary | C_obs | tau_obs |
| 已有 matched isotropic | trace(C_obs)/2 I | 同一 tau_obs |
| 已有 matched constant-T | C_obs | T |
| 新归因对照（若另行冻结） | C_obs | tau_size |

## 能回答什么，不能回答什么

它检验：在固定几何、内容及 h/T 下，observed trace 相对已知原生大小是否
提供额外效用。observed C 在目标图像坐标系中，可包含视角/深度引起的缩放、
非线性投影和区域内散布；native 尺寸在原 query 图像中。二者接近不是恒等，
也不能据此把 C 解释为置信度。若新对照接近或更好，只削弱“温度必须依赖
观测 trace”的论据；full C 几何的作用并未被这个对照移除。

tau_size 也不继承原 trace 温度的全部数学性质。保持 C_obs 几何时，连续
线性内容的位移因子含
`T*h²*(h²+lambda_max(C_obs))/(h²+trace_native)`；它不保证 ≤T*h²。
人为将几何 C_obs 置零而保留 native 尺寸时，tau_size 亦不回退 T。
这些是归因对照的明确差异，不足以预判其经验表现，也不豁免任何旧门槛。

## 后续若执行：最少纪律

- 另冻协议、源、dtype/算术次序、数据人口和分析，再产生新概率。优先使用
  完整预声明旧 TRAIN 人口及全部既定分区，避免只挑上述96个方便样本。
  同 primary 已选 h/T 做配对归因，不扫描 size 温度、不替换本轮 primary。
- 可以复用完全相同的已封存 raw geometry/content；若未保存，则按冻结
  operator 完整重构并用已有 primary 概率/hash绑定。只保留配置名称或
  数学相同公式不够。新的 tau_size/probability 仍须独立数值与完整指标审计。
- 保留各场景/分区共同人口、KL/JS、精确 PCK、原生 argmax 变化，以及逐 query
  trace/tau 差异；按 native 尺寸分层比 pooled 中位数更能区分尺度效应。
  所有统计均为归因，不能把本轮门槛事后扩成又一个过关条件。
- 成本包含它自身从 bounds 计算 native trace/tau 的路径，以及同一完整几何
  准备、积分、内容与读出；不能免费借用 primary 的 tau。按实际实现冻结
  计算/缓存边界，基线同场配对计时，不能以历史耗时推断新对照优势。
- 若 observed 优于 size，结论限于固定 h/T 的匹配对照，尚不证明胜过充分
  校准的 size 方法；若差异不明显，不能叫严格等价。任何等价性/非劣性主张
  都需要在新结果前声明合理界限和统计设计，而不是看结果后选容忍范围。

目前最稳妥的归因表述是：R17 使用观测支撑 trace 调节内容，既有对照检验
方向性与抑制的作用；**观测 trace 超越已知 query 尺寸的额外价值尚未证明**。
