# R20 固定 anchor Gaussian：完整开发集验证报告

2026-10-31。唯一预先指定主方案 `anchor_trace (h=0.25,T=10)` **未通过固定开发筛查**。60 项要求必须同时成立，实际为 58 项通过、2 项失败；不能将其解释成“近似全部通过”。两项失败来自同一个 `g24_g8` 区域 KL 反例：旧 strong quad 与 legacy quad16 在本次恰为同一完整配置，保留两个协议角色，但不把它们算成两份独立反证。

执行、数值和独立指标审计已通过，允许解释这些开发结果；这与方案效果通过是两件事。结论是停止这个固定构造，不开启旧 VAL、实际 V/深度训练、GS/RGB、fresh 数据或补充参数网格，不从归因角色中事后挑选替代主方案。

**验证范围与算子。** 仅使用原有 48 对 TRAIN（c3/c5/c6 各 16 对），全部 14 个原生 query/key 条件。`g8/g16/g24` 表示 8/16/24 像素边长分区，`mixed` 保持原冻结混合分区。新前向共 `48×14×12=8,064` 个 float64 概率矩阵，全部保存。区域和中心的有效性定义分别沿用原合同；不依据预测是否在视野内、是否在相机前方、质量大小或历史成功情况过滤。

令 `a` 为原 R13 指定 query 中心通过合法预测深度/相机投影到 key 图像的位置，除以 8 后保留 float64；`μ,C` 是原 R17 的 float32 full1 矩，提升到 float64 后计算：

```text
d = μ - a
C_a = C + d dᵀ
S = C_a + h² I
τ = T h² / (h² + tr(C_a))
P_k = softmax_k(log[∫_{B_k} N(x;a,S)dx + 10⁻³⁰ area_k/64]
                + float32(τ) × content32)
```

几何积分、最终 logit 与 softmax 为 float64；内容特征及 τ×content 乘法沿用冻结 float32 语义。Gaussian 生产积分使用 GL32，24px 原生 key 由四个 12px 子积分求和后仅加一次原生面积 floor；quartic 控制保留原 GL24。没有新增网络参数或 h/T 搜索。

`C_a` 是原 support 关于指定中心 a 的二阶矩，并非关于自身均值的中心协方差，也不是预测置信度。a 是原 μ/C 之外的合法信息；本实验不能推出“仅凭既有 GS 均值和协方差”也能获得相同收益。对指定中心本就等于均值的 Gaussian，a=μ 时退化为原矩。

**聚合与读表约定。** 区域先在每个 scene/condition 内按完整可见像素质量加权，再对场景等权。common9 是 g8/g16/mixed 的 3×3 条件，先映射到共同 g16-query/g8-key 表示；这九项的 `direct` 列另报原生分区指标。direct5 是含 g24 的五项，其 KL/JS 与 direct KL/JS 相同。`J=(common9 KL+direct5 KL)/2`。KL/JS 越低越好；计算 log 时沿用原 `1e-15` 概率下界，与模型 `1e-30×area/64` floor 不同。本文四舍五入仅供显示，判定来自已冻结浮点比较和精确 Fraction，不加容差。

**全部固定筛查。**

| 要求组 | 通过/总数 |
| --- | --- |
| 旧区域宏观保护 | 8/8 |
| 14 条件×2 旧 quad 保护 | 26/28 |
| 同 a 三种区域控制 | 6/6 |
| 旧中心保护 | 9/9 |
| 同 a 中心保护 | 9/9 |

完整 60 项原值、精确 Fraction、bound 与布尔结果见 [全部检查 CSV](/home/liwt/GRoPE/results/round20_anchor_report_tables/screen_all_checks.csv)；[失败检查 CSV](/home/liwt/GRoPE/results/round20_anchor_report_tables/screen_failed_checks.csv) 保留两条角色记录。

| 失败角色 | 主方案 KL | 旧参考 KL | 允许上界 (=参考×1.02) | 相对劣化 |
| --- | --- | --- | --- | --- |
| strong_quad | 0.3325413374491674 | 0.3083528293223204 | 0.3145198859087668 | 7.844426% |
| legacy_quad16 | 0.3325413374491674 | 0.3083528293223204 | 0.3145198859087668 | 7.844426% |

**全部 14 条件的区域结果。** 下表为主方案三场景等权值；最后一列为主方案相对旧 strong quad/legacy quad16 的 KL 变化，正值表示更差。全部 12 新角色的每条件×每场景 KL/JS/direct、有效像素质量，以及所有继承旧角色，均在 [完整场景区域 CSV](/home/liwt/GRoPE/results/round20_anchor_report_tables/regional_scene_conditions.csv)（新角色 504 行，含旧角色共 1,386 行）和 [等场景区域 CSV](/home/liwt/GRoPE/results/round20_anchor_report_tables/regional_equal_scene.csv)。

| query_key | KL | JS | direct KL | direct JS | 旧 quad KL | KL 变化 |
| --- | --- | --- | --- | --- | --- | --- |
| g8_g8 | 0.305780 | 0.064845 | 0.473843 | 0.088981 | 0.320629 | -4.631% |
| g8_g16 | 0.697918 | 0.177833 | 0.268216 | 0.049878 | 0.702748 | -0.687% |
| g8_mixed | 0.501005 | 0.121914 | 0.369477 | 0.069771 | 0.511397 | -2.032% |
| g16_g8 | 0.329501 | 0.078625 | 0.329501 | 0.078625 | 0.338795 | -2.743% |
| g16_g16 | 0.711198 | 0.181813 | 0.186681 | 0.044057 | 0.728390 | -2.360% |
| g16_mixed | 0.518990 | 0.130932 | 0.256390 | 0.061204 | 0.532729 | -2.579% |
| mixed_g8 | 0.319541 | 0.072098 | 0.399997 | 0.084619 | 0.330415 | -3.291% |
| mixed_g16 | 0.705769 | 0.179952 | 0.225069 | 0.047368 | 0.716019 | -1.431% |
| mixed_mixed | 0.511095 | 0.126578 | 0.309941 | 0.065934 | 0.522569 | -2.196% |
| g8_g24 | 0.183900 | 0.034521 | 0.183900 | 0.034521 | 0.190437 | -3.433% |
| g24_g8 | 0.332541 | 0.085785 | 0.332541 | 0.085785 | 0.308353 | +7.844% |
| g16_g24 | 0.123595 | 0.028740 | 0.123595 | 0.028740 | 0.136343 | -9.350% |
| g24_g16 | 0.181789 | 0.046915 | 0.181789 | 0.046915 | 0.196339 | -7.411% |
| g24_g24 | 0.120872 | 0.031129 | 0.120872 | 0.031129 | 0.141791 | -14.753% |

`g24_g8` 的退化分布在三个场景，均超过 2%；这只是报告观察，不额外添加“所有 scene 必须通过”的事后规则。

| 场景 | 主方案 KL | 旧 quad KL | 相对劣化 | 主方案 JS | 旧 quad JS |
| --- | --- | --- | --- | --- | --- |
| harvard_c3 | 0.264349 | 0.242782 | +8.883% | 0.071098 | 0.057003 |
| harvard_c5 | 0.262381 | 0.222470 | +17.940% | 0.071845 | 0.059232 |
| harvard_c6 | 0.470894 | 0.459806 | +2.411% | 0.114413 | 0.097269 |

**全部 12 个新角色的宏观区域和中心结果。** PCK 为 key=g8 时的三场景等权百分比，阈值 8px；不同 query 尺度分别报告。除第一行外，其余均为预先声明控制/归因，不是可事后替换的候选。

| 角色 | h/T | common9 KL | direct5 KL | J | q8 PCK% | q16 PCK% | q24 PCK% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_trace | 0.25/10 | 0.511200 | 0.188540 | 0.349870 | 93.5281 | 92.9328 | 93.4098 |
| mean_only_matched_tau | 0.25/10 | 0.516070 | 0.192177 | 0.354124 | 93.5378 | 92.9771 | 93.4098 |
| anchor_constant | 0.25/10 | 0.602972 | 0.325122 | 0.464047 | 93.3144 | 92.7194 | 88.7698 |
| anchor_isotropic_matched | 0.25/10 | 0.546597 | 0.219253 | 0.382925 | 93.5805 | 93.0293 | 93.4101 |
| anchor_point_regional | 1/5 | 0.649218 | 0.323059 | 0.486139 | 92.1280 | 92.6084 | 92.5002 |
| anchor_point_localizer | 0.25/5 | 1.747605 | 3.082677 | 2.415141 | 93.6497 | 92.9458 | 93.6238 |
| anchor_point_legacy | 0.5/5 | 0.788247 | 0.937580 | 0.862913 | 93.6539 | 93.1299 | 93.3576 |
| anchor_point_matched | 0.25/10 | 1.854327 | 3.233456 | 2.543891 | 93.6481 | 93.0763 | 93.3576 |
| anchor_fixed_regional | 0.5/5 | 0.621491 | 0.322385 | 0.471938 | 93.5583 | 93.1273 | 92.5002 |
| anchor_fixed_matched | 0.25/10 | 0.793884 | 0.494200 | 0.644042 | 93.4759 | 92.8145 | 90.2620 |
| anchor_quartic_regional | 1/5 | 0.732860 | 0.399992 | 0.566426 | 88.6639 | 88.8820 | 86.2070 |
| anchor_quartic_localizer | 0.25/5 | 2.630896 | 5.454814 | 4.042855 | 93.6358 | 92.9458 | 93.5169 |

`mean_only_matched_tau` 使用 a、原 C、与主方案完全相同的 τ(C_a)，仅隔离 rank-one 几何增宽，不能当成使用 τ(C) 的完整 a,C 候选。`anchor_constant` 使用 a,C_a 与常数 T；`anchor_isotropic_matched` 使用 a,tr(C_a)/2·I 与同一 τ(C_a)，同时去除各向异性和方向信息，不能把差异只归因于旋转。point 的协方差为 0；fixed 为原生 query 面积/(64×12)·I；quartic 保留原四次核。区域、localizer、legacy 配置来自旧选择，matched 配置仅对齐 h/T；这些控制**未针对新 a 重新优化**，不存在另外选过的历史 fixed-PCK 配置。

**继承旧控制与同信息比较。** 旧区域/中心证据直接继承已审计行，未重新执行 baseline 前向或数值积分。所有 21 个旧角色/别名和新角色的场景宏观值见 [完整角色宏观 CSV](/home/liwt/GRoPE/results/round20_anchor_report_tables/regional_role_macros.csv)；重复语义别名保留，但不是独立重复实验。代表性完整配置如下，不能据此宣称主方案优于所有旧 family。

| 继承角色 | h/T | common9 KL | direct5 KL | J |
| --- | --- | --- | --- | --- |
| strong_point | 1/5 | 0.643210 | 0.314361 | 0.478786 |
| strong_quad | 0.5/5 | 0.522632 | 0.194653 | 0.358643 |
| regional_gaussian_fixed | 0.5/5 | 0.613975 | 0.311735 | 0.462855 |
| regional_gaussian_full1 | 0.25/0 | 0.516660 | 0.186106 | 0.351383 |
| regional_gaussian_full4 | 0.25/0 | 0.517702 | 0.166856 | 0.342279 |
| regional_gaussian_quad4 | 0.5/0 | 0.566866 | 0.226865 | 0.396866 |
| legacy_full4 | 0.5/5 | 0.512933 | 0.182365 | 0.347649 |
| legacy_quad4 | 0.5/5 | 0.547268 | 0.246758 | 0.397013 |
| old_trace | 0.25/10 | 0.511794 | 0.184888 | 0.348341 |
| historical_primary | 1/0 | 0.899027 | 0.377000 | 0.638013 |

主方案相对旧 strong/legacy point 的 common9/direct5 KL 分别下降 **20.524%/40.024%**，相对旧 strong/legacy quad16 下降 **2.188%/3.141%**。但旧 regional full4 的 direct5 KL=0.166856、J=0.342279，均低于主方案的 0.188540、0.349870；宏观对 quad16 的改善并不等于对所有旧强控制占优。与固定旧 R17 `old_trace` 相比，common9 仅下降 **0.116%**，direct5 反而上升 **1.975%**，J 上升 **0.439%**。

下表为主方案相对同 a 控制的 KL 降幅，正值表示主方案更低。三种区域控制的两项降幅均超过冻结的 2%，说明这套固定几何与内容组合的收益不能仅由提供 a 解释；但这些控制的 h/T 和温度法则也不同，单独协方差作用仍需看 matched τ 的 isotropic/mean-only 归因，不能据此证明新 a 下的最优 baseline 已被击败。

| 同 a 参考 | common9 降幅 | direct5 降幅 |
| --- | --- | --- |
| anchor_point_regional | 21.259% | 41.639% |
| anchor_quartic_regional | 30.246% | 52.864% |
| anchor_fixed_regional | 17.746% | 41.517% |
| mean_only_matched_tau | 0.944% | 1.893% |
| anchor_constant | 15.220% | 42.010% |
| anchor_isotropic_matched | 6.476% | 14.008% |

归因显示：固定 τ 后，rank-one 增宽只带来较小区域改善，q24 PCK 在三个场景均完全相同，q8/q16 宏观 PCK 反而分别低 0.009687/0.044285 个百分点；没有证据称增宽本身提升了中心定位。q24→g8 的等场景 mean/RMSE 也由 mean-only 的 3.983655/4.810006px 增至主方案的 3.990773/4.836492px；该条件 KL 改善 1.434%，JS 却由 0.085743323 增至 0.085785274。使用完整形状相对同 trace 的 isotropic 几何降低了区域 KL，但中心 PCK 没有同步提升。τ(C_a) 相对常数 T 对区域及 q24 中心均有明显影响，因此收益不能全部归于 a 或协方差朝向。上述观察只解释固定构造，不提升任何归因角色。

**JS 保留了与 KL 不同的证据。** 相对旧 quad16，`g24_g8` 的 JS 上升 **20.539%**；`g24_g16` 虽然 KL 下降 **7.411%**，JS 却上升 **8.191%**；`g24_g24` 的 KL 下降 **14.753%**，JS 上升 **2.786%**。direct5 平均 JS 由旧 quad16 的 **0.043698318** 增至 **0.045417979**，尽管 direct5 平均 KL 改善。相对旧 R17 trace，common9/direct5 JS 也都上升（旧值 **0.125622749/0.044401403**，新值 **0.126065375/0.045417979**）。因此不能把 KL 局部下降解释为全分布或各项距离一致变好。JS 是完整报告诊断，不被事后新增为筛查门槛。

**中心整数、精确 Fraction 与连续误差。** 原中心有效数为 q8=9,521、q16=2,463、q24=1,096；每个 key 条件复用同一 query 有效人口，不是新增样本。每 scene 前向 query 数分别为 q8=6,912、q16=1,728、q24=768。PCK 使用原生 key 中心的 first-index argmax，目标距离平方≤64；区域可见像素 mask 与中心有效 mask 不混用。连续误差为全部中心有效行的距离均值和 RMSE，单位 px，不只统计正确点。

下表给主方案所有 11 个可用 query/key 中心条件。每个 scene 单元为“PCK 正确整数/有效整数；mean / RMSE”，最后一列为三场景等权的“PCK%；mean / scene-RMSE 均值”。后者不是把三场景点合并后的 pooled RMSE。mixed query 没有原独立中心标签，不补造。

| query_key | c3：计数；mean/RMSE | c5：计数；mean/RMSE | c6：计数；mean/RMSE | 等场景 PCK%；mean/RMSE |
| --- | --- | --- | --- | --- |
| g8_g8 | 3134/3252；3.542/4.590 | 3365/3441；3.626/4.093 | 2444/2828；4.803/6.405 | 93.5281%；3.990/5.029 |
| g8_g16 | 2440/3252；6.495/7.318 | 2553/3441；6.377/6.939 | 1872/2828；7.405/8.616 | 71.8065%；6.759/7.624 |
| g8_mixed | 2867/3252；5.106/6.084 | 3037/3441；5.019/5.676 | 2194/2828；6.161/7.583 | 84.6672%；5.428/6.448 |
| g16_g8 | 821/851；3.679/4.543 | 859/882；3.452/4.112 | 620/730；5.333/6.798 | 92.9328%；4.155/5.151 |
| g16_g16 | 659/851；6.408/7.090 | 723/882；5.958/6.517 | 440/730；7.979/9.151 | 73.2284%；6.782/7.586 |
| g16_mixed | 703/851；5.515/6.507 | 765/882；4.945/5.666 | 490/730；7.198/8.759 | 78.8222%；5.886/6.977 |
| g8_g24 | 1153/3252；9.391/10.265 | 1276/3441；9.230/9.931 | 931/2828；10.057/11.127 | 35.1527%；9.559/10.441 |
| g24_g8 | 367/382；3.455/4.178 | 395/402；3.685/4.127 | 268/312；4.832/6.204 | 93.4098%；3.991/4.836 |
| g16_g24 | 321/851；9.257/10.098 | 300/882；9.305/10.051 | 244/730；10.249/11.381 | 35.0529%；9.604/10.510 |
| g24_g16 | 292/382；6.279/6.923 | 297/402；6.348/6.966 | 211/312；7.376/8.470 | 72.6495%；6.668/7.453 |
| g24_g24 | 185/382；8.554/9.357 | 218/402；7.958/8.781 | 92/312；10.067/11.048 | 44.0485%；8.859/9.728 |

全部 12 新角色×11 条件×3 场景的 396 行整数、距离和/平方和、Fraction、mean/RMSE，加上可用旧中心 36 行，完整见 [场景中心 CSV](/home/liwt/GRoPE/results/round20_anchor_report_tables/center_scene_conditions.csv)；所有角色等场景 Fraction 与连续指标见 [等场景中心 CSV](/home/liwt/GRoPE/results/round20_anchor_report_tables/center_equal_scene.csv)。继承旧中心没有连续误差的字段保持空值，不能用零或估计填充。主方案 key=g8 的精确等场景 PCK 分别为 `3699701995/3955711662`、`152760647/164377458`、`310795/332722`。

主方案相对固定旧 R17 trace 的 q8/q16/q24 PCK 分别提高 **0.493329/0.629847/1.474287** 个百分点；但同 a 的 localizer 仍可有更高 PCK。以下是主方案减参考的百分点差，负号表示主方案较低，完整精确 Fraction 和所有场景/条件见 [中心比较 CSV](/home/liwt/GRoPE/results/round20_anchor_report_tables/primary_center_comparisons.csv)。

| 参考 | 来源 | q8 Δpp | q16 Δpp | q24 Δpp |
| --- | --- | --- | --- | --- |
| old_trace | inherited | +0.493329 | +0.629847 | +1.474287 |
| point_localizer | inherited | +0.416557 | +0.482238 | +2.106897 |
| quartic_localizer | inherited | +0.381197 | +0.475746 | +2.213735 |
| legacy_point_localizer | inherited | +0.343173 | +0.437953 | +2.344759 |
| anchor_point_localizer | 同 a 新前向 | -0.121571 | -0.012985 | -0.213942 |
| anchor_quartic_localizer | 同 a 新前向 | -0.107684 | -0.012985 | -0.107105 |
| anchor_point_legacy | 同 a 新前向 | -0.125770 | -0.197010 | +0.052179 |

全部旧/同 a 中心保护都通过冻结的等场景“不差超过 0.5pp”要求；通过保护不表示中心最优或逐场景均改善。例如 q24→g8 相对旧 trace 的 c3/c5/c6 PCK 差分别为 −0.523560/+1.741294/+3.205128pp；相对同 a Gaussian localizer，c3 低 0.785340pp、c5 低 0.497512pp。原筛查只限制等场景 PCK，这些 scene 反例如实报告，不新增门槛。key 变粗时 PCK 明显下降，预测输出仍是相应原生 key 中心，不能跨 key 尺度把这个诊断等同于连续坐标估计质量。特别是 q24→g8 的等场景中心 PCK 改善与该条件区域 KL 退化同时存在：指定中心的定位改善不能代替整个 patch 对应分布的保真。

**数值证据与尾部。** 79 项联合纯合成测试通过；独立重建所有 39,888 个唯一 pair/query 中心，最大投影差为 `1.1368683772161603e-12 px`，低于冻结 `1e-9`。8,064 个矩阵的共享源码重放和 argmax 哈希全部一致；这验证可复现性，不是独立数学参考覆盖了所有矩阵。实际标签字节仅在全量前向封存和独立前向/数值审计通过后访问，root 已验收时间顺序。重复旧 TRAIN 标签仍是开发数据，不成为 fresh 或统计盲测。

独立数学参考固定取每 scene 第一对、14 条件、12 角色、每矩阵 8 行：504 个矩阵、4,032 个 query-condition-role 行。每角色为 42 个矩阵/336 行次，但只有 96 个唯一 query 身份；跨 key 条件和角色重复，不能把 4,032 当作独立数据样本。Gaussian 使用独立 GL64/96；quartic 使用冻结的独立四次核 GL64/96。完整 reference 独立重建矩/内容，hybrid 使用执行时统计量再独立积分，二者用于区分统计量/浮点路径差异和积分差异。

| 角色 | 最大完整参考 P 误差 | 最大 hybrid P 误差 | 最大参考64/96差 | 全量 raw FOV=0 行次 | 全量 raw FOV<1e−20 行次 |
| --- | --- | --- | --- | --- | --- |
| anchor_trace | 5.467e-07 | 9.159e-16 | 1.277e-15 | 95 | 18590 |
| mean_only_matched_tau | 2.656e-07 | 9.437e-16 | 1.221e-15 | 111 | 18826 |
| anchor_constant | 1.032e-06 | 1.443e-15 | 1.998e-15 | 95 | 18590 |
| anchor_isotropic_matched | 1.848e-06 | 6.328e-15 | 9.992e-15 | 11 | 20017 |
| anchor_point_regional | 5.619e-07 | 4.441e-16 | 5.551e-16 | 2929 | 17115 |
| anchor_point_localizer | 4.631e-07 | 1.443e-15 | 1.887e-15 | 16475 | 35396 |
| anchor_point_legacy | 4.776e-07 | 7.772e-16 | 1.055e-15 | 8062 | 25961 |
| anchor_point_matched | 9.712e-07 | 1.887e-15 | 2.276e-15 | 16475 | 35396 |
| anchor_fixed_regional | 5.619e-07 | 6.661e-16 | 6.661e-16 | 5859 | 23012 |
| anchor_fixed_matched | 1.034e-06 | 1.443e-15 | 1.776e-15 | 9512 | 27877 |
| anchor_quartic_regional | 5.824e-07 | 6.661e-16 | 7.772e-16 | 16155 | 25642 |
| anchor_quartic_localizer | 1.254e-06 | 8.514e-07 | 2.248e-13 | 34648 | 41792 |

全角色最大完整参考 P 误差为 `1.847751959771493e-6`（isotropic），最大 hybrid 误差为 `8.513737076443206e-7`（quartic localizer），均低于 `1e-4`；最大参考64/96差 `2.248201624865942e-13` 低于 `1e-8`。同 raw mass 算术误差最大 `4.440892098500626e-16`，τ32/bias32 全部精确一致；全量最大归一化误差 `7.993605777301127e-15`，所有保存 P 有限。独立最终指标审计的最大行差 `7.275957614183426e-12`、汇总差 `5.820766091346741e-11` 均在冻结数值比较范围内；这些容差不用于放宽效果界限。

独立 reference 与 hybrid 在这些固定参考行中均没有 argmax 变化，[argmax 差异 CSV](/home/liwt/GRoPE/results/round20_anchor_report_tables/numerical_argmax_changes.csv) 仅保留表头。这不能推出所有行不存在相等/近并列：表里的 exact/near-tie 计数来自“argmax 已发生差异”的记录集合，该集合为空不构成全体 tie 检测。

主方案参考样本中，原始 FOV 质量为零的行次是 0，小于 `1e-20` 的是 7（仅 2 个唯一 query），最小非零值约 `1.06847e-25`；全量保存诊断则有 **95** 个零质量行次、**18,590** 个 tiny 行次。tiny 包含 zero，不能相加。上表全量各角色的分母都是 145,584 个 query-condition 行次、39,888 个唯一 query 身份；不能把行次计数除以唯一 query 数当作比例，也不能推断这些行全部属于 GT 有效人群。相同几何的不同温度角色会重复同一 tail 事件。原始质量是加 floor 之前的量，数值零不能被称为数学上不相交；面积 floor 总和为 `4.32e-28`，令这些行仍有定义，且不因预测质量被剔除，评价仍服从原标签有效 mask。这不证明它们具有可靠几何定位信息。核验的是加入 floor 后概率的绝对误差，不保证极小原始质量的相对精度。没有删除零/极小质量行，也不把抽样没有发现的失败推论成全量积分误差保证。

完整逐矩阵参考/hybrid/convergence、argmax 与尾部行见 [数值检查 CSV](/home/liwt/GRoPE/results/round20_anchor_report_tables/numerical_checks.csv)、[全部尾部 CSV](/home/liwt/GRoPE/results/round20_anchor_report_tables/numerical_tail_rows.csv)、[参考按角色汇总](/home/liwt/GRoPE/results/round20_anchor_report_tables/numerical_by_role.csv)、[全量 raw mass 按角色汇总](/home/liwt/GRoPE/results/round20_anchor_report_tables/full_forward_numerics_by_role.csv)。12,096 个尾部记录是 4,032 行次×actual/reference/hybrid 三个版本，不能当作 12,096 个独立样本。

**来源和可追溯性。** 本报告只重排已接受结果；未新增前向、标签、网格、模型选择或独立审计。写作前重检 root receipt 所有文件 SHA256 和交付 completion 中全部输出 SHA256。完整数值保留在 [交付 report.json](/home/liwt/GRoPE/results/round20_anchor_report_tables/report.json)；新/旧逐场景、逐条件及宏观差异在 [完整区域比较 CSV](/home/liwt/GRoPE/results/round20_anchor_report_tables/primary_regional_comparisons.csv)。

| 证据 | 链接/身份 |
| --- | --- |
| 冻结协议 | [ROUND20_ANCHOR_FIXED_PROTOCOL.md](/home/liwt/GRoPE/ROUND20_ANCHOR_FIXED_PROTOCOL.md) |
| Producer 锁 | [lock.json](/home/liwt/GRoPE/results/round20_anchor_fixed/lock.json)；`3df45a47d53df9d33d2a99eb2f0d30983a64f581a465a126e8d1a551d891bd4b` |
| 独立 auditor 锁 | [lock.json](/home/liwt/GRoPE/results/round20_anchor_independent_audit/lock.json)；`826bb03c19894d12ccd67850a9b96a154a62956c2816de949c54f05de0f96f88` |
| 全量前向/数值审计 | [forward_verification.json](/home/liwt/GRoPE/results/round20_anchor_independent_audit/forward_verification.json) |
| 独立最终指标审计 | [verification.json](/home/liwt/GRoPE/results/round20_anchor_independent_audit/verification.json) |
| Root 验收 | [acceptance receipt](/home/liwt/GRoPE/results/round20_anchor_root_acceptance_20261031.json)；`5126e73ab29588ef27de99cd64b7e5120df0aeec508663fb8031489d5c6b8f4f` |
| 接受的主 summary SHA256 | `67d6b81b1f63bef26389d2d77de67583cc6f54245dee0ded98a1204b360c66e2` |
| 交付完整性清单 | [report tables completion](/home/liwt/GRoPE/results/round20_anchor_report_tables/completion.json) |

保留下来的研究事实是：这个固定新构造相对旧 trace 提高了本开发人口的中心等场景 PCK，但没有逐场景一致改善；完整 support 形状和 trace 温度在固定控制下仍有区域作用；将二阶矩移到 a 并保持单 Gaussian，仍无法同时守住全部区域尺度要求。R17 的原科学失败、R18 的原数值失败不被本轮改写。没有旧 VAL、扰动/成本、实际学习 Q/K/V 与等容量 head、优化 GS/RGB 或未触碰联合确认的证据，因此未得到“具有一般性、已通过全部验证”的方案。后续若继续研究，必须是另有依据的独立假设与新协议，保留本轮失败，不进行救援式调参。
