# R36 后的真实输入初始成本下界诊断

2026-11-01，执行前方案；这不是 R37 正式参考运行或效果选型。

固定读取已接受 R35 的全部72个 prepared 归档，保持原顺序，只解码原有四项 model input 并校验 canonical hashes。只做 RMS/SVD 几何、原 outer_breaks/断点合并、固定 pilot 和 GK21 横坐标的 X 分区计数。不计算正态 PDF/CDF、front probability、log integrand、求积加权和或概率，不调用参考/生产模块。源、依赖、运行时、两个验收 receipt、全部72个归档及原始人口在第一次数组读取前封存；旧文件不变。

对每个合法正深度回调，每个原生 cell 加独立 direct-FOV rectangle 的标准化 X 区间，按原二进制64运算求 low/high。若 high>low，内部全局宽2网格的数量为

`max(0, ceil(high/2) - floor(low/2) - 1)`；若 rounded high==low，使用原 positive-halfwidth 单 panel 分支。采用 Python 整数，无完整网格分配。原 guard 的 minimum panels 为各 rectangle 的网格点数加433；后续 Y 断点只可能增加有效 panel 数。因此每个成功回调至少需要 order × minimum_panels 个 X 节点。间距/域/有限性检查失败独立记为 geometry_failure，不把它们当低成本成功。

随机深度分支的成功执行必须在第一次收敛判断前完成 K 个 pilot 和全部 K×21 个初始 GK21 回调。独立提取冻结 helper 的 X 节点，保持原先后次序。每个回调用同一几何计算一次，累计 panel 下界，再分别乘32/48。严格正深度路径与原实现相同：z<=0且zeta<=threshold时该回调无 inner 节点；zeta>threshold而算出的z<=0必须记录失败。原 z=0/负常深度为 structural_zero；正常深度只计算一次 rectangle panel 下界。原 factor_z 非零却归一化 sigma=0不能改成常深度。

不计算 front probability，因此不认证被省略的 front 浮点准入；所得计数表述为“若该冻结参考成功完成，则必需的初始节点下界”。若任何完整或已完成前缀的 order×panel 下界严格大于250,000,000，则该调用不可能在原预算内成功；更早数值失败并不推翻此必要条件。等于预算不是反例。未超过预算仅为 unknown，完全不能预测自适应细分、实际窄CDF节点、精度、墙钟或最终通过。

诊断固定300秒，覆盖装载、几何和结果写出，每个回调前检查。每次只处理一个归档，不因途中出现反例更换或删除人口；超时后全部剩余人口明确 not_processed，保留已完成前缀和72条最终状态。每例保存原始/保留边界、pilot坐标等几何证据，另存 progress JSONL、每例整数账本及完整库存；没有概率数组。独立数值调用、参考积分、PDF/CDF次数均为0，真实数组读取如实记录。超时未完成的行不视为低成本通过。

首阶段仅 minimum-X admission 下界，不构造 Y roots 或 inner GL 节点。若未发现反例，结论只能是“本下界尚未排除预算可行性”；是否另冻完整几何计数或正式144次参考由 root 再决定。不得据此选择成功子集、放宽250M/36B/时间/精度门槛或改写R35/R36结果。
