# Round14：关闭内容不足以解释24px定位失败

2026-10-30。冻结协议：[ROUND14_TEMPERATURE_PROTOCOL](../ROUND14_TEMPERATURE_PROTOCOL.md)。48个旧训练pair、三个场景等权、q8/q16/q24→g8、七个固定角色，无选择网格。全部前向完成后才读已独立审计的R13标签。完成1,008个概率矩阵、1,008条指标及432条配对记录。

**主要结论：内容温度不是24px定位缺口的主要原因。** q24关闭full4内容项仅提高PCK 0.8599pp，回补同一训练集相对point_pck缺口的10.84%；仍落后7.0713pp。此结论来自训练诊断，没有重评或修复R12验证集的−11.20pp失败。

| q24→g8角色 | direct KL | PCK@8 | winner到预测μ距离，中心有效群体 |
|---|---:|---:|---:|
| full4，h=.5/T5 | 0.28998158 | 83.1338% | 5.1727px |
| full4，h=.5/T0 | 0.28972257 | 83.9937% | 5.0780px |
| full1，h=.5/T5 | 0.35719209 | 90.0962% | 3.8017px |
| full1，h=.5/T0 | 0.34072754 | 91.4013% | 3.1175px |
| point_pck，h=.5/T5 | 1.91442977 | 91.0651% | 3.1465px |
| quad16，h=.5/T5 | 0.30835282 | 84.8244% | 4.9034px |
| 旧full1，h=.25/T0 | 0.32604103 | 91.9116% | 3.1261px |

μ是全区域投影像素均值、与冻结point控制完全相同；它不是sensor中心。完整结果另有概率期望位置到μ的偏移、μ附近8px概率质量、μ到sensor中心误差，以及全query与中心有效两种分母。

full4关闭内容后的direct KL相对变化/PCK变化：q8 **+24.6858% / −0.4343pp**，q16 **+12.1838% / +0.2553pp**，q24 **−0.0893% / +0.8599pp**。因此全局切换T0会明显损害小尺度区域KL。

q24的full4 PCK变化也不场景一致：c3 −0.2618pp、c5 +3.4826pp、c6 −0.6410pp。关闭内容显著改变约42%–57%的中心有效query的winner，却仅小幅改变整体PCK；它不能独自解释剩余定位损失。

同h=.5、T0下，full1比full4高7.4076pp PCK，但direct KL差17.6048%；同h=.5、T5下，full1高6.9624pp PCK、KL差23.1775%。几何形式之间的区域/中心权衡在匹配h/T后仍明显存在。旧full1的h=.25相对同T0的h=.5再提高0.5103pp PCK、降低4.3103% KL；这说明旧配置确有带宽贡献，但不能消除匹配后的几何差别。

6项实现合成测试和4项独立审计测试通过。独立审计复算全部1,008指标、432配对记录、720几何诊断与来源/阶段链；指标及汇总误差为0。固定metadata-first数值子集对独立GL64的最大概率差5.8947e−6，全部pair的T0→T5同几何重建最大差3.1382e−7，保存区域均值误差≤0.000659px。69个完成工件哈希另经重算。

下一步可以继续研究支撑近似的峰值结构及区域/中心表征关系；本轮不选择新T/h、不新增模块、不改变任何旧gate，也不允许据此启动被R12阻止的下游升级。

工件：[汇总](../results/round14_temperature/summary.json)、[全部指标](../results/round14_temperature/rows.json)、[配对指标](../results/round14_temperature/paired_rows.json)、[独立审计](../results/round14_temperature_independent_audit/verification.json)、[完成哈希](../results/round14_temperature/completion.json)。lock SHA256：`7a1bd092abdd904465ad0d58bfd6fba16aa288fc69444323cd8620c80a9c41db`。
