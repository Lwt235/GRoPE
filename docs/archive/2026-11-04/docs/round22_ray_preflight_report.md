# R22：单个 3D Gaussian 射线投影的固定数值预检

2026-10-31。**15 个基础样本、8 个恒等式检查与 30 次独立参考计算全部通过冻结的数值门槛；这不是效果候选通过。** 本轮验证了给定少量合成输入时，保留单个 3D Gaussian、在正深度域做透视投影并积分到 image cells 的实现具有可信的数值结果。没有读取 TRAIN/VAL/GS/fresh 输入或标签，也没有评估区域 KL、中心 PCK、学习 Q/K/V 或实际渲染效果。

基础样本最大 P 绝对误差为 **1.915134717478395e−15**。即使 raw FOV mass 只有 **1.776482112060074e−33**，其相对误差也通过，结果没有只依赖面积 floor 掩盖小质量误差。但变深度相关 Gaussian 的单次 16-cell 调用耗时 **0.484642116秒**，明显高于其他路径，不能把整个预检通过解释为实际大规模计算已可用。

**被验证的对象。** 输入是 camera-frame `X=μ+Lξ`，`ξ~N(0,I_r)`，float64 的 μ[3]、L[3,r]，r=0…3。不是从历史 2D μ/C 反推 3D。定义

```text
M(B) = E[ 1{Z>0} · Pr(HU/Z + c + ε ∈ B | X) ],  ε~N(0,h²I)
P(B) = [M(B) + 1e−30·area(B)] / [Σcells M + 1e−30·Σcells area]
```

基础样本固定 H=I、c=0、h=.25 reference-8 单位，FOV=[−2,2]² 的 16 个单位 cell，T=0，无内容模块。固定总 floor 为 **1.6e−29**。严格保留 `Z>0` 的原始概率质量，不先除以 p+=Pr(Z>0)，也不进行 blur 前 FOV 剔除。p+不是遮挡可见性；零 raw 质量返回明确记录的面积先验。

v=||Lz||²>0 时，生产路径按 Z 条件化：β=ΣUz/v，D=(LU N)(LU N)ᵀ，N 为 Lz 的正交零空间基。每个 z 下的 image Gaussian 均值为 H[μU+β(z−μz)]/z+c，covariance 为 HDHᵀ/z²+h²I，再积分原始正深度 Gaussian 密度。这里没有额外 z² Jacobian。v=0 走精确分支；不通过阈值把微小非零 v 变为零。因数值下溢无法表示的非零因子会显式失败，不加 jitter 或修补 covariance。

**生产与独立参考分工。** 生产使用固定规则的 adaptive GK21 深度积分：standard-normal t 的范围为 [max(−14,−μz/√v),14]，epsabs=1e−12、epsrel=1e−10；初始断点由密度位置及投影均值与 cell 边界的几何交点决定。相关 image covariance 另做一维条件矩形积分；全部预算在前向前冻结。对很小 p+ 只作数值缩放，加入 floor 前还原原质量。

参考不调用生产条件化/积分/归一化代码：point 与 radial 使用解析式，固定深度平面使用独立 GL，过相机平面使用半质量 Cauchy 公式，各向同性与相关零均值 fullrank 使用 solid-angle cone 加 GH blur；offset line 与倾斜平面通过 projected coordinate 的逆映射及 Jacobian 使用固定 GL panels。GL/GH 两阶均为 64/96。NumPy/SciPy 基础函数是披露的共享依赖，代码独立不意味着数学库完全独立。

新增的 correlated_origin 在首次正式前向前确定：μ=0，L=[[1,0,0],[.4,.7,0],[.3,−.2,1]]。其 v=1.13、条件 D 的非对角项为 .458/1.13，真实覆盖随 z 变化的相关积分路径。独立参考将 cell 四角 r=(x,y,1) 映到 L⁻¹r 的单位方向，分成两球面三角形，立体角之和除以 **4π** 后再积分 blur，保留 p+=.5。采用冻结协议所列经典三角形公式，不主张公式或分布家族的新颖性。

**全部 15 个基础样本。** raw 为原生 cell 质量之和；P误差、FOV相对误差和 L1/raw-FOV 误差均对独立 reference96。精确零情形按 raw 必须全零检查，不引入任意小分母。下列全部通过；详细每 cell 误差、ties、argmax 与双阶证据见 [comparisons.json](/home/liwt/GRoPE/results/round22_ray_independent_audit/comparisons.json)。

| 样本 | p+ | raw FOV | 最大P绝对差 | FOV 相对误差 | raw L1/ref FOV | 生产秒 | depth / inner evals |
| --- | --- | --- | --- | --- | --- | --- | --- |
| point_front | 1.000e+00 | 1.000e+00 | 2.776e-17 | 0.000e+00 | 4.218e-17 | 0.001285 | 0 / 0 |
| point_behind | 0.000e+00 | 0.000e+00 | 0.000e+00 | 精确零检查 | 精确零检查 | 0.000467 | 0 / 0 |
| point_camera_origin | 0.000e+00 | 0.000e+00 | 0.000e+00 | 精确零检查 | 精确零检查 | 0.000455 | 0 / 0 |
| point_outside_fov | 1.000e+00 | 3.167e-05 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000672 | 0 / 0 |
| plane_constant_diagonal | 1.000e+00 | 9.996e-01 | 2.220e-16 | 1.111e-16 | 4.274e-16 | 0.000618 | 0 / 0 |
| plane_constant_correlated | 1.000e+00 | 9.993e-01 | 1.110e-16 | 1.111e-16 | 3.776e-16 | 0.003348 | 0 / 504 |
| radial_front | 9.772e-01 | 9.772e-01 | 1.110e-16 | 1.136e-16 | 1.283e-16 | 0.023945 | 126 / 0 |
| radial_behind | 2.275e-02 | 2.275e-02 | 1.110e-16 | 7.625e-16 | 7.742e-16 | 0.016741 | 105 / 0 |
| radial_below_floor | 1.776e-33 | 1.776e-33 | 2.914e-16 | 4.363e-12 | 4.363e-12 | 0.010384 | 63 / 0 |
| offset_rank1_beta32 | 8.413e-01 | 6.183e-02 | 1.665e-16 | 1.122e-16 | 1.094e-15 | 0.048857 | 315 / 0 |
| offset_rank1_beta512 | 8.413e-01 | 3.873e-03 | 1.915e-15 | 1.680e-15 | 8.540e-15 | 0.048803 | 315 / 0 |
| plane_through_origin | 5.000e-01 | 3.508e-01 | 2.776e-17 | 1.582e-16 | 1.187e-16 | 0.023248 | 147 / 0 |
| isotropic_origin | 5.000e-01 | 2.924e-01 | 1.110e-16 | 1.899e-16 | 3.589e-16 | 0.023019 | 147 / 0 |
| correlated_origin | 5.000e-01 | 3.263e-01 | 1.110e-16 | 6.804e-16 | 5.688e-16 | 0.484642 | 147 / 97650 |
| plane_tilted | 9.772e-01 | 8.366e-01 | 5.551e-17 | 1.327e-16 | 1.827e-16 | 0.035923 | 231 / 0 |

四个 point 分别检查前方、后方、camera origin 与 FOV 外；两个固定深度 plane 检查对角/相关 covariance。三个 radial 的 depth mean 为 2、−2、−12。两个 offset rank1 的 β 为 32/512，用于检查窄深度区间；剩余四项分别是过相机平面、各向同性 fullrank、相关 fullrank 和固定倾斜 rank2 plane。完整 float64 μ/L/边界由冻结输入文件给出，没有根据结果更换样本。

`point_behind` 与 `point_camera_origin` 的 raw 精确为零，16 个 cell 全为面积先验，first argmax=0 没有定位含义。`point_outside_fov` 的 raw 仍为 3.167124183311986e−5，符合 blur 后可流入 FOV 的定义；没有提前删除。rank0 相机原点与任意正尺度的零均值 isotropic Gaussian 并不是连续等价的正深度测度。

**30 次独立参考全部保留。** 每行列两阶的实际 cell-node 计数及纯函数计时；解析分支也按两次调用记录，但它们的“64/96一致”不是两种数值积分的独立收敛证据。共 **1,038,048 cell-node evaluations**，计数包含对同一 cell 的重复求积工作，不是 103 万个独立样本。

| 样本 / 方法 | 64 / 96 cell-node | 64 / 96秒 | 64↔96 最大P绝对差 | FOV相对 / L1相对 |
| --- | --- | --- | --- | --- |
| point_front / point解析 | 16 / 16 | 0.000707 / 0.000549 | 0.000e+00 | 0.000e+00 / 0.000e+00 |
| point_behind / point解析 | 16 / 16 | 0.000401 / 0.000399 | 0.000e+00 | 精确零检查 / 精确零检查 |
| point_camera_origin / point解析 | 16 / 16 | 0.000406 / 0.000397 | 0.000e+00 | 精确零检查 / 精确零检查 |
| point_outside_fov / point解析 | 16 / 16 | 0.000536 / 0.000532 | 0.000e+00 | 0.000e+00 / 0.000e+00 |
| plane_constant_diagonal / Gaussian GL | 1024 / 1536 | 0.003723 / 0.004067 | 5.551e-17 | 0.000e+00 / 1.717e-16 |
| plane_constant_correlated / Gaussian GL | 1024 / 1536 | 0.002788 / 0.004030 | 1.110e-16 | 0.000e+00 / 3.298e-16 |
| radial_front / radial解析 | 16 / 16 | 0.000569 / 0.000572 | 0.000e+00 | 0.000e+00 / 0.000e+00 |
| radial_behind / radial解析 | 16 / 16 | 0.000564 / 0.000559 | 0.000e+00 | 0.000e+00 / 0.000e+00 |
| radial_below_floor / radial解析 | 16 / 16 | 0.000596 / 0.000558 | 0.000e+00 | 0.000e+00 / 0.000e+00 |
| offset_rank1_beta32 / projected-u GL | 90112 / 135168 | 0.021371 / 0.030758 | 2.776e-17 | 0.000e+00 / 8.418e-17 |
| offset_rank1_beta512 / projected-u GL | 90112 / 135168 | 0.019966 / 0.029820 | 1.388e-17 | 1.120e-16 / 8.400e-17 |
| plane_through_origin / 半质量Cauchy+GH | 1024 / 1536 | 0.002186 / 0.002682 | 5.551e-17 | 0.000e+00 / 2.374e-16 |
| isotropic_origin / solid angle+GH² | 65536 / 147456 | 0.011427 / 0.023174 | 1.110e-16 | 0.000e+00 / 3.708e-16 |
| correlated_origin / latent cone+GH² | 65536 / 147456 | 0.041090 / 0.109851 | 2.498e-16 | 1.701e-16 / 9.143e-16 |
| plane_tilted / tilted-u GL | 61440 / 92160 | 0.027070 / 0.038222 | 3.469e-18 | 0.000e+00 / 3.117e-17 |

Offset 两个 case 每阶各用 88 个 panels，tilted 每阶用 60 个，均低于 128 上限；参考所有 raw/P 数组按样本保存在 reference_arrays。所有方法、每阶 tail、计数和计时见 [reference_records.json](/home/liwt/GRoPE/results/round22_ray_independent_audit/reference_records.json)。两阶一致是这组固定输入的数值证据，不是整个参数域的误差证明。

**8 个生产恒等式/极限检查。** 只与规定的 base 生产结果比较，没有为这些变换输入另做独立高阶参考。screen 单位变换同时改变 h 和 inverse-square floor density；不等面积 merge 对 raw 和 P 都按预定 cell groups 求和。表中误差门槛统一为1e−8，全部通过。

| 检查 / parent | p+ / raw FOV | 最大raw绝对差 | 最大P绝对差 | 生产秒 | depth / inner |
| --- | --- | --- | --- | --- | --- |
| 世界正尺度×3.7 / offset_rank1_beta32 | 8.413e-01 / 6.183e-02 | 1.388e-17 | 1.943e-16 | 0.048666 | 315 / 0 |
| origin L×1e−6 / isotropic_origin | 5.000e-01 / 2.924e-01 | 0.000e+00 | 0.000e+00 | 0.022867 | 147 / 0 |
| screen单位×3.7 / isotropic_origin | 5.000e-01 / 2.924e-01 | 0.000e+00 | 0.000e+00 | 0.023005 | 147 / 0 |
| screen平移 / isotropic_origin | 5.000e-01 / 2.924e-01 | 0.000e+00 | 0.000e+00 | 0.023134 | 147 / 0 |
| key逆序 / isotropic_origin | 5.000e-01 / 2.924e-01 | 0.000e+00 | 0.000e+00 | 0.023011 | 147 / 0 |
| 合并为面积2/6/2/6 / isotropic_origin | 5.000e-01 / 2.924e-01 | 2.776e-17 | 1.388e-17 | 0.022120 | 147 / 0 |
| factor正交列变换 / plane_constant_correlated | 1.000e+00 / 9.993e-01 | 5.551e-17 | 3.469e-17 | 0.003130 | 0 / 504 |
| 固定L=1e−6I点极限 / point_front | 1.000e+00 / 1.000e+00 | 3.079e-12 | 3.079e-12 | 0.026270 | 168 / 0 |

世界缩放 parent 是 β32 offset line，其余 isotropic 变换均对应 isotropic_origin；factor 变换对应固定深度相关 plane。该 factor 检查是一个具体正交列交换/符号变换，不验证一般 latent rotation 下全部参数。固定 point-limit 只比较一次 ε=1e−6：P误差 **3.0794256034027967e−12**，不能当作有限 fullrank 输入的独立参考或整个 ε→0 极限证明。详情见 [identities.json](/home/liwt/GRoPE/results/round22_ray_independent_audit/identities.json)。

**冻结门槛与最差值。** 全部采用 conjunction，没有把通过项平均后覆盖失败项；本次没有正式样本失败或参考缺失。生产 P 门槛直接比较封存 P，另外检查由同 raw 重算 P 的算术一致性。

| 检查 | 实际最差值 | 冻结上限 |
| --- | --- | --- |
| 生产 vs ref96：P绝对误差 | 1.915e-15 | 1e−4 |
| 生产 vs ref96：raw FOV相对误差 | 4.363e-12 | 1e−6 |
| 生产 vs ref96：raw L1/ref FOV | 4.363e-12 | 1e−6 |
| ref64 vs96：P绝对误差 | 2.498e-16 | 1e−8 |
| ref64 vs96：raw FOV相对误差 | 1.701e-16 | 1e−8 |
| ref64 vs96：raw L1/ref FOV | 9.143e-16 | 1e−8 |
| 23生产P归一化 | 2.220e-16 | 2e−12 |
| 同raw重算P算术差 | 1.110e-16 | 2e−12 |
| 独立direct FOV − native cell sum | 1.110e-16 | 1e−8 |
| 生产/参考 omitted-tail放大界 | 1.948e-15 | 1e−8 |
| 8identity最大raw / P差 | 3.079e-12 / 3.079e-12 | 各1e−8 |
| 两项exact-zero raw | 均精确0 | 精确0 |

**极小质量没有被 floor 绕过。** radial_below_floor 的 p+=1.776482112077654e−33，生产 raw FOV=1.776482112060074e−33，仅为总 floor 的约 1.1103e−4。该项 P 绝对误差为2.914335439641036e−16，同时 raw FOV 相对误差 **4.363464346702341e−12**、raw L1/ref FOV **4.363492195971912e−12**。后两项都直接在 floor 前计算，是通过的必要条件。floor 占主导并不表示获得可靠定位信息；这里检验的是测度与数值实现。

基础样本全部最大逐 cell 相对误差约4.364e−12，已逐 cell 报告；协议没有设置每个极小 cell 的独立相对误差门槛，不能由本次观察宣称一般逐 cell 相对精度保证。基础样本的生产/ref96与ref64/ref96均无 first-argmax 变化，但多项对称/零质量输入确实有 ties，且同一最大项与微小 P 误差都不保证未来任意输入的排序稳定性。

**尾部界与积分误差估计分开。** 下表 T 只界定明确被截断的事件质量。界 `2T/total_floor` 使用真实分母的固定下界，约束该截断对单个归一化 P 分量的影响；它不包含求积或舍入误差，也不是 L1 总误差证书。

| 方法/分支 | omitted raw FOV界 T | 2T/total_floor |
| --- | --- | --- |
| 生产：v=0 | 0 | 0 |
| 生产：v>0，截断 abs(t)>14 | 1.558707363838560e−44 | 1.948384204798200e−15 |
| 参考：point/radial/constant/Cauchy/solid-angle | 0（无显式截断域） | 0 |
| 参考：两个offset projected-u panels | 1.3114102512239213e−44 | 1.6392628140299015e−15 |
| 参考：tilted projected-u panels | 1.5232465656426343e−44 | 1.904058207053293e−15 |

参考 GH 使用完整 Gaussian 测度的有限阶求积，没有把“有限节点范围外”的概率冒充截断证书。Offset/tilted 的有限 projected-u 范围是 FOV 扩14h，其 T 是对整个 FOV 推导的界，不遗漏 cell union 因子。没有显式域截断的 T=0 也不表示数值积分误差为零。

Outer/inner adaptive 误差估计与库状态完整保存。最大生产 outer estimate 是 point-limit 的 **2.449101454712841e−12**；相关 origin 的 outer estimate为6.032408936342484e−13，direct-FOV inner estimate为1.2307975983118746e−13。这些是数值误差估计，不能代替独立参考或作为 rigorous total bound。

**预算与成本。** 本机冻结环境为 Python3.11.7、NumPy1.26.4、SciPy1.13.1，CPU，OMP/MKL/OpenBLAS均4线程。生产 phase为 **0.9589879718秒**，覆盖输入组装、23次调用及每 case 输出直到最后 records JSON；最终source/inventory/sealing另计。其 completion 写入前额外 bookkeeping为 **0.0630584974秒**，两者合计1.0220467485秒，不包含最终completion写入本身。

参考 phase为 **0.7536364309秒**，包括初始生产校验、30次参考及数组写入、数值/identity检查和最终source核验；最终summary/inventory封存写入不计入该 phase；其completion写入前额外bookkeeping为 **0.0068712533秒**，总计0.7605079934秒。两 phase上限各120秒，参考工作上限200万 cell-node。生产最大depth eval=315（上限32,768），最大inner eval=97,650（上限400万），所有breakpoint/interval/status门槛通过。生产进程峰值RSS为 **92,424KiB，约90.26MiB**，是进程累计高水位，不是单case独占内存或训练显存。

相关 origin 同样仅有16 keys，却需0.484642秒、147个depth eval、97,650个inner eval，约占整个生产phase的一半；同样147个depth eval的 isotropic origin只需0.023019秒。条件化降低了显式积分维数，并未消除 correlated cell primitive 的成本。本次是小型CPU预检，不能线性外推到432 keys、大量query、GPU训练或实际渲染吞吐；也没有用special-law reference替换生产规则。

**开发期失败与修正透明记录。** 正式冻结前的第一轮 producer toy 测试为 **21通过、2失败**：相关矩形积分出现略高于1的浮点结果被严格 `>1` 断言拒绝；tiny radial 的 quadrature 节点舍入到严格排除的 t=a 边界。修正为保留未裁剪的数值及越界量交给独立误差检查，以及只在 `t==a 且 z==0` 的零测度端点返回零并计数；t>a 却无法表示正深度仍失败，没有放宽正式误差或预算。随后原23项通过，新增toy后producer25项通过，最终联合 **49项通过**。这些toy不是正式15+8输入，正式case只在联合冻结后执行一次。

开发记录已持久归档，完整哈希见 [archive_receipt.json](/home/liwt/GRoPE/results/round22_development_unit_history_20261031/archive_receipt.json)；原材料见 [开发期 receipt](/home/liwt/GRoPE/results/round22_development_unit_history_20261031/receipt.json)、[首跑失败日志](/home/liwt/GRoPE/results/round22_development_unit_history_20261031/initial_failure.log)、[精确修正patch](/home/liwt/GRoPE/results/round22_development_unit_history_20261031/boundary_fix.patch)。receipt明确：修正前源码文件是由patch重建，**不是当时的源码快照**；修后23pass日志为工具输出转录，不能混称原始实时日志。最终联合日志见 [49 tests PASS](/home/liwt/GRoPE/results/round22_joint_preflight_20261031_tests.log)，冻结前metadata检查还禁止了np.load和正式integrate调用。原开发期receipt SHA256为`3d2298a332085fc3bf42feef295ed950bd51cdde3778f5ea790f6da8c1952c79`。

**范围与后续边界。** 本轮覆盖rank0、固定/变深度rank1/rank2、零均值各向同性与相关fullrank，两个较窄offset line以及八个指定变换。它没有覆盖任意非零均值fullrank、任意极端相关/病态因子、所有相机/分区/h、整段点极限、任意世界坐标预处理或整个参数域。单个3D Gaussian仍只保留其参数化信息；精确投影不会恢复已由Gaussian闭合丢失的patch高阶信息，也不会自动把3D均值变为指定的源中心。

R20固定主方案的失败、R21区域与中心定位相反的诊断证据均不改变；R21本身没有效果通过门槛。R22仅解除这套固定合成检查中的数值疑问，不通过既有scientific gate，不晋升任何效果候选。任何真实开发比较仍须另行冻结合法相同3D信息下的Jacobian/pushforward与历史2D/full-pixel控制，重新执行完整区域、中心、尺度、压力及后续actual-V/GS/untouched confirmation流程。本报告不打开任何真实输入或标签，也不把反复使用的旧TRAIN/VAL称为fresh确认。

Root 的[结果后决定](/home/liwt/GRoPE/results/round22_ray_post_result_decision_20261031.json)仅保留算子定义，下一项 R23 是72个固定合法真实输入、432 keys上的无标签适用性/成本失败筛查，先封存并审计输入再运行。它仍不能建立一般非零均值数值精度或任务效果；本报告没有执行这一步，标签继续禁止访问。

**来源与完整性。** 写作前重核root receipt全部绑定、producer42个文件/audit37个文件完整清单、12个科学来源的当前文件和两套快照。仅解析已接受JSON与日志，没有重新前向或读取真实数组。Root验收为numerical PASS、scientific false、goal complete false。

| 证据 | 路径 / SHA256 |
| --- | --- |
| 固定协议 | [ROUND22_RAY_PREFLIGHT_PROTOCOL.md](/home/liwt/GRoPE/ROUND22_RAY_PREFLIGHT_PROTOCOL.md) |
| 固定μ/L/identity及策略 | [fixtures.json](/home/liwt/GRoPE/results/round22_ray_preflight/fixtures.json)；[lock.json](/home/liwt/GRoPE/results/round22_ray_preflight/lock.json) |
| 23生产调用 / 数值核验 | [records.json](/home/liwt/GRoPE/results/round22_ray_preflight/records.json)；[production_checks.json](/home/liwt/GRoPE/results/round22_ray_independent_audit/production_checks.json) |
| 独立汇总 | [summary.json](/home/liwt/GRoPE/results/round22_ray_independent_audit/summary.json)；`46d8253c51a0300e8c5220b9e2b620b0ac99dd1cd2af66d41679c0099433488d` |
| 正式执行 / 开发联合检查 | [execution.json](/home/liwt/GRoPE/results/round22_ray_preflight_20261031_execution.json)；[joint_preflight.json](/home/liwt/GRoPE/results/round22_joint_preflight_20261031.json) |
| Root验收 | [acceptance.json](/home/liwt/GRoPE/results/round22_ray_root_acceptance_20261031.json)；`14bde11bbe13eb036660c7390148c9d2db3c105bd7c64b56d9a2e0bd04c73005` |
| 完整输出清单 | [producer completion](/home/liwt/GRoPE/results/round22_ray_preflight/completion.json)；[audit completion](/home/liwt/GRoPE/results/round22_ray_independent_audit/completion.json) |
