# R70：relative5 在 Transformer 图像任务中的配对开发验证

2026-11-04。承接 R69，整体目标尚未完成。此轮不把渲染器替换当作 PE
贡献，不再增加静态核门禁。实现前确定下面的比较；预检只修代码、测代价，
不按验证效果挑配置。最终源码、数据身份及运行参数在首个正式拟合前写入 lock。

## 问题与可改变的决定

假设：R69 的五个高斯相对量经单个 MLP 产生 q·r attention 项，在同信息、
同骨干下也能帮助新视角图像预测。竞争解释：匹配标签偏好这些关系；信息已经
被内容通道吸收；从 RGB 学出的几何没有稳定意义；增益来自额外参数或相机 PE。
先检验实际用途，再决定是否值得进行未使用场景确认，不能以匹配成功代替用途。

主要实验 RGB-only：rgb_raymap、rgb_prope、rgb_distance、rgb_direct、
rgb_relative5、rgb_hrope，各 seeds 11/22/33。所有角色共享 RGB/Plücker 内容、
7 参数高斯生成头及完整生成几何内容。raymap 为没有额外 attention PE 的共同
骨干；prope 使用官方相机编码；distance/direct/relative5 在同一 PRoPE 上加入
相对项；hrope 采用论文的预测式算子改编。生成的几何不是额外传感器信息。

诊断实验 RGB-D：observed_prope、observed_distance、observed_direct、
observed_relative5，同三种子。共同使用参考视图传感器深度形成的 patch 均值、
完整协方差和有效覆盖率，所有角色输入完全一致。该组只回答给定几何时的用途，
不得把它与 RGB-only 排名混合，也不称其为 RGB-only 成功或原生优化 3DGS。

本轮最多 30 次拟合，不增加学习率/结构搜索。RGB relative5 若对 Distance、
Direct、PRoPE 的 PSNR 三种子均正且均值至少 +0.10 dB，LPIPS 均值不更差，
才视为有进一步确认价值；同时完整报告 raymap/HRoPE，未超过它们不称最佳。
SSIM、逐场景与其他条件作为解释，不要求每个微小差异同号。不满足时保留失败。
RGB-D 单独满足只支持条件性用途、说明 RGB 几何学习仍待研究，不解锁最终结论。
二者都无用则停止此 NVS 接法；不得用更长训练/删困难场景自动补救。
如果所有模型明显没有学到图像任务（训练曲线、输出及无学习基线共同支持），
本轮只能说明预算/入口不足，不能据此证明 PE 无效，也不能据此宣称成功。

## 数据、输入边界与扰动

完全复用 R46 已用 SUN3D 帧，按共同 q 的 (q,q+15)、(q,q+60) 组成三帧。
288 fit /32 val /240 old_eval 三元组，场景严格继承原 3/1/5 划分。
两端为参考，中间为目标；96×128 RGB，patch=8，192 tokens/view。
只用这些旧帧，不读保留确认场景。新任务不使旧数据成为 fresh confirmation。
传感器矩仅从参考 patch 中各像素的最近原生深度样本形成，米制除以 3；
协方差加 (1/300)^2 I。零有效点 patch 使用初始化预测几何及 coverage=0，
不删 token、不把有效覆盖当目标标签。完整 RGB 监督无深度/可见性过滤。
第一参考相机为世界原点，所有平移统一除以 3m；保持与旧预测参数量级一致。
这是与原 PRoPE/HRoPE 论文单位基线规范不同的公共适配，不冒称原基准复现。
旧数据 K 为零基像素中心；传给官方射线函数时主点加 0.5，以匹配其半整数
像素约定。所有几何和图像变换一致，含主点项，不能丢掉偏心主点。

旧评估三个固定条件：base（中间目标）、extrapolate（前两帧参考，末帧目标）、
target_zoom（base 的目标原生 RGB 中心裁剪 75% 后缩放，K 同步变换）。
目标 RGB 仅作 loss/metric，目标深度不传入模型，目标相机可输入。
模型接口显式字段白名单；缓存中供其他角色使用的深度不能泄漏到 RGB-only。
固定展示每个旧评估场景第一个三元组，不挑最好看的输出。

## 紧凑接入与公平性

使用 PRoPE nvs 分支 LVSM 源码（commit 48b6dd26f1c7e906765379e4a0ef60cc50ddff7c）
的 tokenizer、encoder、图像输出和可替换 SDPA。6 blocks，192 宽，4 heads，
FF=768，q/k RMSNorm，dropout=0；这是缩小的开发模型，不是论文全量训练。
所有角色输入 stem 与共享参数同种子精确匹配。几何头由参考 RGB+ray token
预测深度及 Cholesky 形状，沿用 R62 的固定映射/界限，初始线性头为零。
生成的 μ3、对称 matrix-log Σ6、coverage1 由共同线性层加入内容。
不新增几何监督、预训练视觉骨干、GS renderer 或目标图像分支。

relative5=[||δ||/.2, log1p(δᵀΣi⁻¹δ), log1p(δᵀΣj⁻¹δ),
log1p(tr(Σi⁻¹Σj)), log1p(tr(Σj⁻¹Σi))]，MLP 5→96→192；
Distance 为相同 MLP 的 1 输入，Direct 为 δ/.2、距离、两端 matrix-log Σ 共16输入。
几何与相对向量每次 forward 计算一次，6 层复用；q·r 仅作用参考—参考 token。
目标不具备可用形状，不凭目标 RGB 构造它；参考—目标及目标—目标保留 PRoPE。
因此它测试的是高斯参考表示能否帮助合成，而非声称已解决所有目标 ray 支持问题。
额外参数和 pairwise 显存/时间如实报告，5 输入不消除二次复杂度。

HRoPE 是论文公式的本地适配，官方仓库未提供可核验完整实现。13 参数头只
从参考 token 预测；目标 A=I（论文 Algorithm 1），H=K A；q用Hᵀ，k/v用H⁻¹，
输出用H，半头为3×3块、半头为2D RoPE。使用附录 A.5 的 t/κ/分母裁剪；
与旧 R51 相同非零 κ=.25 初始值以避免 rank-one 双零梯度。
K 用与 PRoPE 相同归一化，保留主点；共同 Plücker 内容让所有角色均有相机信息。
不能把此小型改编成绩写成优于/复现完整 HRoPE 论文。

## 训练、评价与核查

拟定固定 60 epochs，batch8，36 updates/epoch，共2160updates，AdamW，
LR 4e-4→4e-6，前三轮 warmup，weight_decay=.01，clip1。纯 RGB MSE loss；
不引入感知训练损失来改变主要研究变量。bf16 matmul、fp32 geometry，关键
逆矩阵/关系收缩用 float64，梯度有限核查。初始学习率和轮数不按 val 结果调整。
若预检硬件不能承受，仅在拟合前公开修改公共 batch/累积并保留原记录。
每3轮 val，按未裁剪输出 MSE 选唯一checkpoint，全部30个选模后再评估旧场景。
输入顺序和参考顺序增强逐 seed 配对；不对图像随机裁剪造成隐藏相机差异。

主指标逐图 PSNR 后按场景均值、再对场景等权；SSIM、LPIPS-Squeeze(v0.1)
共同报告（预测clip[0,1]，全图，零裁边）。同时未clip MSE、饱和比例、验证曲线、
训练耗时、峰值显存、参数量。LPIPS仅作指标，版本及预训练权重记录 SHA。
无学习基线为最近相机参考图直接拷贝及两参考均值，用于识别未学到有效任务。
逐种子差、逐场景差和场景重采样区间完整保存；5场景区间仅探索性证据。

必要核查：真实图像/K 对齐、target 白名单、同种子共同参数、禁用关系时
PRoPE 等价、HRoPE 矩阵逆/方向、各角色前后向有限、矩阵 log 重复特征值梯度。
指标独立抽样重算及固定样例实际查看；按结论重要性检查，不追逐无影响尾数。
首跑错误用 pre_run_fix 记录，不单列新科学轮次。原 R62/R64 RGB 失败、R65–R67
原生分类不稳定和本轮所有负结果保留，最终 fresh confirmation 仍是独立阶段。

来源：[官方 LVSM 接口](https://github.com/liruilong940607/prope/blob/48b6dd26f1c7e906765379e4a0ef60cc50ddff7c/nvs/lvsm.py)，
[HRoPE 正文及 Algorithm 1/A.5](https://arxiv.org/html/2610.28284v1)。

预检完成：4 项 CPU 单测、10 角色 GPU 梯度/显存检查通过，固定预算可承受。
RGB/observed 禁用额外关系时与 PRoPE 对照逐值相同；移除共同几何内容后，
float32 下与官方 LVSM forward 逐值相同。官方在 autocast 内乘相机矩阵，
本适配固定用 float32，故 bf16 整体输出不要求逐位复现（初始最大差 .015625）。
精度处理对所有本轮角色一致，记录在 pre_run_fix2，不调整科学门槛。
matrix-log 的预检修正对称输入处理，记录 pre_run_fix1；未进行正式训练。
控制器最多并行 3 个 GPU 任务，每次分配须连续两次确认无计算进程、显存<16MiB、
利用率0；不触碰他人作业。预检首个角色包含 CUDA 冷启动，不能比较其计时排名。
