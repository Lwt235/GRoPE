# 高斯场景记忆的 Transformer 读取：结果、修正与确认

2026-11-10。**一层小型 cross-attention 可以从保存的高斯记忆读取当前视角的粗特征；显式中心定位是目前更稳妥的读取基线。高斯形状位置项在开发场景有额外收益，但没有在新于本设计的 12 个场景上稳定超过中心项。** 本轮完成原九次配对训练、一次投影接口修正后的三次配对训练、全部开发评价和 12 场景固定模型确认。所有结果均保留，未根据确认结果再训练、排除场景或选 checkpoint。

## 1. 实际研究了哪个环节

[首轮原型](../prototype1/report.md)已经用共享写入网络建立并保存粗高斯状态。本轮冻结同一写入器和所有开发记忆，研究如何通过 Transformer 读取这些状态。读取只输入查询相机及 patch 射线，不输入查询图像、查询教师、传感器深度或参考上下文。

每份记忆为 256 个 `(μ,Σ,α,f)`，`f` 为 32 维，裸状态约 45 KiB。单层 cross-attention 宽 64、4 头；Q/K 为两层小 MLP，V 和输出为线性层。MLP 组 15,552 个可训练参数，中心和高斯组各 15,556 个。三组 K 都得到相同完整内容、中心、协方差和 opacity；只有额外关系项不同。V 只由相同记忆特征产生，没有 query 残差旁路。

最终高斯读取的分数为：

\[
s_{hij}=q_{hi}^{\mathsf T}k_{hj}/\sqrt{16}+\log\alpha_j
-\frac{\lambda_h}{2}(u_i-\pi(\mu_j))^{\mathsf T}S_j^{-1}(u_i-\pi(\mu_j)).
\]

`S` 使用写入阶段 gsplat 的透视协方差投影，再加固定 `0.1I`；每头一个正系数 λ，初值 1。中心组用固定尺度的欧氏距离替代二次型，尺度是训练查询几何的投影方差中位数 `0.7146816254`。普通组没有额外项。所有组共用前方深度范围、log opacity 和零值背景槽。没有新增遮挡模块、全模型微调或逐场景拟合。这是几何 attention bias；本轮不声称 RoPE、频率编码或结构新颖性。

**此时 Query 仍是相机/patch 几何，不是完整 VGGT 的实际视图 token。** 这一步验证记忆读取机制，尚未完成对 VGGT 主网络的接入。

## 2. 数据与公平比较

开发数据复用首轮 3 个训练场景、12 个片段：72 个训练查询、36 个同场景留出查询；另有 6 个未参与优化场景、54 个查询。全部已用于历史研究。三组各三个种子 `20261110/11/12`，配对初始化和采样，AdamW `3e-4`、weight decay `.01`，各 2,000 步，固定最后一步。损失为标准化 32 维教师特征 MSE。

确认选择本地 R6/R7 数据的全部 12 个可用场景，每场景按原始序号取首 13 个已下载帧，固定参考 `0,4,8,12`，其余 9 帧为查询，共 108 个。没有新下载、逐场景优化或重拟合 PCA。原写入器从每个新场景四张参考图建立状态；同一状态供九视角读取。查询图像仅独立计算教师目标，不进入参考 VGGT 聚合和写入。

这 12 个场景未进入当前写入/读取设计，但曾用于更早研究；只能称新于本设计的场景确认。VGGT 预训练是否包含这些场景未作判定。全部 12 场景均完成，没有替换或排除。

来源：[原实验计划](../../../docs/gaussian_memory_attention_plan.md)、[输入与源码身份](identity.json)、[确认场景选择](confirmation/cohort.json)、[最终九模型与确认规则](confirmation/adaptation_lock.json)、[确认数据身份](confirmation/data_manifest.json)。

## 3. 原始负结果与投影修正

初版直接 Jacobian 投影的 Gaussian，在同场景留出视角 MSE 为 **0.40463**，优于 Center 的 **0.42390**；但六场景为 **0.49219**，差于 Center 的 **0.41054**，三个种子均如此。酒店、studyroom、实验室是主要失败场景。圆形替换仅部分减轻错误。

一次事前记录的零训练可见性诊断没有挽救它：六场景加正向遮挡后 MSE 为 **0.67172**，反向深度控制为 **0.46625**。这不支持“补上正向遮挡就能解决”的解释，也不能把反向排序当作合理的新渲染方案。该操作同时存在训练后分布改变，未被加入最终模型。[诊断方案与全部结果](visibility_diagnostic1/plan.json) · [结果](visibility_diagnostic1/scores.json)

检查实际写入渲染器代码后发现：gsplat 在计算协方差 Jacobian 时限制画面外基元的视角范围，初版 attention 没有复用这个处理。直接投影会使部分画面外或近相机基元产生过大的屏幕支撑。固定权重、只替换这一投影，六场景 MSE 从 **0.49219 降为 0.39852**，三个种子均优于中心组；中心对照输出变化为 **0**。这说明初版失败包含实质性的投影接口因素，不能直接归因于高斯形状本身。[固定权重兼容性检验](projection_diagnostic1/scores.json)

随后按[单独记录的修正计划](../../../docs/gaussian_memory_projection_alignment.md)，只让高斯项在训练与读取时都使用原 gsplat 投影，再完成三次同预算训练。原六个 MLP/Center 模型复用，因为其所有输入和计算逐值不变。修正训练的初始化和采样与原训练一致，没有加入模块或扫描投影范围。[配对核对](projection_matched/paired_training_check.json)

原初版结果没有被覆盖，固定权重干预的最好数字也没有替代新训练的最终模型。下文 Gaussian 均指按一致投影重新训练的固定最后一步。

## 4. 最终结果

下表为三个读取种子的均值，MSE 越低越好。Renderer 是既有固定投影/alpha 合成参照，其读取器没有新增参数，不与三种 attention 构成完全匹配的模型比较。

| 数据范围 | 普通 MLP attention | 中心位置项 | 高斯区域项 | 既有 Renderer |
|---|---:|---:|---:|---:|
| 同场景留出视角，36 查询 | 0.63513 | 0.42390 | **0.39910** | 0.41795 |
| 六个开发场景，54 查询 | 0.61913 | 0.41054 | **0.40036** | 0.42546 |
| 新于本设计的 12 场景，108 查询 | 0.74001 | **0.53253** | 0.55451 | 0.56857 |

开发阶段，Gaussian 相对 Center 的误差分别降低 **5.85% / 2.48%**，三个种子均为同方向，六场景有四个改善。确认阶段则误差**增加 4.13%**，三个种子均为中心组更好，12 场景六胜六负。相对 MLP，Gaussian 的确认误差降低 **25.07%**，12 场景均改善；Center 对 MLP 的平均改善更大，约 **28.04%**。

确认余弦相似度为 MLP **0.49695**、Center **0.64898**、Gaussian **0.63561**、Renderer **0.61973**，方向与 MSE 一致，不能用幅值校准单独解释两种 attention 的排序。

以“对照 MSE − Gaussian MSE”为正向改善，先在每个场景内平均三个固定种子，再对 12 场景做预定的 10,000 次配对 bootstrap：

| 确认比较 | 平均改善 | 场景 bootstrap 95% 区间 | 三个种子的改善 |
|---|---:|---:|---|
| MLP − Gaussian | +0.18550 | [0.14484, 0.22590] | +0.18628 / +0.18733 / +0.18290 |
| Center − Gaussian | −0.02197 | [−0.05125, 0.00156] | −0.02270 / −0.02383 / −0.01939 |

因此，**相对中心项的稳定额外用途未得到确认**。区间跨零也意味着不能据此断言所有场景上的形状项都必然有害。该不确定性针对所选 12 场景，不覆盖写入器种子或更广分布。

### 逐确认场景

| 场景 | Center MSE | Gaussian MSE |
|---|---:|---:|
| brown_bm_1 | **0.57583** | 0.71838 |
| brown_bm_4 | 0.73262 | **0.72436** |
| brown_cogsci_1 | **0.57740** | 0.67468 |
| brown_cs_3 | 0.45319 | **0.44415** |
| harvard_robotics_lab | 0.45228 | **0.44024** |
| home_at | **0.53594** | 0.57908 |
| home_bksh | **0.54589** | 0.55448 |
| home_md | 0.58144 | **0.57479** |
| mit_32_d507 | 0.46891 | **0.46291** |
| mit_32_g725 | **0.52551** | 0.53065 |
| mit_46_6conf | **0.54715** | 0.57202 |
| mit_w59_conference | 0.39424 | **0.37833** |

[全部统计和完整场景名](analysis.json) · [确认逐查询数据](confirmation/scores.json) · [对比图](final_figures/comparison.png)

## 5. 记忆、形状与实现检查说明什么

最终 Gaussian 在六个开发场景正常读取 MSE **0.40036**；打乱特征与高斯的绑定为 **1.08056**，始终使用第一参考相机为 **0.79675**，移除记忆为 **0.93807**。结合实际保存、场景留出和多视角结果，支持读取依赖空间绑定与相机，而非只依靠固定输出模板。[最终记忆使用检查](projection_matched/memory_use_check.json)

只把显式位置项中的投影协方差替换成保迹圆形，保持完整 key 输入不变，最终 Gaussian 的 MSE 在同场景留出、六开发场景、12 确认场景分别变为 **0.44089 / 0.40704 / 0.60087**。模型确实依赖所学的形状关系；但这是训练后干预，**不等同于从头训练的球形强对照，也不能推翻 Center 在确认中更好的结果**。依赖某种信息与它比简单方案更值得采用，是两个不同命题。

原始和最终版本均通过只加载 reader 权重、保存记忆与相机的独立命令读取；九视角与评估逐值一致，最大误差 **0**。CPU 解析投影、整体坐标/基元顺序、空记忆、序列化检查与 GPU 前后向检查通过；新增画面外投影检查证明修正只改变对应支撑，共同输入不变。这些支持实现正确性，不独立证明任务收益。[预检](preflight.json) · [最终独立读取](projection_matched/standalone_check.json)

## 6. 实际查看的效果与决策

已实际查看原版本训练曲线和多视角图、三组可见性诊断图、修正后的酒店/实验室，以及确认中的两个明显退步场景和两个改善场景。酒店、实验室原来的大范围误读在投影一致后显著减轻；确认里的 brown_bm_1 和 brown_cogsci_1 仍存在区域混合、错位或信息缺失。特征图仍较粗，伪彩色不能当作 RGB 重建或准确几何。

可直接查看：[修正后酒店](final_figures/old_eval_14.png)、[修正后实验室](final_figures/old_eval_17.png)、[确认退步 brown_bm_1](final_figures/confirm_00.png)、[确认退步 brown_cogsci_1](final_figures/confirm_02.png)、[确认改善 conference](final_figures/confirm_11.png)、[确认改善 robotics lab](final_figures/confirm_04.png)。查看记录见 [review.json](review.json)。

本轮交付的是：**共享高斯记忆可以供一个精炼 Transformer 读取，显式定位明显优于本次普通 MLP 编码，但形状位置项尚无相对中心项的可靠跨场景增量。** 不将形状项晋升为默认读取方式；不在这些确认样本上继续调尺度、筛场景或增添模块。

下一阶段建议保留高斯场景记忆，以中心定位读取作为默认基线，接到 VGGT 的实际 view token，比较同预算的“无记忆/读取记忆”，验证可见观测不完整时是否获得信息补充。高斯形状项保留为有明确适用边界的对照。此阶段尚未执行。

证据仍限定于给定相机、冻结 VGGT/写入器、一次性静态局部记忆、32 维教师特征和少量场景；没有建立未知相机、增量/长期记忆、完整 VGGT 几何指标、点或二维 token 记忆对照的优势，也没有验证一般性 PE 或创新性。原写入器只有一个训练种子。

## 7. 资源与复现

GPU 1 在两次占用检查通过后单进程共享，显存上限 12 GiB，未停止其他任务。原九次加修正三次，训练循环累计约 **180.7 秒**；最大 PyTorch 分配约 **0.10 GiB**。确认 VGGT 提取峰值约 **4.66 GiB**。这些不含 CUDA 上下文、排队、完整准备与分析，不是严格性能基准。

实现：[基础读取器](../../../models/gaussian_memory_attention.py)、[一致投影接口](../../../models/gaussian_memory_projection.py)、[配对训练](../../../experiments/gaussian_memory_attention_train.py)、[修正训练](../../../experiments/gaussian_memory_attention_matched.py)、[确认提取与评估](../../../experiments/gaussian_memory_attention_confirmation.py)、[统计与作图](../../../experiments/gaussian_memory_attention_report.py)。

在 `/home/liwt/GRoPE` 中可按下列已实测方式独立读取最终高斯模型；中心基线则使用原 `gaussian_memory_attention_read` 命令与 `center_20261110/step_2000.pt`：

```bash
/home/liwt/miniColmap/.venv/bin/python -m experiments.gaussian_memory_launch \
  --gpu 1 --allow-shared --wait-for-gpu experiments.gaussian_memory_attention_projected_read \
  --checkpoint results/gaussian_memory/attention1/projection_matched/gaussian_20261110/step_2000.pt \
  --memory results/gaussian_memory/prototype1/learned_geometry/memories/val_12.pt \
  --cameras results/gaussian_memory/prototype1/evaluation/val_12_cameras.npz \
  --output /tmp/gaussian_attention_val12.npz
```

确认缓存位于用户指定数据盘的 `GRoPE/gaussian_memory/attention_confirmation_cache`，12 份保存记忆位于本目录 `confirmation/memories/`。原九次模型、诊断与失败记录全部保留；确认完成后没有待跑训练。
