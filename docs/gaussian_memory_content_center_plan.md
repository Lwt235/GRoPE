# 固定几何、学习内容时的协方差位置项控制

2026-11-10。输出 `results/gaussian_memory/content_center1`。本轮只补Content Center三个拟合，不搜索结构、损失或训练预算，不访问新数据。

## 问题、竞争解释与决策

write_confirmation1在19个新目录保留了Full Gaussian对Full Center的平均8.05%优势，但完整几何写入未稳定超过仅内容写入，完整输入空间误差仍有代价。下一项问题是：固定初始几何与opacity、只学习共享内容时，协方差位置项是否仍比中心距离更有用？竞争解释是此前协方差增量依赖几何/内容联合更新，或两种位置项的差别只影响整体偏差而未带来空间细节收益。

新增Content Center；复用Content Gaussian为主候选，并完整保留Full Gaussian、Full Center、Frozen Gaussian、Frozen Center、GlobalSummary、小Local和强Local。主比较为Content Center−Content Gaussian的整体log-MSE；主用途背景为Content Gaussian−强Local/Global的误差差。共同报告去均值空间误差，辅助报告均值偏差平方、遮挡/可见/完整输入、RMSE和AbsRel。完整/内容写入差的已知结果保留，微小或跨零差异不用于宣布等效。

若Content Gaussian相对匹配Center保留用途，且相对Full的空间代价更小，可把固定几何＋学习内容作为精炼工作基线；这不是学会构造几何。若位置项收益消失，说明旧Full比较不足以支持简化方案的协方差增量，不扩大PE主张。若强Local整体优势仍不确定，限定于有证据的缺失观测补充用途。允许有限正结果，不要求全部辅助指标获胜。新结果开放后不再训练其他候选挽救结果。

## 唯一训练差异与信息边界

三个种子20261110/11/12，各2000步，十二原训练片段、原查询/遮挡、batch=2、学习率、weight decay、梯度裁剪全部复用joint_write1。Content Center逐张量复制同种子的Content Gaussian初始reader和writer，逐步采用相同采样序列；只将reader的Gaussian Mahalanobis距离换成既有Center固定方差距离。现有center_variance不重新估计。使用完全相同的CompactGaussianWriter(geometry=False)，几何和opacity固定，仅内容残差可学；reader也训练。参数量相同，writer的650个几何输出参数在两组均不参与预测。

共同key/value与writer输入仍包含既有高斯几何信息。本轮比较协方差在attention位置项中的组织，不是“完全没有形状信息”与“有形状信息”的比较，也不是完整VGGT或通用RoPE优劣。四参考构建256槽状态，查询不进入writer，已知相机和尺度规范化不变。只有三个新模型训练，24旧模型不改变。

## 全部数据已用，分析不冒充确认

复用joint_write1的66片段及write_confirmation1的19片段，共85。主汇总为51个非训练地点，先平均同地点片段，再对地点、种子等权；同时单独报告old12、used_summary20、used_writer19，防止合并掩盖异质性。训练地点留出视角与其余已用片段仅作开发背景。原19地点本轮明确改标used_writer19，不再称为新确认；目录物理别名和VGGT预训练独立性的既有限制仍在。

差值使用原10000次配对地点×种子bootstrap，不把patch/query当独立地点，不合成新评分、不重新选模、不按结果去掉地点。三种子方向、每地点方向、全部整体及空间指标均保留。对照模型和预测按哈希复用，重新评价只针对新增模型，避免浪费GPU或复制大量旧文件。

## 有界核对、运行与图像

训练前记录数据、来源、初值和24旧模型身份；核对Content Center与Content Gaussian初始参数完全相同，2000步查询采样逐项相同，损失/梯度有限，内容梯度非零而几何梯度为零。所有85片段写入一次并保存再读，确认几何/opacity没有更新；恢复初始几何的读取差沿用2e-5容差，不要求CPU/CUDA位级相同。

独立CPU读取固定fit_00、confirm_00、fresh_19三个片段、三个新种子共九次，NumPy评分容差2e-6；同三个片段的旧Content Gaussian首种子作接口哨兵。检查原24模型和全部复用评分不变。无需重复完整旧GPU推理。普通预运行问题记录版本和错误，不升级成新科学轮次。

固定查看fit_00、transfer_00、confirm_00、confirm_15、fresh_00、fresh_19的首查询首遮挡，以及训练/分组比较总览，共七图。图像同时保留Full Gaussian、Content Gaussian、Content Center、Global和强Local，不按成功案例选图。图中深度和误差显示范围不裁切评分。

GPU启动时再次检查占用与CUDA，仅使用1号卡一个有余量的共享worker，显存上限12GiB，全部三次训练完成后再评价。最终更新研究入口、精简证据文档及完整报告，明确下一步是否值得继续，不自动叠加新模块。
