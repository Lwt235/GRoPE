# Current research state

最新调度更新，2026-11-04：用户明确允许在GPU1显存/算力有余量时共享执行。
此前“必须整卡无人使用”的调度限制已由这次用户指示覆盖。已保留25次完成
拟合，原正在执行的observed_direct/11自然完成后才交接；未中断任何训练，
未操作其他用户进程。新的experiments/round70_gpu1_shared.py只在物理GPU1
串行运行，一个自有GPU任务，启动前至少12288MiB空闲显存；不要求利用率0。
当前控制器PID102818、启动tick1406386491、exec会话29990，已启动第26次
observed_relative5/22。源码和全部冻结配置保持原SHA，CPU核查保留25次完整
run/checkpoint身份。交接记录及旧源码在results/round70_nvs/attempt1/execution_shared_gpu1。
旧控制器12348、后处理67335、诊断等待器100417均已按计划终止（退出143），
不能再重启它们。新控制器集成剩余训练、选模、CPU几何诊断、三种子预测导出、
无学习基线、汇总及指标复核。实际图表/科学审阅仍须随后完成，整体目标未达成。
显存余量及利用率每5秒记录；共享执行下的训练耗时受共驻任务影响，不据此
直接进行方法速度排名。下文旧调度/阻塞记录为历史，当前进程以本段及实时状态为准。

当前执行，2026-11-04：R70 已完成实现、数据准备与预检，开始固定预算
Transformer NVS 用途验证。计划 docs/round70_nvs_plan.md，产物
results/round70_nvs/attempt1。RGB-only 6角色、同参考深度诊断4角色，各
3种子，共30次拟合，每次60epochs/2160updates；骨干为官方 PRoPE nvs
LVSM 的缩小适配（6层/192宽/4头），relative5只加到参考—参考 attention。
所有角色共享完整几何内容和骨干；RGB角色共同从参考RGB预测几何，不接入
目标图像/目标深度。RGB-D只单独解释，不用于宣称 RGB-only 成功。
288fit/32val/240old_eval三元组已缓存，全部来自旧3/1/5场景，保留场景未读。
验证后统一选模，再评估 base/extrapolate/target_zoom；不得提前使用旧评估挑选模型。
4项CPU测试、10角色GPU前后向、禁用关系与官方接口代数检查通过；候选峰值
预检显存约4GB，固定预算可行。预检修复/精度记录见pre_run_fix1/2/3，
无科学配置搜索。控制器最多3卡，仅分配连续两次确认没有计算进程的空闲卡。
本轮尚无用途结论，完整目标ACTIVE未完成。勿把预检通过当作效果验证完成。
冻结SHA a4dee280476efaaba2cd57a3ddf0db3563d9ad059a9a5007a47073458c8d5fd3。
控制器PID12348、exec会话45903，运行身份见execution_identity.json；
初始分配GPU1/2/7，每次后续分配仍实时检查。不要重复启动队列。
后处理等待器PID67335、exec会话36731（experiments/round70_nvs_finish.py）
等待训练和预测导出全完成，然后自动计算无学习基线、生成图表、复核指标。
finish_status.json和monitor_status.json可读状态；不要重复启动等待器。
训练初期各角色loss/梯度有限，RGB relative5已进入正式训练；已完成对照的
旧val MSE约.028–.032，显著低于复制参考图.114/双参考平均.098，仅说明
任务确实有学习，不用于挑新配置或预判old_eval编码排名。LPIPS/SSIM基本
可用性检查通过。完整结论须等全部30选模和90条件评估、图表实际审阅。
R70运行时补查URoPE正文与官方源码，记录docs/round70_related_work_update.md。
该文献提醒区分参考区域内部关系与目标查询读取参考区域的几何关系；后者
尚未被R70的参考—参考高斯项直接检验。仅记录解释边界和后续可检验问题，
不改变当前30次预算/判据，不抢先宣布新接法有效或抹去既有负结果。
另已预先记录一次CPU只读几何解释检查：docs/round70_geometry_diagnostic_plan.md。
六个RGB角色三种子全部选模后，只用旧val32三元组比较生成μ/Σ与参考区域
传感器矩；不添加拟合、不选几何更好的模型、不把物理一致性作为新门禁。
等待进程exec会话25545，PID及启动身份追加在execution_identity.json；不要重启。
结果计划写入geometry_diagnostic1。正式NVS仍按既定画质/成本比较完成。
用户恢复目标后，原队列利用释放的GPU1完成observed_prope/11和
observed_distance/22，最新为20/30拟合正常完成，每次2160步、loss/梯度有限。
共享GPU随后再次均被外部任务占用，主控制器12348正常存活并等待空闲卡，尚有
10次待执行；不得因暂时无活动子进程而重启。后处理67335及CPU诊断100417
也均已核实存活，等待各自依赖。详见monitor_status.json的时间戳和占用快照。
这是资源等待而非训练失败，自动队列将继续分配两次核实空闲的显卡。
上次阻塞记录（本次恢复不沿用其连续回合计数）：GPU不可用曾连续三个目标回合出现；当时主机级
核查显示八卡均有外部计算任务、利用率96–100%，自身无训练子进程运行。
三个既有控制/等待进程的 PID 与启动身份均匹配，已完成18次、待执行12次。
后续完整比较和诊断依赖剩余选模，当前无可独立完成的剩余验证；目标的
阻塞条件为外部 GPU 资源释放。保留现有自动队列，不终止、不重复启动，
不改训练精度/预算来绕过资源等待。资源释放后队列会继续执行；完整目标
仍未达成，结果产生后仍须实际审阅并据证据决定下一步。
恢复后还完成一次不读数据、不训练的小型CPU代数/成本检查，代码
experiments/round70_relation_cost_probe.py，产物cost_diagnostic1。由于关系
MLP末层无bias，q_h·(W_h hidden_ij)可改写为(W_h^T q_h)·hidden_ij。
float64前向/梯度一致至浮点误差；bf16存在舍入顺序差异，未替换任何冻结路径。
本轮形状下可避免显式432MiB关系张量，保留216MiB成对隐藏量；理论MAC
统计见interpretation.json。这只是未来实现选项，不是实测GPU峰值/速度收益，
也不改变本轮模型、预算、预定比较或用途判据。此次恢复后，GPU不可用条件
再次连续出现在三个目标回合；最新主机级核查八卡均有外部计算进程，自身
20次完成、10次待执行，三个既有控制/等待进程身份均匹配。当前后续验证
依赖外部GPU释放，目标按blocked记录资源阻塞；后台队列保持自动续跑，
完整研究目标仍未完成，尚不能解释本轮用途结果。

最新决定，2026-11-04：R66诊断、R67配对续训、R68固定查询匹配、R69输入
精简均已完成并实际审阅图表。保留R69 relative5作为下一阶段开发候选，
全局目标仍ACTIVE且未达成；全部本批训练/评估控制器正常结束，无需重启。
候选使用[r,log1p(delta^T Sigma_i^-1 delta),log1p(delta^T Sigma_j^-1 delta),
log1p(tr(Sigma_i^-1 Sigma_j)),log1p(tr(Sigma_j^-1 Sigma_i))]，r=norm(delta)/.2，
经单个5→96→96 MLP形成q·r相对位置项。它是高斯参数到相对PE的精简组织，
不是新RoPE/新距离定理；全网和pairwise计算复杂度没有因此大幅缩小。

R69基础/固定查询稀疏区域R@1为40.628/37.576%，对Distance增益
+3.958/+4.703pp，对Direct增益+4.986/+4.806pp，三个种子和五个场景均
正向；三个种子的中心PCK、Soft EPE和CE也改善。相对full11的召回均值
约持平，但两种子较低，Soft EPE均值更差，不能宣称全面优于或严格等效。
metric3和spectrum7均损失部分效果，停止进一步压缩/扫描，采用5维版。
该候选尚未测原生优化3DGS分类、RGB/NVS或新数据，不能用匹配结果覆盖
此前负结果。下一阶段应检验Transformer图像任务用途，而非再加静态核门禁。

当前执行，2026-11-04：用户已恢复目标。R66分类诊断及24组GPU前向核查完成，
科学/图表审阅完成；见results/round66_logshape/classification_diagnostic1/report.md。
CE与准确率反向不是仅由类别权重造成：三种子对变错比错变对多4/6/7个；
CE收益来自分类对错未变化样本的概率变化。fit/val方向随种子不同，不能统一
归因于优化失败或过拟合。原R66失败结论保持。
R67按docs/round67_training_path_plan.md完成唯一的两组×三种子、各20轮
配对续训诊断。控制器87470（exec会话36986）及12个子任务退出0，勿重启。
延后形状78.642%mCA、距离续训78.580%，差+0.062pp，种子−0.222/+0.685/
−0.278，区间[−0.241,+0.370]；近似持平，没有稳定实用收益。结束这条
诊断，不追加训练或学习率搜索，也不据此排除所有优化路径。两组均选20/5/15
附加轮次。36组独立指标最大差6.66e-16，置换预测不变；图表已审阅。
报告results/round67_training_path/attempt1/report.md，review SHA
6f1fe49da511a2da0d1e0feaf8ec0d6720003ffaa460d3901ca5a54cf57b3e28。
冻结SHA 6a3846288b9a8560e012d9060d65b3b085e0cbbd06d8e3bbbef97275c9ae8353。
预检首轮发现非连续输入列导致1.67e-6浮点差异、预测未变；已在pre_run_fix1
保留失败及原源码，修正内存布局后初始函数和禁用形状时的一步更新差均为0。
非零形状与R66原公式差9.54e-7、梯度有效，原Adam步46801、新参数步1；
没有改变公式、实验预算或通过门槛。该诊断完成，整体目标尚未达成。

R68机制实验及科学/图表复核已完成，控制器30371（exec会话3420）退出0，
21个子任务全部退出0，勿重启。计划：
docs/round68_fixed_region_plan.md。冻结SHA
a2d20da1d09b24d5a4dddeb2862390e42ee85c4c34f69e7fd3e1f52701234701。
执行队列限制自身并发为2，R67启动GPU任务期间暂停新的分配；各次启动
仍须两次实时空闲观察。执行顺序调整已在execution_revision1保留原版，
仅queue函数改变，模型、数据、指标与预算不变；两项固定比较独立报告。
数据与核查位于results/round68_fixed_region/attempt1。沿用旧SUN3D的576fit、
64val、480old_eval视图对，四个输入几何分组阈值仅由fit确定。基础及固定区域
稀疏条件均保留20422/61440个有效查询（33.2389%），标签、世界参考中心和
分组完全相同；四组有效数为4998/5305/5057/5062。稀疏每区域保留6–29个
观测，中位数16。这是给定原区域划分的观测稀疏化，不等于从稀疏点云重建
区域的完整流程。共同内容包含原11维描述符、中心3维与完整log协方差6维；
旧R59–R61共同内容只有谱/径向描述符，不能误称已经做过同样的完整信息比较。
三个数据/结构单测及两个分组指标单测已通过；实际样例独立稀疏矩误差1.19e-7，
五角色置换logit差不超过4.77e-6、预测不变。保留数据仍未使用。
五角色各3种子、60epochs/1080updates，共15次拟合与30组评估完成。
区域R@1基础/稀疏：Gaussian40.436/37.351%、Distance36.670/32.873%、
Direct35.642/32.770%、Normal36.816/33.976%、Shuffle35.138/32.192%。
高斯相对距离+3.767/+4.477pp、相对Direct+4.795/+4.581pp；区域差在
三个种子、五个场景均为正，中心PCK/SoftEPE/CE也均三个种子改善。
所有预定义分组都改善，但未确证收益集中在高歧义/高形状差异组。
固定查询后所有方法的绝对召回均因稀疏下降；Gaussian的降幅并非小于
所有对照。保持领先不等于对每个对照都更耐稀疏。不能唯一归因于三轴性。
全量微平均独立指标最大差4.44e-16，另按逐对逐方向计数重算全部场景及
分组的区域/中心指标，误差和计数差均为0。图已查看，三种子和60轮输入流
与共同初始化配对。完整报告results/round68_fixed_region/attempt1/report.md；
review SHA 69e64f50c51770d90d2a039bad8c2ac6ed8606d9d9aded7c53c3f5f1600469a1。
这是更可信的同信息开发收益，不是原生3DGS或新数据确认；目标仍未达成。

R69完成一次有边界的输入精简：docs/round69_compact_relation_plan.md。
三类×三种子：relative5去绝对谱、metric3仅距离与双侧马氏项、spectrum7仅
距离与两端谱。完整复用R68数据/对照、骨干、初始化和实际输入流；最多9次。
先查能否将11输入缩为3–5而保留收益；若不能则保留full11，不继续扫描。
控制器6305（exec会话99890）及15个子任务全部退出0，勿重启；冻结SHA
bcd61fb7e4184455ba44cf98cdd65b5a6121630d1742cb4c9dc4eb89193a51fc。
源码列对应及全部共同参数配对两项单测、GPU预检通过；9次训练均完成。
共同初始化及60轮实际增强流与R68精确配对。18组新评估＋30组身份核实的
复用评估；新指标独立复算最大差8.88e-16，另重算全部场景/分组的主指标及
中心指标，误差和计数差均为0。图已实际查看，报告及review位于
results/round69_compact_relation/attempt1，review SHA
d493885be2051359e5c53f4af6ae288679e6d4be5a831358de8d3fcc9432fa2d。
该精简不成为实际图像任务的开放式前置门槛；R68已足以支持下一项用途验证。

并行核实下一阶段入口（仅可行性阅读，不改变R67/R68）：PRoPE官方仓库
的nvs分支提供LVSM及可替换attention接口，README与源码已读取；入口为
https://github.com/liruilong940607/prope/blob/nvs/nvs/lvsm.py
（Git blob 0b74c5f543895bf9264a9165b3937d5a90fff6f9）。参考图像和全部相机
进入模型，目标图像不进入前向；后续任何高斯几何生成也必须保持这一点。
HRoPE官方仓库当前README仍为Coming soon，不能把本仓库的标定平面
对照冒称官方完整实现或论文最佳预测版本。当前匹配实验也未独立隔离
所有不变性因素，完整高斯与法向/尺度的差异只能支持对应组合的比较。

R69完成时提出、现已由上方R70启动的下一步：将已选relative5接入Transformer新视角合成的
配对用途验证，不再开启分类续训或更多关系精简。现有旧SUN3D已选帧可组成
288fit/32val/240old_eval三视图样本（q,q+15,q+60），已按元数据检查全部
RGB文件存在，未读取保留场景；两端作参考，中间作目标可作初步任务。
这只有3/1/5场景，足以开发试验，不冒称大规模NVS基准复现。
先固定相同几何来源/内容/骨干，区分RGB-D参考几何的机制迁移和RGB-only
参数预测用途，不混为公平输入比较。目标RGB/深度仅可作损失或评价，不能
进入前向或生成位置编码；目标相机可以输入。保留Distance、同信息MLP、
框架原编码和适用的PRoPE/HRoPE对照，测PSNR/SSIM/LPIPS及代价；真实
图像输出是Transformer PE的用途检验，不是更换高斯渲染器后的画质比较。
实施前确定输入生成与attention接口、固定预算和对应决定；模型/原始数据
不因本次可行性阅读而改变。不要把这一后续设计状态误报为已启动或成功。

以下保留R66完整开发验证及实际图表审阅的结论，2026-11-04。
中心距离加一个log-shape距离的精炼方案未建立相对强MLP的稳定收益；
按原计划停止这版及其谱对照。用户于2026-11-04恢复研究，全局目标为ACTIVE，
尚未得到可推广的最终方案。
R65完整验证、固定选模诊断及更早几何／迁移／RGB负结果全部保留。

全部7757个官方TRAIN对象已下载验收；本轮复用完整60类、两个来源、
6227fit/765val/765dev_eval，六次新训练均完成120epochs/46800steps。
两个新角色各3种子，36组新预测与原R65五角色的90组CE-selected预测配对，
共126组。预检、六次训练、选模、三个导出和报告共12个子任务均退出0。
原控制器28606及所有子任务正常结束，只读观察器45843也退出0；勿重启。
官方TEST及保留确认场景未读取，这仍是用过数据上的开发验证。

| 方法 | 基础mCA（三种子均值） |
| --- | ---: |
| R66 log-shape距离 | 77.799% |
| R66谱距离对照 | 77.226% |
| R65高斯关系PE | 77.092% |
| Direct MLP | 76.776% |
| Distance MLP | 78.426% |
| Center RoPE | 71.163% |
| NoPE | 69.829% |

主候选减Distance为-0.627pp，配对种子为-0.643/-0.621/-0.617，类别区间
[-2.304,+1.046]pp。主候选减Direct为+1.024pp，种子+2.725/+1.720/-1.374，
区间[-0.641,+2.671]。两种新角色在六条件下相对两个强MLP的24项比较均未
满足原定稳定收益标准。这支持停止当前版本，不证明形状PE普遍无效或普遍更差。

完整形状减谱对照的基础差为+0.573pp，种子+0.461/+3.397/-2.138，区间跨零。
稀疏256下均值为+2.293pp、类别区间[+0.476,+4.146]，但seed33为-1.299pp；
不能据此宣称稳定的方向信息收益。主候选相对Distance在稀疏256的均值仅
+0.161pp且种子分歧，稠密1024为-1.463pp且三个种子均为负；jitter为
-0.903pp且三个均负，tilt为-0.947pp且种子分歧。完整固定比较见报告。

需保留的数据依赖现象：基础条件主候选减Distance在MACGS为-2.396pp，
三个种子均负；在ModelNet为+1.143pp，三个种子均正。两个来源类别不重叠，
这是同一开发研究内的描述性差异，不是跨来源迁移、成因证明或选择有利子组
作为成功结果的依据。全部60类仍纳入，所有类别和来源指标保存在result.json。

主候选基础CE为0.72196，低于Direct的0.75136和Distance的0.73671，三个
种子方向一致；R65高斯关系的CE0.70246仍更低，三个种子也一致。这一概率
评分现象保留为次要结果，不能替代原定mCA结论，也没有证明校准改善。

结构与范围：R66使用同一个2→96→96 MLP处理
`r=||mu_i-mu_j||/0.2`和
`s=log1p(||0.5log(Sigma_i/0.04)-0.5log(Sigma_j/0.04)||F²)`，
谱对照用排序后的三个log特征值替代完整矩阵。PE有9504参数，比距离基线
多96个；共同13维内容仍包含完整协方差。新增列从零开始，初始函数、共同
参数、距离MLP权重及实际120轮输入流均与原距离基线配对。这检验额外相对
PE的作用，不等于“协方差信息整体无用”。它使用已知Log-Euclidean距离，
不是新距离定理或新RoPE；与R3周期形状相位、R40/R41非正形状偏置及R65
十一维含逆协方差关系不同，但这些区别本身不证明新颖性或效果。

解析解释仍保留：对相同轴对称谱Sigma=t²I+(s²-t²)nnᵀ，形状输入为
`log1p(2 log²(s/t) [1-(n_i·n_j)²])`，即各向异性加权的无向法向夹角关系；
谱对照此时为零，球形时方向项消失。12个解析样例最大差2.24e-8，无需修补。
这不是一般异谱／三轴公式或任务收益证据；R61的条件性法向收益及迁移失败保留。

实现与审阅：五项预检测试通过，GPU初始函数对照最大差2.38e-7。
两角色在三个种子上均按验证CE选出95/100/120轮；GPU选模重放误差0，
CPU每模型两对象抽查最大5.72e-6、预测不变。独立形状公式最大差1.04e-7；
独立指标重算最大差4.44e-16，六个模型置换均未改变预测，最大logit差6.80e-6。
这些误差无需数值修补。图表已实际查看，科学/图表审阅标志均已清除；机器
原报告及结果存于review_before1，审阅没有更改源码、检查点或预测。
复核使用独立公式和指标，但由同一代理执行，并非另一位研究者的独立审计。

冻结SHA d64ca1ff3e3d403d8b746d4b5213749148ee604f86320c1a46056b5866cc8d69。
审阅SHA 1b34999cdfb6184d7757587f6f9fa5883636b8f9bb00effa71efbff3efe8aa53。
[固定协议](/home/liwt/GRoPE/ROUND66_LOGSHAPE_PROTOCOL.md)，
[完整报告](/home/liwt/GRoPE/results/round66_logshape/attempt1/report.md)，
[审阅记录](/home/liwt/GRoPE/results/round66_logshape/attempt1/review.json)。

决定：结束R66，不通过搜索shape尺度、检查点、额外训练或有利子组挽救它。
恢复后应先明确一个不同的、与任务相关的高斯参数→PE假设，以及相对中心
距离和共同内容的新增作用，再冻结新的精炼设计与比较。来源差异值得解释，
但尚未定位病因；不能直接据此推出新模块或宣称高斯形状没有价值。
未使用数据确认继续保留，完整全局目标尚未达成。

## 2026-11-04 对话同步后的实验计划（已授权执行）

已读取指定分支[检索3DGS与RoPE结合进展 (2)](codex://threads/01a1a49f-9378-7d23-b063-47b033c19575?hostId=remote-ssh-discovered%3AGPU)
最新讨论，并读取其引用的[表达能力与实际训练分支](codex://threads/01a1a4bd-7411-7a40-bf66-3f0969f67b2c?hostId=remote-ssh-discovered%3AGPU)。
用户已恢复目标并授权自主实验。以下计划从已有分类结果诊断开始执行，
不改写R65/R66的既有结论。GPU 0–3初查空闲且CUDA可用，4–7为他人任务。
本阶段记录见docs/round66_classification_diagnostic_plan.md及
results/round66_logshape/classification_diagnostic1；尚未据此决定启动续训。

1. **先诊断已有分类结果。** 主比较为原CE-selected的R66 log-shape与
   R65 distance_mlp，三个配对种子、全部60类和两个来源。区分优化路径、
   泛化差异，以及概率评分/类别权重/分类决策差异三类解释。
   同轮次比较各自训练曲线和验证曲线；复用保存的val与dev_eval预测，
   计算样本平均CE、类别平均CE、OA、mCA，按双方均对、均错、对变错、
   错变对分解概率损失和准确率变化，并保留逐类/来源结果。
   原训练loss含增强及0.1标签平滑，验证CE不含；不能直接相减当泛化间隙。
   若需要判断拟合差距，固定fit样本与token采样，补同一eval模式下的
   无增强、无平滑前向评价。诊断不重选检查点，不改变mCA主结论。

2. **仅在训练路径仍是影响下一步判断的关键疑问时，做一次配对续训。**
   问题是“先学会中心关系，再引入形状，是否带来额外收益”。拟议上限
   为两组×种子11/22/33，各额外20epochs；不展开超参数搜索。
   两组从同种子的距离模型第120轮resume.pt出发：原距离模型续训，
   与函数等价嵌入、形状权重为零的R66扩展续训。best.pt仅有模型权重，
   不得把它与第120轮优化器状态拼接。共同参数/优化器历史、输入流、
   预算、学习率方案及验证CE选模规则配对；新增形状参数状态单独初始化。
   原首层两列同属一个Parameter，若需独立Adam步数，可作代数等价的
   W_r*r + W_s*s + b分解，并验证起始函数及状态映射，保持原表示能力。
   执行前固定该短续训配置；单独记录为训练路径诊断，保留原结果。
   若有效，只支持该训练路径下的开发收益；若CE仍改善而mCA下降，
   不宣布分类成功；若无增益则结束此诊断，不将解释所有失败作为后续门禁。

3. **跨视角区域匹配验证机制。** 沿用已使用的SUN3D开发划分，先使用
   深度观测区域；固定物理查询与原始表面/位姿标签，采样变化不重定义答案。
   拟用已有R59–R61高斯关系公式作为探针，最多五组×三个种子：距离PE、
   同信息通用几何MLP、距离加法向/尺度、高斯关系PE、训练与评估均按
   固定规则打乱PE形状对应的同结构对照。共同内容保留相同完整形状，
   骨干、共同初始化、输入流、训练预算和选模规则配对；记录容量/计算成本。
   按中心布局歧义与形状差异预定义四组，阈值只用fit输入确定；报告全部
   分组及整体的区域匹配、中心定位、覆盖率和种子/场景差异。
   这用于区分“额外关系有用”“普通MLP已足够”“法向/尺度已足够”与
   “高斯组织方式有额外收益”；单独打乱后下降不构成效用证据。
   分组是机制关联证据，不单独证明因果；观测区域也不等于原生3DGS。

4. **出现可信机制信号后验证用途与确认。** 选一个精炼设计，接入成熟
   Transformer新视角合成框架；固定图像输入、几何来源和主体网络，
   比较同信息MLP、简化对照、框架原编码及适用的PRoPE/HRoPE。
   输入几何只来自参考视图，报告常规及预定义视角/内参变化条件下的
   PSNR/SSIM/LPIPS与成本。匹配收益不能自动证明图像收益。
   原生3DGS价值需真实高斯输入证据，跨重建主张需独立重建数据；
   分类负结果继续保留。公式、预算和主指标固定后才使用未用数据确认。

本计划的实现检查针对标签、信息泄漏、对照配对和影响任务的数值错误。
不恢复历史静态积分通用门禁。后续GPU前向或训练启动前检查CUDA及实时占用。

以下是已完成的R65科学状态及更早执行轨迹。

最新科学状态，2026-11-04：完整 R65 验证与实际图表审阅已经完成。
当前 11 维高斯关系 PE 在完整数据上仍未建立对强 MLP 对照的稳定收益，
不能作为已通过完整目标的最终方案。全局研究目标保持 ACTIVE、尚未达成。

全部 7757 个官方 TRAIN 对象已下载验收，使用全部 60 类和两个来源，
按冻结划分得到 6227 fit / 765 val / 765 dev_eval。15 次训练均完成
120 epochs / 46800 steps，90 组评估全部完成；原队列 21 个子任务均退出 0。
控制器 49698、报告进程 49795、所有训练及导出进程已正常结束，勿重新启动。
官方 TEST 与保留确认场景没有读取。这是完整训练数据上的开发验证。

| 方法 | 基础 mCA（三种子均值） |
| --- | ---: |
| 高斯关系 PE | 77.092% |
| Direct MLP | 76.776% |
| Distance MLP | 78.426% |
| Center RoPE | 71.163% |
| NoPE | 69.829% |

高斯减 Direct MLP 为 +0.316 pp，三个配对种子为 +3.412 / -1.187 / -1.277，
配对类别区间为 [-2.417, +2.639] pp。高斯减 Distance MLP 为 -1.334 pp，
种子差为 +0.044 / -3.528 / -0.519，区间为 [-3.519, +0.663] pp。
这支持“没有建立稳定提升”，不支持“已证明普遍劣于 MLP”。对 Center RoPE
和 NoPE 的均值收益分别为 +5.929、+7.263 pp，三个种子均为正。
所有模型的共同 13 维内容都包含协方差；结论针对额外关系 PE 的收益，
不能解释为形状信息普遍无用。

稀疏256、稠密1024和tilt条件下，相对 Distance MLP 的均值差分别为
-1.917、-1.038、-1.473 pp，三个种子均为负；所有固定条件均未建立对两个
强 MLP 的稳定优势。每个来源内也未建立稳定强对照收益：基础条件相对
Distance MLP 的 MACGS / ModelNet 均值差为 -1.848 / -0.821 pp。
来源误差分解显示，合并任务的 -1.334 pp 中，-1.273 pp 来自同来源内错误，
-0.062 pp 来自跨来源错误；提供真实来源并限制分类标签后仍为 -1.520 pp。
这些是描述性诊断，不能作为跨来源迁移或来源捷径的因果证明。

保留一个次要现象：高斯的基础 CE 为 0.70246，低于 Direct MLP 的 0.75136
和 Distance MLP 的 0.73671，三个种子方向一致。它不能替代原定 mCA 结论，
也没有直接证明校准更好。验证曲线后段趋缓；高斯按 CE 选出的 epochs 为
100/90/90。分类与 CE 的差异、选模策略与几何表示的作用尚需区分。

实现及审阅记录：15 个选定模型的 GPU 验证重放误差为 0；独立指标重算
最大差为 4.44e-16。所有15模型的置换均未改变分类预测，最大 logit 差
6.26e-6，无需数值修补。四张图均已实际查看，科学/图表审阅标志已清除。
共享主机耗时不是受控效率基准。每类样本数有限、非同文件的物理实例分组
未知，类别 bootstrap 不包含独立训练种子不确定性。R61 迁移与 R62/R64 RGB
负结果、内容冗余解释未成立、解析编码碰撞和形状下限诊断的范围限制全部保留。

固定选模诊断 checkpoint_policy1 已完成并审阅：15 个最终120轮模型先
重放原验证指标，再在三个空闲 GPU 上导出全部90组新预测；没有新增训练。
计划 SHA 5d0ba6f1a20026bf1821c2dc49670098a1b71a5e01b8b4a562ebb40a8f714f08；
源码 experiments/round65_checkpoint_diagnostic.py，三个有明确预期值的测试
通过38aa04。seal、三个export和report共五个进程均退出0，勿重新启动。
第120轮验证指标最大重算差4.44e-16；三项原本就选在120轮的模型在全部
18组条件中逐logit完全一致。所有15模型的置换仍未改变预测。

final120基础mCA：高斯77.993%、Direct MLP77.176%、Distance MLP78.733%。
高斯相对自身原选模的均值提高0.901pp，但三个种子为+0.152/+2.973/-0.420；
类区间跨零。相对Direct的final差为+0.818pp，种子+2.251/+2.445/-2.244，
区间[-1.042,+2.582]；相对Distance为-0.739pp，种子-0.996/-0.283/-0.939，
区间[-2.624,+1.104]。相对两个强对照的差距变化均为(-,+,-)，对应区间也
跨零；全部六种条件仍未建立稳定强对照收益。最终高斯在dev_eval上的CE
也比原选模低，故不能由此诊断“CE与准确率存在固有冲突”或“校准更好”。
本次只比较final120与CE-selected，并没有测试按验证mCA选模。

决定：结束这一处选模诊断，保留原方案与final120的负强对照结论，停止
继续寻找有利checkpoint或混合各方法的选模规则。下一阶段回到精炼的
高斯参数→位置算子设计：结合既有中心/形状解耦证据，先明确新表示与
R3、R44–R57等已试结构的区别及其任务相关机制，再固定一个候选和强对照。
不能把此次均值改善、解析编码碰撞或微小数值误差当成已确认的任务病因。
官方TEST及保留场景继续保持未使用；全局目标尚未达成。
[选模诊断报告](/home/liwt/GRoPE/results/round65_actual_gs/attempt1/checkpoint_policy1/report.md)
与实际审阅 SHA 9b4913e147d868feb4977baa4b1a30290fbfb085d2d7bfdd3f37dc3aa3f1bae0
记录全部方法、种子、条件、差距变化与适用范围。

完整科学审阅 SHA a6c34466a94d2817e8aafd2da53053945c923ac7b4a7171cd8634db655a97e11；
来源诊断审阅 SHA a444195c599c4f3e6485233c86792dad852223a3451a21002e2de719809b4704。
[完整比较报告](/home/liwt/GRoPE/results/round65_actual_gs/attempt1/report1/report.md)，
[科学审阅](/home/liwt/GRoPE/results/round65_actual_gs/attempt1/review.json)，
[来源诊断](/home/liwt/GRoPE/results/round65_actual_gs/attempt1/source_diagnostic1/report.md)。
原机器生成状态保存在 attempt1/review_before1；模型、源码、配置、预测与指标
未因审阅改变。以下 RUNNING、pending 和 PID 记录均为历史执行轨迹，
以本节的已完成状态为准。

Earlier execution,2026-11-04:the complete60-class validation was RUNNING.
Both user conditions were satisfied:content_route1 finished and was interpreted,
and all7757TRAIN downloads were accepted(116419005518bytes;MACGS3246,ModelNet4511).
Original mirror downloader99977 ended successfully. Acceptance record:
preparation/acquisition_1793682705671116425/acceptance_20261104.json,sha256
dbe22d81ce4badd98e0830ca9245eafa4b5af7358ba4658efb9ab98759c0b820.
Complete unique path/size coverage and pinned source/completion identities were
checked; unchanged downloader code already verifies every file's published SHA256.

New full controller49698,start_ticks1400448946,was launched atc7aec1 using the
unchanged experiments.round65_queue. Reporter49795,start_ticks1400448957,
automatically follows it using experiments.round65_report_watch. Launch record:
attempt1/full_validation_launch1.json. Preprocessing has now COMPLETED:child49746
exited0; all6227fit/765val/765dev_eval objects and all60classes in every split
are present,no exclusions or conversion failure. The earlier wait for GPUs
ended automatically. Checkdabfba verifies original controller49698 live with
matching start_ticks. All15fits have now COMPLETED normally:allfive roles and
allthree seeds. No training job remains pending or active;do not restart the
completed Gaussian22 PID115654,Gaussian33 PID114493,directMLP33 PID116394,or any
earlier fit. Sixteen children(data and fifteen fits)exited0. The original queue
has entered selections:child19041,start_ticks1402574668,on GPU1 is live running
experiments.round65_audit selections. Reporter49795 identity was verified again
atdabfba. Keep these existing controller/audit/reporter handles;
do not duplicate the queue.
Cache link:attempt1/data points to the
user-requested data disk's GRoPE/round65_actual_gs_cache/attempt1.
Old stopped queue/reporter/execution records were preserved in
attempt1/stopped_execution_before_full_1793726229406865680. They are historical,
not handles to restart. Do not duplicate this new controller or its child jobs.

Full input preparation review77901a:attempt1/preprocessing_review1.json,sha256
49322617f42beef7880f40bb4e6aa8edf32d4994673ec631ea62c928e97c5d17.
Original partition membership/order and conversion/source identities match.
All7757objects had enough primitives for sampling2048 without replacement.
Fixed covariance-floor trace contribution has fit object-median medians
1.256e-4(MACGS)and9.774e-6(ModelNet); individual maxima can be large(.686/.967).
These are metadata diagnostics,not a thin-axis distortion bound or evidence
that the floor is universally negligible. Keep the fixed conversion and record
the source differences; no floor tuning,task prediction or new numerical gate
was added. Preparation success establishes usable inputs,not PE effectiveness.

Axiswise fitting-input diagnostic is now COMPLETE:experiments/round65_axis_floor_diagnostic.py,
attempt1/axis_floor_diagnostic1/{plan.json,objects.json,summary.json}.
Prospective planf5d09d SHA79c26a02651915695cd0c57ff6e4cb5841744b8320de210238a7e3fb7937b7de;
summary SHA2ac2574c1480dfffd10040fc36f7d8ed957d1fb2b5866ada4155ca6f09653c22.
Only6227fit objects/12752896cached primitives were read,using one CPU thread
and15.73seconds of analysis;no val/dev_eval/TEST arrays,predictions or GPU work.
The known rotated-diagonal fixture and unequal-token aggregation check passed.
With eigenvalues sorted ascending,the measured quantity is epsilon/lambda_k
of the regularized covariance,versus3epsilon/trace. Floor share exceeds50%
on the thinnest axis in2.5414%of primitives:MACGS.4482%,ModelNet4.0539%.
The >90%share rate is.8920%overall. The median of per-object median thin-axis
shares is.04973%;large local tails remain,including some heavily affected objects.
Most >50%thin-axis cases still have trace-floor share below1%,so trace alone
was insufficient to characterize those tails. This weakens a broad floor-
domination explanation;it does not measure task harm or rule out minority effects.
All roles share the same preparation. Keep the current floor and primary study;
this diagnostic adds no acceptance gate and establishes no PE efficacy.

Pair-information scope diagnostic is now COMPLETE and reviewed84d49b:
experiments/round65_relation_information_diagnostic.py and
attempt1/relation_information_diagnostic1/{plan.json,result.json,review.json}.
Prospective planc2eaabb6bf9839b47bace83d796ecffa501222327823a8947b3ca7c3ac57034a;
resultcc06c810ff4b4988d6a82cad57aaa8ca36ea17c11d2f2e803ea359916a67d7e6.
Two fixed analytic Gaussian pairs have identical spectra and exact pair terms
(distance squared2,mi3/2,mj5,traces33/38 and20 before the documented scaling/logs).
Their original11D MLP inputs agree exactly in the FP64 check,but their second
covariance's variance along the center displacement is2/95 versus8/475.
They are not related by a common orthogonal transform:distinct eigenvalues of A
force diagonal-sign stabilizers,while B's nonzero xz/yz precision entries leave
only plus/minus identity;neither maps(1,1,0)to(1,-1,0).
This establishes an information limitation of the pair map,not a whole-model
collision:direct descriptors and the shared13D content distinguish the inputs.
No task data,trained checkpoint or GPU was used;executionbc17db took1.77seconds.
The cases are valid positive-definite inputs,but their frequency or task relevance
in native GS was not measured. Task sufficiency does not require a complete pair
descriptor,and no extra invariant,width search,new gate or accuracy explanation
follows from this result. Keep the full study unchanged. Observationf303fd still
finds12/15fits complete and the three original jobs live at epochs55/50/50 with
finite training/validation histories;all full development predictions remain pending.

Candidate-history review while the final R65 fits run:do not rebrand tested
compact routes as new candidates. R3 already implemented separate center and
covariance-descriptor rotary channels(docs/round3_world_review.md). R44's learned
spatial Gaussian basis beat its additive MLP control by1.6443pp but trailed the
strong point RoPE by2.5182pp(allfive seeds);R45's Gaussian coordinate warp changed
PCK8 by only+.006694pp versus point RoPE,with2/5positive seeds. R46's native3D
coordinate warp beat its point control by2.7573pp but trailed MLP by.3356pp in
allfive seeds. These are bounded development findings,not impossibility theorems.
R56's unit center phase plus single-sided Gaussian quadratic and R57's19-channel
two-sided Gaussian relation are also implemented and tested;neither established
a stable strong-control lead. Those shapes were predicted from token features,
not observed optimized-GS covariances. R65 directly tests the latter distinction;
it does not erase those earlier results. Sources checked:round44_spatial_report,
round45_phase_warp_report,round46_native3d_report,round56_native_joint_report,and
round57_gaussian_pair_report under docs. No new architecture or retraining was
launched from this review;wait for the fixed full-data comparison before deciding.

Initial complete-data runtime observation31302d:attempt1/initial_runtime_health1.json
covers the first completed epoch/390steps of Gaussian11,directMLP22,distanceMLP33;
losses and recorded gradients are finite,peak allocated memory12.10-12.34GiB.
DirectMLP11 had only the epoch0 checkpoint at that snapshot,so its gradient
observation is pending(null),not a failure. This is startup health,not a method
ranking,convergence result or selected validation replay. Original full runner
saves resume state every epoch but history.json only with five-epoch validation;
the observer therefore prints logged epochs and does not call empty training
history finite. A stale logged epoch alone must not trigger a restart.

Latest completed-training milestone97d020:all15fits have finished,all exit0.
Gaussian22 completed120epochs/46800steps with all25validation epochs,finite
histories,matching source/saved-artifact identities,and actual sampling streams
and shared initial weights matched to directMLP22. Its earliest validation-CE
minimum selects epoch90;common and PE weights changed. The original selected
GPU replay is running(dabfba);all90development prediction groups,metrics,reporting
and scientific/visual review remain pending. Training completion is not efficacy.

Earlier completed-training milestoned49454:fourteen of15fits have finished,all exit0.
Reviewc5277c checks Gaussian33's complete120epochs/46800steps,all25validation
epochs,finite histories,source/artifact identities,and actual sampling streams
and common initial weights matched to directMLP33. Its earliest validation-CE
minimum selects epoch90;common and PE weights changed. Original selected GPU
replay and development evaluation remain pending. At observed_unix1793747161.25
the final Gaussian22 job is live at logged epoch115 with finite training/validation
records and no failure. Continue the original controller through all evaluations.

Earlier completed-training milestone06614a:thirteen of15fits have finished,all exit0.
Review6a70c9 checks directMLP33's complete120epochs/46800steps,all25validation
epochs,finite histories,source and saved-artifact identities,and actual sampling
streams/shared initial weights matched to distanceMLP33. Its earliest validation
CE minimum selects epoch105;common and PE weights changed. Selected GPU replay
and development evaluation remain pending the original queue. At observed_unix
1793746702.34 the remaining Gaussian33/22 logged epochs are115/110;both original
processes are live with finite training/validation records and no failure.

Earlier completed-training milestone66a3f1:twelve of15fits have finished,all exit0.
The last NoPE11 and distanceMLP11 fits completed120epochs/46800steps. Reviewea62d8
confirms complete finite histories,correct recorded/source identities and saved
artifacts,plus actual sampling-stream and shared-initialization agreement with
directMLP11. Their earliest validation-CE minima are epochs115 and100,respectively.
At observed_unix1793739214.55 the remaining logged epochs were Gaussian33 25,
Gaussian22 20,and directMLP33 20. All three are live with finite histories and
no failure. They still need the full120epoch budget;do not confuse12/15complete
fits with nearly complete remaining training time. Original selection replay,
all90 development evaluation groups,reporting and scientific review remain.

Earlier completed-training milestonec240a7:ten of15fits have finished,all exit0;
the final pending models were launched automatically. Reviewsdc1c24 and55deb2
check the new distanceMLP22,NoPE33,centerRoPE11,and NoPE22 completions:all
120epochs/46800steps,complete finite histories,correct saved/source identities,
and matching actual sampling streams/common initial weights with completed
same-seed controls. Their earliest validation-CE minima are105,110,115,and110,
respectively. No development predictions or selected GPU replay yet;the original
full-queue audit and evaluation remain pending. At observed_unix1793738128.27
the continuing logged epochs were distanceMLP11 100,NoPE11 80,Gaussian33 10,
Gaussian22 5,and directMLP33 5;all five have finite histories and no failure.
Keep the same controller and complete the remaining fits and90-group evaluation.

Earlier completed-training milestone733be3:Gaussian11 and centerRoPE22 both
finished120epochs/46800steps with exit0;the queue automatically launched
Gaussian33 and NoPE22. Read-only review7161dd confirms complete histories,
all25scheduled validation epochs,finite records,matching history/lock identities,
saved artifacts,and changed common weights. Gaussian PE parameters also changed.
Each new completed fit matches its completed same-seed directMLP control's
actual120epoch sampling stream and shared initial weights. The earliest
validation-CE minima select epoch100(Gaussian11)and120(centerRoPE22).
This is execution and pairing evidence,not a task-effect result. Full-queue
selected replay,remaining seeds and all development predictions remain pending.
At observed_unix1793737047.18 the continuing logged epochs were distanceMLP11
85,distanceMLP22 115,centerRoPE11 75,NoPE33 90,NoPE22 30;the new Gaussian33
had only epoch0 recorded. Observed histories are finite with no failure.

Earlier completed-training milestone143d93:directMLP11 and22 both finished
120epochs/46800steps with exit0;the queue automatically launched NoPE33 and
centerRoPE11. Read-only reviewa21968 confirms complete120-row training histories,
all25scheduled validation epochs,finite records,matching history/lock identities,
saved artifacts,and changed common/PE parameters. The fixed validation-CE rule
selects epochs95(seed11)and110(seed22),matching the earliest logged minima.
Original full-queue selected-model replay and complete same-seed pairing remain
pending;no development predictions or task-effect conclusion yet. At
observed_unix1793736054.04 the continuing logged epochs were Gaussian11 105,
distanceMLP11 70,distanceMLP22 100,centerRoPE22 70;new NoPE33 had five finite
epochs and new centerRoPE11 had only epoch0. Allsix active processes were live,
with no recorded failure. Continue the original15-fit queue and90-group study.

Earlier completed-training milestone9b7be5:distanceMLP33 finished120epochs and
46800steps,exit0;the original queue automatically started centerRoPE22.
Read-only reviewfea17a confirms120training rows,all25scheduled validation epochs,
finite recorded losses/gradients/validation,matching history/scientific-lock
identities,and saved artifacts. Its actual sampling-stream identity and common
initial weights match the completed centerRoPE33 seed. Common weights and
learned PE parameters changed. Its best validation CE was at epoch120;retain
the fixed budget. Original full-queue selection replay and development
predictions are still pending;two completed control fits do not establish the
Gaussian candidate's effect. At observed_unix1793735099.63 the continuing logged
epochs were Gaussian11 95,directMLP11/22 105,distanceMLP11 55,distanceMLP22 85;
the new centerRoPE22 had only epoch0 recorded. All observed histories were
finite and no failure was present. Continue this same queue through all15fits.
Follow-up5968e7 confirms centerRoPE22 now has five finite training epochs;
distanceMLP11/22 have reached60/90 logged epochs. Allsix active PIDs remain live.

First completed-training milestone47d92e:CenterRoPE33 finished120epochs and
46800steps;the original queue recorded exit0 and automatically started
distanceMLP11. Read-only review4701fb checks120training rows,all25scheduled
validation epochs,finite losses/gradients/validation records,matching history
and scientific-lock identities,and presence of saved artifacts. Best validation
CE was at epoch120. This is one completed fit,not convergence evidence or a
method ranking;retain the fixed budget. Original selection replay and all
development predictions remain pending the complete training queue.
At47d92e the other logged epochs were Gaussian11 50,directMLP11/22 55,
distanceMLP33 60,distanceMLP22 30,and new distanceMLP11 0. The five continuing
fits have finite histories and no failures;the newly launched fit's training
health is unobserved until it records an epoch,not a failed or passed check.
Subsequent verified wait(observed_unix1793731473.55)confirms the same controller
and allsix active processes live. New distanceMLP11 now has five recorded
epochs with finite losses,gradients and validation records;its startup health
is observed. Other logged epochs:Gaussian11 50,directMLP11 55,directMLP22 60,
distanceMLP22 35,distanceMLP33 65. One fit remains completed,eight pending;
no failure,training change or additional experiment was introduced.
Earlier health observation2c0aaf found decreasing training loss and validation
CE in all six then-running fits. New distanceMLP22 first-three-epoch losses
3.800/3.315/3.077 and first gradients were finite at4b67d7. These are learning
health observations,not matched final comparisons. Earlier at epoch5,Gaussian11,directMLP22,distanceMLP33 validation CE decreased from
about4.1 at initialization to2.51/2.43/2.52 at epoch5(82a5b3). This is learning
health on checkpoint-selection data,not a matched final method ranking.

A prospective output-only source-error diagnostic was added before any full
development predictions existed(8c9140). Source categories are disjoint30-class
sets and native conversion scales differ; combined classification gains could
partly reflect reduced source confusion. Code:experiments/round65_source_diagnostic.py.
Plan:attempt1/source_confusion_plan1.json. After original full validation and
metrics complete,run python -m experiments.round65_source_diagnostic report.
It reuses all90 existing exports and decomposes class-balanced errors into
wrong-source and wrong-class-within-source terms,with every paired seed sign.
An extra source-known30-label argmax is explicitly diagnostic,not the primary
60-way score. Source/category confounding prevents causal shortcut attribution;
never divide signed contributions by a small net gain or rescue a failed
primary comparison using this diagnostic. No training/selection/config change,
new data access,new pass gate or original report-source modification.
Four hand-calculated fixture tests passed25445d:error partition and limits of
source restriction,class-versus-object weighting,paired signs/complete coverage,
and rejection of missing classes/nonfinite outputs. Freeze the helper and plan;
scientific interpretation remains pending actual complete results.

The original fixed scientific lock remains unchanged:five roles x three seeds,
6227fit/765val/765development evaluation,120epochs/46800steps perfit,90groups
across six conditions. All13D common input channels remain. Scheduler uses two
idle-GPU observations before launches and preserves external jobs. Fits save
per-epoch resume state and may return75 after an intentional8-hour shard;
the original queue requeues that same fit without changing its total budget.
Monitor with python -m experiments.round65_monitor --experiment attempt1 on the
host; verify actual live handles,not only old status files. Complete all fits,
selection checks,exports,metrics,automatic report and actual scientific/plot
review. The global goal remains ACTIVE and unachieved; no fresh confirmation
or official TEST data is accessed. Earlier pending-download counts below are
historical and superseded by this completed acquisition and new execution.

Latest result,2026-11-04:content_route1 is COMPLETE and scientifically reviewed.
Allnine fits finished120epochs/16200steps; all12owned children exited0. All120
actual training streams match pilot3 for each paired seed/role. All54 new
evaluations and54 baseline groups were compared. Selected GPU validation
replay error is0. The actual content_route_summary.png was viewed; allnine
permutation checks preserve predictions,max logit rounding5.245e-6. Both review
flags are false. Records:content_route1/{report.md,result.json,review.json}.
Controller112181 exited normally; do not restart it or its completed children.

Base means(full/removed covariance content):Gaussian74.73/77.04%,directMLP
72.75/73.69%,distanceMLP76.48/76.38%. Removing covariance content raises the
Gaussian mean2.30pp,but seed effects are(+4.45,-3.39,+5.85),and its paired-class
interval crosses zero. Removed Gaussian-direct=+3.35pp,interval[+.96,+5.73],
but seed gains(+11.77,-3.19,+1.47)are unstable; a class interval averaged over
seeds is not evidence of seed reproducibility. Gaussian-distance=+.66pp,
interval[-2.44,+3.72],seeds(+7.12,-7.33,+2.18). Shape-free distance remains
competitive. Both routing interaction intervals cross zero. Sparse256 removed
Gaussian72.49% trails direct72.94% and distance73.78%. Retain this negative.
Content redundancy is not established as the cause of the missing stable
Gaussian advantage; no unique shape-structure or general-effectiveness claim.

Decision for the user-authorized full-data validation:retain the original
five-model,13D common-content R65 specification and120epoch/three-seed recipe.
Expand only cohort coverage to the complete60classes/two sources. The unstable
removed-content mean does not justify promoting that route to the primary.
Preserve the ablation as development mechanism evidence. Full validation will
test cohort/source composition explanations,not tune a new architecture or
extend training to rescue an absent effect. Acquisition acceptance is complete
and the newly identified full controller above is executing this decision.
The prospective full-data decision is now recorded in
attempt1/full_validation_plan_20261104.json,sha256
29016642d9cd19fb90f87e8da170783a95038d9ecdc579ed2015008e7c5f1b9f.
It reuses the unchanged original scientific lock,allfive models,three seeds,
120epochs/46800steps and90 evaluation groups. Existing unit/CUDA preflights
remain applicable; no model or training source was changed. Archive the old
stopped operational records before a newly identified full controller is
launched. The observer now also accepts --experiment attempt1 and handles
CPU preprocessing stages and the original resumable runner's history schema.
The former waiting snapshotf4bb87(7598/7757files)is superseded by full acquisition
acceptance and launchc7aec1;the download is no longer a pending dependency.

Latest user instruction,2026-11-04:after the current content-routing study has
finished and been interpreted,and the mirror download has completed and been
accepted,autonomously run a complete-data validation. This explicitly authorizes
the next full60-class,two-source study; no additional permission is needed.
Use all7757 official TRAIN objects already declared DEVELOPMENT,retain the
fixed within-class split,matched training budget and strong controls,and report
source-specific,seed-specific and sampling-condition results. Choose the
candidate/content route from the complete current diagnostic evidence and
freeze its hypothesis,alternative,comparison and decision before new training.
Catalog split verificationca76c9:6227fit/765val/765dev_eval,all60classes in each;
MACGS2612/317/317 and ModelNet3615/448/448. Fixed120epochs imply46800steps per
fit. The two sources contain different30-class taxonomies; their separate
metrics describe within-source performance,not cross-domain transfer.
Do not silently change the current cohort or blindly restart the stopped old
full controller. GSModel60 TEST,original ModelNet TEST and the reserved SUN3D
scenes remain unread; complete TRAIN validation is not fresh confirmation.
Live check30a9a7:the original controller112181 and downloader99977 identities
match; eight fits are complete and the last distanceMLP33 is at109/120epochs.
Download7309/7757files,113.829/116.419GB verified,no failure. These newer
observations supersede the operational counts below.

Latest 2026-11-03: pilot3 is COMPLETE and scientifically reviewed. All15 fits
finished120epochs/16200steps; all18owned children exited0, all90 fixed
method/seed/condition evaluations completed. Selected validation replay max
error is0; source/cache/checkpoint identities, matched initialization and all
120 actual input streams were checked. The actual cohort_summary.png was
viewed. All15 permutation checks preserve predictions; maximum logit difference
6.914e-6 is harmless rounding, not a reason to repair the method.
Records:results/round65_actual_gs/pilot3/{report.md,result.json,review.json}.
Both actual_plot_review_pending and scientific_review_pending are false.
Controller103783 has exited normally; do not restart it or duplicate its fits.

This remains the fixed25-class MACGS development cohort:2157fit/264val/264eval,
seeds11/22/33,120epochs, six fixed conditions, same13D common token content.
Base mean-class accuracy:Gaussian74.73%,directMLP72.75%,distanceMLP76.48%,
centerRoPE64.63%,NoPE56.91%. Gaussian-direct paired gains are+1.06/+2.25/+2.63pp,
mean+1.98pp, paired-class95% interval[-1.23,+5.10]. Gaussian-distance gains
are-0.76/+3.94/-8.43pp, mean-1.75pp, interval[-4.58,+1.18]. Clear gains over
RoPE/NoPE do not establish special Gaussian structure over strong controls.
The failure is substantive, not a numerical-audit threshold artifact.

Sparse256:Gaussian72.16%,distance70.70%,mean difference+1.45pp but signs(-,+,-)
and interval includes0. Dense1024:Gaussian75.71%,distance79.00%; allthree seeds
favor distance. No stable sampling advantage. Training curves show learning;
Gaussian's earlier loss descent is only descriptive. Selected Gaussian
epochs80/90/70 and distance85/85/80 precede the endpoint; do not launch another
longer-budget rescue merely because the desired outcome was absent.
All methods still receive covariance through common content. This tests the
additional value of shape in relational PE, not whether shape has any value.
No fresh confirmation, full60class/two-source result or global success claim.
Keep prior RGB/transfer negatives and pilot1/2 seed/density instability.

The user-restored global research goal remains ACTIVE and unachieved. A bounded
content-routing ablation is now RUNNING on this already-used development cohort:
remove the six covariance channels from common token content for Gaussian,
directMLP and distanceMLP; retrain allthree seeds with the same data/budget and
compare with their completed full-content counterparts. Preserve covariance
unchanged in each explicit PE. This distinguishes shape-information routing
from unique Gaussian relational structure. It is a diagnostic, not a new
architecture or fresh confirmation. The shape-free distance model deliberately
has less geometry information; beating it alone cannot support special Gaussian
structure. DirectMLP retains the same shape information in its PE.

Code:models/round65_content_route.py,experiments/round65_content_route.py.
Retain the original seven center/appearance stem columns at initialization;
all other shared/PE weights and RNG state match each original seed/role. All
three reduced models have576 fewer active parameters, no unused padding.
Same existing2157/264/264 arrays,120epochs/16200steps, optimizer, SO3/jitter,
sampling, minimum validation CE selection, allsix conditions. Nine new fits,
54 new evaluation groups, paired with nine completed pilot3 fits/54 groups.
All original scientific sources and baseline artifacts remain unchanged.

Four meaningful CPU unit tests passed at80ec53:initialization/RNG pairing,
equivalence to masking only covariance content, covariance retained in PE with
finite active gradients, and factorial-contrast direction/complete coverage.
Synthetic CUDA forward/backward on twice-observed-idle GPU5 passed atd21305,
zero task optimizer updates. Plan session34409 exited0 atf01f44, before fits.
Plan SHA555d6d236ecd22ae0cf026a147aa282f37808a4cdd482922cf10401ccd3235b5.
Records:results/round65_actual_gs/content_route1. Detached controller112181,
start_ticks1399151612 launched atcf0784;launch.json/queue_status.json are
authoritative. Latest live check8d551f (2026-11-04):eight fits completed120epochs/16200steps,
all exit0:Gaussian11/22/33,directMLP11/22/33,distanceMLP11/22. Those worker
handles are terminal, not jobs to restart. Only distanceMLP33 remains active
onGPU4(PID28708),100epochs;none pending. All recorded values are finite; no
failure. Completed-fit budgets and normal exits were checked; final paired
comparison still awaits allnine fits. Allnine jobs have now started.
Queue requires two idle observations and at most4owned jobs; preserve external
jobs. Subsequent turns have been verified waits on actual live handles.

Health observation47d7ba:the then-active directMLP33/distanceMLP11/22 had lower
recent training loss than their first five epochs (about3.3 to.9-1.0),with lower
validation CE;new distanceMLP33 also showed learning. This is process health,
not a method ranking. Resource snapshotsbedbdd/b239de show additional compute
jobs subsequently sharing the four training GPUs,about29GiB total memory and
100% utilization. Existing fits remained finite and advancing;preserve these
jobs and do not turn shared-host elapsed times into a method speed comparison.

One completed-fit health checka4c8b5 passed:Gaussian11 has complete budget,
unchanged plan/cache/artifact hashes, all120 input streams identical to pilot3,
expected original/reduced initialization hashes and changed trainable tensors.
Selected epoch90 by validation CE.861142577; independently recomputed saved CE
agrees to1.11e-16. Record:content_route1/completed_fit_health_gaussian_11.json.
This is one fit's health/selection evidence, not the final comparison or a GPU
logit replay. Allnine sealing,54 new evaluations and scientific review remain
pending. The earlier launch_health.json covers only the first four epochs.
Do not duplicate this queue. Continue through allnine fits, selection/input
pairing, three export jobs, report and actual plot interpretation.

Historical design cross-check during this wait (75aceb/a33273/3e3c17):R56 already
tested unit-amplitude centerRoPE plus a separate learned Gaussian quadratic
preference;R57 tested a two-Gaussian Jeffreys relation. Neither established
stable gains over its strong controls. Do not present either combination as an
untried repair. Those used learned latent covariance on point clouds, whereas
the running ablation uses observed native optimized-GS covariance. Their
negative results are relevant history, not universal rejection of every native
support encoder. Finish this information-routing comparison before choosing
another descriptor. No new variant, tuning or confirmation access was added.

Mirror download PID99977,start_ticks1396096550 remains live. Latest check8d551f:
7101/7757files,112.740/116.419GB verified, no failure/completion record. Acquisition:
results/round65_actual_gs/preparation/acquisition_1793682705671116425.
Keep hf-mirror.com direct routing, four workers and shared server cooldowns;
do not consume proxy quota or restart an advancing download. The original
full60class R65 controller remains stopped. Do not automatically resume it
or change the completed pilot3 cohort as newly downloaded files arrive.
The read-only helper python -m experiments.round65_monitor --experiment
content_route1 observes the active ablation and download; omit the option to
inspect completed pilot3. It never launches/stops work. Growing .part files can explain
unchanged verified-file counts. Use live PID/start_ticks checks on the host.

Training-catalog availability snapshot924b4d (metadata/file sizes only):all3246
MACGS files across30categories are present with catalog-matching sizes;ModelNet
has2153/4511 such files and12/30 complete categories(30..41). Record:
preparation/availability_snapshot_1793718659569131672.json. This does not replace
the downloader's SHA256 checks; its concurrent verified count was5398/7757.
No PLY payload, prediction or test split was read for this snapshot, and no
running cohort was changed. Extra availability supports later study planning,
not an automatic full-controller restart or a fresh-confirmation claim.

The bounded ModelNet native-input preflight also completed:16 already-declared
fit objects, all class30 glass_box, same16x512 inputs/common seed11 initialization,
allfive unchanged models, random-initialization squared-logit backward,
zero optimizer updates. Records:attempt1/native_modelnet_preflight1; wrapper:
experiments/round65_modelnet_preflight.py. GPU1 was idle in two observations;
owned diagnostic22299 exited0. All forward/backward values finite, no entirely
zero parameter-gradient tensors; peak11.147GiB. This supports these16 inputs'
arithmetic only, not source-wide task effectiveness. Floor trace median6.53e-6;
1.036% of axes receive over50% variance from the fixed floor. Comparison with
the earlier16-object MACGS class0 batch confounds source and class; do not tune
the floor or infer a domain law. All16 original scientific sources unchanged.

Earlier operational snapshots and scientific history follow. Paused/running
statuses below are historical; current state above takes precedence.

Pilot2 is now COMPLETE: all15fits,45evaluation groups and all18owned child
exits succeeded. All selected checkpoint hashes match and selected validation
replay max error is0. The actual final plot and scientific result were reviewed;
records:results/round65_actual_gs/pilot2/{result,review}.json and report.md.
Base mCA means:Gaussian63.33%, directMLP62.22%, distanceMLP63.89%, centerRoPE
57.22%, NoPE56.67%. Gaussian-minus-direct across11/22/33 is-10/+6.67/+6.67pp;
Gaussian-minus-distance is-11.67/+3.33/+6.67pp. Longer training improves all
methods, but does not establish stable special Gaussian-structure value.
Sparse256 means:55.56/54.44/57.78/54.44/56.11% in the same role order. Keep
the instability and lack of density advantage alongside gains over RoPE/NoPE.
The earlier four-completed snapshots below are historical.

Latest 2026-11-03: user requested a training-health and interpretation check.
Pilot2 has completed4/15 fits: Gaussian11/22/33 and directMLP11, each120epochs
and1800steps, all exited0. The remaining11fits are waiting for free GPUs;
there are currently no active pilot children. Original controller PID101352
and start_ticks match and its heartbeat is live (host check499bc7). Continue
this same controller; do not duplicate completed fits or occupy other users'
devices. Full selection, development evaluation and sparse/tilt results are
still pending. The full scientific objective remains unachieved.

Completed-fit diagnostics passed atab8d97: finite losses/gradient norms,
complete budgets, changed trainable parameters, source/cache/artifact hashes,
selection rule, independently recomputed validation metrics, same-seed input
streams and shared initialization. Original20epoch inputs match the first20
epochs of each corresponding longer run. The actual partial curves were viewed.
Records:results/round65_actual_gs/pilot2/status_review_1793684772098202094/
review.json and curves.png. This check changes no experiment setting or fit.

On checkpoint-selection VALIDATION data, Gaussian11/22 improves from
33.33/36.67% mCA at the20epoch budget to55/65% at120epochs; Gaussian33 is63.33%.
DirectMLP11 improves from33.33% to63.33%. Thus the longer budget/schedule
helps learning but does not yet establish special Gaussian structure value:
the only completed matched control comparison favors directMLP by8.33pp.
Gaussian11/22 select epochs65/95; directMLP11 selects70, with small later CE
rebounds. Gaussian33 selects120, so convergence is not uniform. These are
partial validation diagnostics, not independent task-evaluation results or
fresh confirmation. Complete the planned controls and evaluations before
ranking methods or attributing sampling robustness to longer training.

Previously the user requested one further verification. Pilot2 keeps the exact
five-class240/60/60 cohort and allfive models, train from scratch for120epochs
with seeds11/22/33. The20-to120epoch comparison uses only shared seeds11/22;
three-seed control comparisons are reported separately. The cosine schedule
also spans120epochs, so this is a joint budget/schedule intervention. No
architecture, data or other training hyperparameter was tuned. The original
full R65 controller remains stopped and the global goal remains unachieved.

New driver: experiments/round65_budget_pilot.py. Records:
results/round65_actual_gs/pilot2/plan.json, launch.json, queue_status.json and
controller.log. The source/plan/cache identities are fixed; all16 original
sources and the completed pilot1 source remain unchanged. Reuses the exact
pilot1 cache arrays; no additional raw-data decoding/download is needed.
Evaluation objects were already used in pilot1 and remain development data.
Plan session44670 exited0 atd08403. Synthetic report/schedule checks passed
atfac4a4, including preserving a third-seed regression and paired-budget
comparison rather than comparing unequal seed averages.

Detached controller PID101352, start_ticks1395829105, launched by session76227
(terminal0 ateb6bb6). Snapshot64083a: waiting_for_gpu,15pending fits,0active,
0completed; all8GPUs have other users' live tasks. The controller checks live
occupancy and requires two free observations before launch, up to4owned jobs.
It will train15models, check selections, evaluate45groups and generate the
report/plots automatically. Do not duplicate the queue or take occupied GPUs.
That zero-completed snapshot is historical; the latest four-completed status
and partial interpretation are above. Final interpretation and actual complete
result-plot review remain pending. Continue this handle when checking
progress; the existing mirror download is independent and remains untouched.

Latest 2026-11-03: the user authorized a small experiment using currently
downloaded data. The separate pilot is COMPLETE; the original full R65 controller
and reporter remain stopped. The global goal is still unachieved and its saved
status remains paused; this request authorizes the bounded pilot only.

Pilot result: Gaussian base mean-class accuracy40.83%, directMLP38.33%,
distanceMLP38.33%, centerRoPE36.67%, NoPE39.17% (two-seed means). The primary
Gaussian-minus-direct comparison changes sign: +8.33/-3.33percentage points.
Sparse256 Gaussian mCA33.33% is below every control, including NoPE45.83%.
Thus there is a small average base-condition lead but no stable strong-control
advantage or density robustness. Nine of ten selections are at the final
epoch20; convergence is not established by this short budget. Do not label
the idea generally effective or infeasible on this pilot alone. Keep the full
R65 specification fixed; its complete cohort/budget remains the next broader
comparison when authorized. No fresh confirmation was accessed.

Allten fits completed300steps each. Alltwelve owned children exited0, the
controller session89626 exited0 atcf630e, and no pilot children remain active.
Matched common initialization, actual per-epoch streams, source/data identities,
validation selection, exact selected GPU-logit replay and independent metrics
passed. The actual plot was viewed and scientific review completed. Records:
results/round65_actual_gs/pilot1/{result,review}.json, report.md, pilot_summary.png.
Measured shared-host end-to-end controller duration was162.83seconds; this is
not a controlled model-speed comparison. Concurrent mirror acquisition was
observed running separately atbac39c, PID92476, acquisition_1793675993308119664;
do not restart an older download handle from the historical notes below.

Pilot: experiments/round65_pilot.py, results/round65_actual_gs/pilot1.
Plan session65140 exited0 atacae9c before new decoding or task training.
Classes0/1/2/3/5 (bag/bottle/train/telephone/pistol) were selected by availability
only, preserving the original R65 partitions and order:48fit/12val/12dev_eval
per class, total240/60/60. Allfive unchanged models, seeds11/22,20epochs,
512tokens,2048 cached native primitives. AdamW and five-epoch warmup unchanged;
cosine schedule ends at20epochs. The classifier still has60outputs. Baseline,
sparse256 and tilt evaluation are fixed; allten validation-based checkpoint
selections must finish before pilot evaluation decoding. This is development
evidence from one source/five classes, not fresh confirmation or full R65.

Selection/partition and schedule checks passed at8a7cba; all16 original frozen
sources still match. Pilot controller session89626 launched at83a05c; its
queue_status.json, child logs and result.json are authoritative.
Derived data goes to the user's data disk under
GRoPE/round65_actual_gs_cache/pilot1, separate from full attempt1. Preserve
earlier native diagnostic, RGB, transfer and other negative evidence.

Latest 2026-11-03: DATA DOWNLOAD RESUMED THROUGH HF-MIRROR at the user's
subsequent request; the research goal, controller and reporter remain PAUSED.
No R65 task training has started. R65 is implemented and frozen; the full
research objective remains UNACHIEVED. R64 is complete as recorded below.

Active download: detached nohup PID99977, start_ticks1396096550, launched
ata10e6e with the server-aware rate-limit fix, using
experiments.round65_mirror_acquire acquire. Transport record:
results/round65_actual_gs/preparation/mirror_transport_1793682705565592808.
The user-requested hfd comparison is COMPLETE with a negative deployment
decision: keep this SAME Python download PID99977, now resumed and verified
not suspended atcae329. No hfd production download was started. Snapshot:
1686verified files/49,624,730,923bytes. Summary:
preparation/hfd_download_comparison1.json. Existing complete files and legacy
.part files were preserved; four SHA-verified benchmark files/63,489,162bytes
were hardlinked into RAW for future reuse, with no vertex/TEST decoding.

Benchmark driver: experiments/round65_hfd_transfer.py, user tool
/home/liwt/hfd.sh unmodified. Local hfd metadata/allowlist was seeded from the
fixed TRAIN catalog; proxy environment and aria2 user configuration disabled.
Four offline preparation/reuse checks passed. Only the identity-checked
download PID was temporarily SIGSTOP'ed during each test and SIGCONT'ed in
finally, including on errors. Allthree benchmark sessions11816/42074/24181
terminated1 after native hfd failures; their failure records are preserved.
Research processes were untouched. No benchmark process remains active.

Records: preparation/hfd_benchmark_1793689787538151457 (4threads/2jobs),
hfd_benchmark_1793690466136248666 (4threads/1job), and
hfd_benchmark_1793690805957730527 (4threads/1job,4MiB min split).
The fixed comparison planned alternating orders on two batches, each mixing
MACGS and ModelNet TRAIN files. It could NOT complete: allthree configurations
failed their first hfd batch with HTTP429 (4,2,4 failed file requests).
The Python first batch fully verified63.49MB in125.52seconds/0.506MB/s.
In the second hfd attempt, two large files totaling51.96MB passed full SHA256
after about27.62seconds, but both smaller files failed429. This partial
throughput is not a paired complete-batch speedup. Single-connection hfd
diagnostic5.41MB/20.21seconds succeeded; its success alone does not establish
an accelerated replacement. The prespecified deployment criterion (both
batches faster, >=1.25x aggregate) was not established, so no switch.
Acquisition record: preparation/acquisition_1793682705671116425 under the
same R65 results directory. Continue this download rather than starting a
duplicate; use its progress.json and live process state for current progress.
The existing complete files and partial downloads are reused on the user's
specified data disk. Other research processes were left untouched.
Persistent output: preparation/mirror_background_1793682705338693458.log;
replacement/launch identity: mirror_rate_fix_1793682705338693458.json in that
same preparation folder. The old download PID92476 was identity-checked and
intentionally stopped with SIGTERM to install this transport fix. Its last
record was1283files,35,534,082,082bytes. Startup rechecks those cached files;
an initially smaller new progress counter does not mean data was removed.
The old acquisition_1793675993308119664 and background log are historical.

Rate-limit fix: obey Retry-After seconds/HTTP-date and Hugging Face RateLimit
exhausted-bucket reset timers; share cooldown across allfour workers, with
exponential60-300second fallback plus jitter when no timer is supplied. A
server-specified longer delay is never shortened. Pace initial/redirect
request starts by0.25seconds. Record only response host, status and allowlisted
rate headers, never signed URLs/credentials. Keep direct-only mirror/CDN
transport and the unchanged downloader's retry bound, Range, hashes and
population checks. No frozen scientific source changed. Fifteen offline
transport/acquisition checks passed at3f42a5; cooldown-monitor display and ETA
suppression passed at02e3bc. Restart running monitors to load the new display.
Sources: https://huggingface.co/docs/hub/rate-limits and
https://www.rfc-editor.org/rfc/rfc9110.html#name-retry-after .
The prior six429events lacked response headers; their enforcing origin is
UNKNOWN. The old logs do not establish continuous bandwidth throttling.
This corrects early retries; it is not evidence of an increased download rate.
Live post-fix check4559df:1287verified files/35,672,086,565bytes, eight direct
CDN responses and zero new429/network failures; new data has passed the
unchanged full-file hashes. Startup cache rechecks are complete. This brief
observation verifies continued acquisition, not long-term absence of limits.

Previous download session69833/PID24332 was found absent at ee664f/766fdf;
the tool session was also unavailable. Its last progress was677files and
18,132,901,176bytes with neither failure.json nor completion.json. The exact
exit cause is UNKNOWN; do not attribute it to a terminal disconnect, OOM,
user action or DNS without evidence. Earlier DNS retries were recorded but
the current mirror/CDN DNS checks passed; disk free space was322GiB. Record:
preparation/mirror_interruption_1793675993077485750.json. Host check766fdf
found no other acquisition or research process before restoration. The old
acquisition_1793672768213940263 progress file is now historical. The live
monitor follows the latest attempt automatically; hardcoded watch commands
targeting the old progress file must be updated.

Read-only live download monitor: run
/home/liwt/miniColmap/.venv/bin/python /home/liwt/GRoPE/experiments/round65_download_monitor.py.
It follows the latest acquisition by default, refreshes every5seconds, and
estimates ETA from new verified bytes over a rolling300second window after
30seconds of observation. Initial cached bytes are excluded from the rate.
In-flight partial files are not counted; short-window estimates may fluctuate.
Completion/failure, stopped process and120seconds without completed files are
reported, with ETA suppressed for stopped/failed/stale work. Ctrl+C exits only
the monitor. --once shows a snapshot; --attempt pins a specific acquisition.
This monitoring utility does not change frozen sources or resume research.
Monitor correction on2026-11-03: the initial implementation excluded the first
observed byte count but could count subsequent cached-file revalidation after
an acquisition restart as throughput. It now snapshots complete file sizes
from the fixed catalog at monitor/attempt startup and waits for the verified
counter to reach that baseline before sampling speed. Partial files are not
included in the cache baseline. Synthetic restart/reuse checks passed at39d127;
live snapshot925c0a succeeded. Restart a running monitor to load this correction.
Read-only speed diagnostic2ac94d measured164,370,164new bytes over48.63seconds
including partial-file growth:3.38MB/s. The restarted download's longer average
was3.22MB/s after subtracting its prior verified18.13GB. Current attempt logs
contain6earlier HTTP429 responses and1later read timeout, not proof of a
continuous rate limit. No downloader settings, frozen sources or data changed.

The mirror rules exist in /home/liwt/.bashrc (not /.bashrc), but its early
noninteractive return means this download environment did not inherit them.
The new transport adapter disables all proxies within its own process and
uses an explicitly empty urllib ProxyHandler for both hf-mirror.com and CDN
redirects. The observed large-file redirect is cas-bridge.xethub.hf.co.
It does not modify the user's shell configuration or use a proxy fallback.
The original downloader still enforces the same pinned revision, TRAIN
allowlist, full SHA256 identities, resume checks and storage reserve. All16
frozen sources remain unchanged; only the download transport is adapted.

Direct mirror probes completed with full published SHA256 matches:
59/radio_0016.ply (4,348,474bytes; session75090 exited0 atca2a08) and resumed
1/0000d545.ply (43,945,147bytes, reusing34,603,008bytes; session68002 exited0
at468875). The latter returned exact HTTP206 Content-Range
bytes34603008-43945146/43945147. Records: preparation/
mirror_transport_1793672661753170448 and mirror_transport_1793672699513503386.
These validate acquisition only; no vertex arrays were decoded. Host check
df8109 found no duplicate acquisition or research processes before launch.

Earlier pause: the user initially elected to download through a mirror
themselves and requested all download/research execution to stop. The later
mirror-download request supersedes that pause for data acquisition only.

The exact original acquisition PID97208/session82596, controller PID99999 and
reporter PID108323/session7520 were identity-checked and stopped with SIGTERM
at the user's explicit request. Host checke9e0af confirmed none remained live.
Record:results/round65_actual_gs/attempt1/user_pause_1793669755645691268.json.
The last download record had203verified files,6,313,833,208bytes. Complete and
partial files are preserved at the user's data directory; no files were deleted.
The proposed supplementary download pilot was not started. Controller state
stopped/SystemExit143 is user-directed cessation, not a scientific failure.
The reporter status file may retain its last waiting snapshot; its process has
been stopped. All earlier live-process snapshots below are historical.

Latest continuation is PROGRESS after verified waiting: a prospectively recorded
native-input diagnostic has completed. New experiments/round65_native_preflight.py
uses the first16 lexicographically complete frozen-fit objects selected before
decoding. All are class0/MACGS; this availability-selected batch does not cover
all60classes or the ModelNet source. The full R65 population remains7757objects.
Plan session77550 exited0 atca950a before any diagnostic numerical decoding.

Native diagnostic session49318 exited0 atd0bebe on twice-observed idle GPU4,
RTX A6000. Allfive frozen models completed B16/N512 forward/backward with finite,
nonzero parameter gradients and matched common initialization. The objective was
mean squared initialization logits; no labels, optimizer update, task training,
validation, development evaluation or confirmation data were used. Relation
models needed10.9-11.1GiB peak allocation, centerRoPE/NoPE about0.4GiB. Shared-host
first-call timings are not controlled speed comparisons. This supports native
input feasibility for these16objects only, not task efficacy or generality.

The sampled98304physical axes span variances2.455e-13 to0.4093 after the frozen
object normalization. Median floor share peraxis is0.0311%;153axes(0.1556%) have
floor share above50%,69axes(0.0702%) above90%. Median of per-object median trace
shares is0.0167%; the largest sampled trace share is1.30%. Trace contribution
alone therefore does not bound thin-axis distortion. This is descriptive;
there is no new threshold, post-hoc floor tuning or measured task-effect claim.
Decision: continue full R65 unchanged and retain this interpretation limit.

Records:results/round65_actual_gs/attempt1/native_preflight1/{plan,conversion,result}.json.
Derived diagnostic inputs are on the user data disk under
GRoPE/round65_actual_gs_cache/native_preflight1/inputs.npz, separate from the
future full attempt1 cache. All16frozen sources still match at9a2459. Original
download82596 remains live at66423d; snapshot140/7757,4,872,487,023verified bytes.
The existing controller99999 and reporter108323 are retained. Actual task fits,
complete result review and the full research objective remain unfinished.

Previous progress: executable report generation and an attached
watcher for the original controller. The preceding directory/status continuation
was a VERIFIED WAIT on the original download and controller. New scripts are
experiments/round65_report.py and round65_report_watch.py; no frozen scientific
source changed. Six synthetic report checks passed (ca950d), including refusal
to report incomplete work, legitimate training-shard handling, full90-group
coverage and preservation of a losing primary result. All four synthetic plots
were actually viewed. These are report checks, not new scientific results.

Reporter session7520/PID108323, start_ticks1394570782, launched8aa5a3 and
confirmed live on the host atf8e066 with the SAME download97208/controller99999.
Its controller attachment records the exact command/start_ticks. It will generate
report1 only after the original controller finishes successfully, then leave
scientific and actual result-plot review pending. Do not duplicate this watcher.
Record:results/round65_actual_gs/attempt1/report_preparation/attachment.json.
Latest download snapshot87/7757,3,228,099,440verified bytes atf8e066.
Continue existing processes; the full research goal remains unachieved.

User explicitly moved data storage to
/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data.
Raw files now go to its GSModel60 subdirectory; future caches go to
GRoPE/round65_actual_gs_cache. All29 old complete/partial files (1,042,916,310
bytes) were copied and hash-verified before removing old copies. The old
download89929/PID46265 was intentionally terminated143 for this user-directed
relocation, not an observation timeout. Record:data_relocation.json inR65prep.

Current acquisition is session82596/PID97208, start_ticks1394467239, attempt
results/round65_actual_gs/preparation/acquisition_1793666412509915562.
Original controller PID99999, start_ticks1394489062, is also live. Both exact
commands and identities were checked on the host at8e941c; download pollf65c39
remained live. The snapshot had68/7757 verified files (2,359,869,723 bytes).
Continue these SAME processes; queue_status.json and the current acquisition's
progress.json are authoritative. Do not launch a duplicate from an old snapshot.
The controller waits for complete verified acquisition, then preprocesses,
trains15models on observed idle GPUs, audits selections, exports and evaluates.
Report preparation is complete and the watcher is attached as above; actual
report generation and scientific/visual review await the completed experiment.

Retired external download79822/PID117360 exited1 after a truncated HTTP body;
59014/PID121135 exited1 after bounded connection timeouts. Short-body resumption
and bounded network retries were fixed and tested before the freeze. The formerly
short file was completed and matched its published SHA256. Preserve failure
records short_response_recovery1.json and connection_timeout_diagnostic1.json.
The download remains incomplete. It resumes partial bytes, verifies full LFS
hashes and preserves50GiB free space on the target disk (375.93GiB free at8e941c).

Pinned GSModel60 revision:e6407400f768fc7c0d8f8d1b2ed112739b38e9dc.
All7,757 official TRAIN objects/all60classes:3,246MACGS and4,511ModelNet derived.
All4,511ModelNet IDs belong to ShapeSplat's published original training list;
all7,757source byte hashes are distinct. Catalog SHA256:
b0ffc82bbe846ccde0e0db07ba457194046e1d8bc9810d17081e52b4627f0333.
Expected bytes116,419,005,518. Internal prospective split6227/765/765;
the whole selected TRAIN population is development, never fresh confirmation.
No test split content has been decoded. The16 fitting-only native diagnostic
objects above are now numerically decoded; full cohort preprocessing is pending.
Binary PLY headers from both sources retain scales/quaternion/opacity/DC.
Nonidentical MACGS physical-instance grouping is not independently established.

Read ROUND65_ACTUAL_GS_PROTOCOL.md. The experiment is now frozen at
results/round65_actual_gs/attempt1/lock.json, SHA256
45a9ac9ee02ff1bd7c8a8b19f937eed13f3f923fe9ddf0b43d5e8179253eb6fe.
All16frozen sources were rechecked unchanged at8e941c. Freeze session66477
exited0 at517c00;16tests passed. Training runner, audit, GPU controller and
resume support are implemented. No frozen scientific setting has changed.
models/round65_actual_gs.py retains the exact11-dimensional Gaussian relation
and compares direct16D relationMLP, distanceMLP, centerRoPE andNoPE. All roles
receive the same13D center/log-covariance/opacity/DC content. Five roles x
three seeds,120epochs planned. Tests cover native conversion, independent relation
arithmetic, matched inputs/initialization/permutation/gradients, split guards,
training execution helpers, and download resumption/hash/retry behavior.

CUDA preflight session72068 exited0 at0d151b on an observed idle RTX A6000.
All5roles completed a synthetic full-batch B16/N512 forward/backward/update,
with finite gradients and changed learned positional parameters. Common initial
weights match. Peak allocated memory was about10.9-11.1GiB for relation models
and0.4GiB for centerRoPE/NoPE; this is feasibility/cost evidence, not a controlled
speed benchmark or a task result. Records:attempt1/units.json and
attempt1/cuda_preflight.json (also preparation/cuda_preflight1/result.json).
No real numerical vertex arrays were decoded by the synthetic checks; the later
native fitting-only diagnostic is recorded above. Retain RGB,
transfer, classification and earlier negative evidence; actual-GS success alone
cannot establish the complete goal. Confirmation data remains untouched.

Latest 2026-11-02: R64 is COMPLETE. The nine new fits, selection checks,
66 metric groups, scientific interpretation and actual plot review are done.
See docs/round64_depth_field_report.md and
results/round64_depth_field/attempt1/completion.json. Original controller
79308/PID122444 and reporter 51872/PID75260 both exited 0; all 15 controller
children exited 0. Do not restart them. Earlier live snapshots below are historical.

R64 depth_surface EPE is 3.94646 versus ray_affine 3.66693: 7.62% worse,
with all three paired seeds and all five scene means worse. Relative to the
no-depth-derivative footprint control, PCK8 improves 0.434 percentage points
and CE improves 0.05998 across all three seeds and five scene means. Retain
this partial benefit alongside the failure against strong RGB controls.
Implementation checks passed; no fresh confirmation was accessed or warranted.

R65 experiment: directly supply optimized 3DGS parameters to a
Transformer and compare the current relation PE with strong controls receiving
the same attributes. This tests the core parameter-to-PE claim separately from
RGB parameter prediction. See the current execution state above. R62/R64 RGB
negatives, earlier transfer failures, and the complete generality/robustness/
fresh-confirmation objective remain in force. The full goal is UNACHIEVED.

R64 progress:all9fits COMPLETE andall9selection/operator checks PASSED.
Original controller79308/PID122444 LIVEat0ee58c. Lastfitfootprint/33 exited0
ata5fabc;selectionchild45503 exited0at0ee58c. Newold_evalexports startedONLY
AFTER selected.json wassealed. Exportchildren11/22/33 PID75507/75508/75509
onverifiedidleGPU0/1/3. No restart/no settingschange/no freshconfirmation.
Reporterwatcher51872/PID75260 LIVEat03655d;itwillrunreportafterterminalmetrics.
Selectionlogfa642e:9checks,CEmax4.78e-8,CPUlogitmax5.86e-5. This istechnical
validation,not efficacy. Nextfinish18newexports+48reused independent metrics,
automaticreport,actualimage/scientificreview. FullgoalACTIVE/unachieved.
Read R64control_scope_note.md beforeattributinganyperformancegain. Earlier
statussnapshotsbelowarehistorical;useexistinghandlesandcurrentqueuestate.

R64 live progress:8/9fits COMPLETE,all completedchildren exited0. Allthree
primarysurface andallthree localMLP seeds aredone;lastfootprint/33 PID88212
onGPU0 continues(step3700 ata1e4e6). No pendingunlaunchedfits. Original
controller79308/PID122444 LIVE(bd56b6),reportwatcher51872/PID75260 LIVE(0d22c7).
Continue SAMEhandles;do not restart. Selectionseal/oldevalexports/66metrics
andreport remainpending. Currentturnhasactualtrainingprogress,notutility
results. FullgoalACTIVE/unachieved;confirmationunused. Retain the R64 control
scope note when interpreting results;no frozen setting/gate changes.
Earlier snapshots below arehistorical.

R64 progress:3/9newfits COMPLETE;allnine fits have nowbeenlaunched. New
localMLP/33 child58297 exited0(fd3124);thelastpendingfootprint/33 launched
PID88212 onGPU0. Sixremainingactive;no pendingjobs. GPU2/3 hadpreviously
becomeidle andlaunchedlocalMLP/11 PID17513 andsurface/22 PID17514 at1551a1.
Original controller79308/PID122444 LIVE(fd3124);reportwatcher51872/PID75260
LIVE(9778e6). Do not restart. Snapshot56c6b5:footprint/11 step3100,
localMLP/22 step4600,surface/33 step2200,localMLP/11 step3500,surface/22step2000;
newfootprint/33 loading. Allnine-selectionaudit/evaluation stillpending.

Data-free interpretation check completed8c6bba;52frozen sources unchanged.
Read docs/round64_control_scope_note.md andR64control_scope_check.json.
The directMLP hasthesamefullnetworkRGB/cameraaccess andlocalfield source,but
its10Dpairdescriptors donotexplicitlycarryallfocal/relativeposecalculations
usedbyGaussiancommon-frame relations. Constructed examples:zero-gradient
z1/ray(0,0,1),fx100vs200 givesidentical[1,0,0,0,0]localdescriptor butdifferent
covariance;relativecamera tx0vs.1changesGaussiandistance0to.5. This doesNOT
show thefullMLP hasno calibrationaccess orcannotlearntheserelations,andisNOT
a measured cause of future performance. IfGaussianwins,reportwholepipeline
benefit;don'tuniquelycreditSPD/covariancefromthiscontrastalone. Surfacevs
footprintremainsamorefocused matchedcomparison. No newuniversalprerequisite,
experimentchange,tasknumericarrays/checkpoint/predictionaccess orfreshdata.
Reportsource/watcher remainunchanged;include thisnoteinfinalinterpretation.

CurrentturnPROGRESS:bounded source/algebra interpretation plusnewactualfits.
Metadata:results/round64_depth_field/attempt1/training_and_scope_progress.json.
NEXT:continueSAMEcontroller/watcher;finishremainingfits,allnine selectionaudit,
66groups andautomaticreport,thenactualimage/scientificreview. Completegoal
remainsACTIVE/unachieved;earliernegativesandactualGS/cross-task/generalization/
robustness/freshconfirmation requirements retained. Below entries historical.

R64 progress:2/9newfits COMPLETE. New primary depth_surface/11 finished
5000steps,selected step3300,51validation records;child27485 exited0
e0d5b2. New artifact/initialization/actualstream/selection-metadata checks
passed;sharedweights/allPEtensors changed;52frozen sources unchanged. Record:
results/round64_depth_field/attempt1/second_completed_fit_check.json.
This checks one new completed fit's metadata,not whole-network replay orutility.
Fullnine-modelselectionaudit/evaluation remainpending;no task-effect ranking.

Original controller79308/PID122444 LIVE,e0d5b2;reportwatcher51872/PID75260 LIVE,
cfa9b4. Primary/33 newlylaunched PID130141 onGPU6;localMLP/33 PID58297 GPU0,
footprint/11 PID58298 GPU1,localMLP/22 PID92175 GPU7 continue. At49ace2their
stepswere100/1400/900/800,respectively. Fouractive/threepending/twocomplete.
Continue SAME handles;do not restart oldwork. Automaticreportingafterterminal
metrics remainsattached;actualscientific/image review stillrequired. Previous
turnPROGRESS;thisturnPROGRESS:newcompletedfit andnextseedactualexecution.
FullgoalACTIVE/unachieved;confirmationunused. Oldsnapshotsbelowarehistorical.

R64 progress: first of9newfits COMPLETE. depth_footprint/22 ran5000steps,
selected step4000 by the frozen earliest-minimum-valCE rule,
51validation records. Original child32611 exited0 (a10504). Artifact hashes,
matchedcommoninit/actualstream andchangedcommon/allPEweights checked;52frozen
sources unchanged. This is a bounded one-fit metadata check,not full selection
replay or task-effect evidence. Record:first_completed_fit_check.json inR64root.
No old_eval exports orfreshconfirmation. The complete research goal isACTIVE.

Original controller79308/PID122444 LIVE (18e64f). GPU0/1 becameidle andtwo new
fits launched at ebb351:localMLP/33 PID58297 onGPU0;footprint/11 PID58298 onGPU1.
GPU7 next launchedlocalMLP/22 PID92175 at18e64f;surface/11 PID27485 continues
onGPU6(step4600 atsnapshot3caefd). Fouractive/fourpending/onecomplete; use
currentqueue/historyfortransientprogress. Do not restart anyexistingprocess.
Reporterwatcher51872/PID75260 LIVE at98584d,awaitingoriginalcontroller completion.
NEXT:finishremainingfits,allnine selections,66metricgroups,thenautomaticreport
plusactualimage/scientificreview. No change to frozen settings orsuccesscriteria.
PreviousgoalturnwasVERIFIED WAIT;thisturnisPROGRESS:firstfullfit/metadata check
andadditionalactualtraininglaunches. Earlier status snapshots below arehistorical.

R64 continuation: previous goal turn VERIFIED WAIT; this turn adds an
executing postprocessing watcher and verifies the original live training.
Controller79308/PID122444 still LIVE (1efdc6);do not restart. Reporter watcher
51872/PID75260 is also LIVE (50bed9),attached after checking original process
start_ticks1390571297 andcommand. New experiments/round64_report_watch.py
only observes the existing controller;after successful terminal completion it
runs the already-prepared report. No training/evaluation/settings changes.
Watcher will leave scientific/image review pending,not mark the goal complete.
Attachment metadata:results/round64_depth_field/attempt1/report_watch_attachment.json.
52scientific sources unchanged. Snapshotc90fe9:surface/11 step1000 and
footprint/22 step1400;use latestqueue/historyforcurrentprogress. Full9fits,
selection,66metricgroups,actualreport review andfullgoal remain unfinished.
No fresh confirmation access. Continue SAME controller andwatcher handles.

R64 verified-wait continuation: the previous goal turn was PROGRESS. This
turn directly confirmed original controller79308/PID122444 alive (9d4dd1)
and both child processes present (9379a6). Snapshot819bba:surface/11 step700,
footprint/22 step900;twoactive/sevenpending/zero complete fits.52frozen sources
still match (568102). No change/restart/new gate/evaluation/confirmation.
Latest transient snapshot:results/round64_depth_field/attempt1/wait_snapshot1.json.
Continue the SAME controller through all9fits and frozen evaluation;the full
objective remains ACTIVE and UNACHIEVED. Training duration is not a blocker.

Latest 2026-11-02: R64 is IMPLEMENTED, FROZEN and RUNNING. The full goal
remains ACTIVE and UNACHIEVED. Previous goal turn was PROGRESS (R63 closure
and R64 preparation); this turn is PROGRESS (model/runner/audit/controller,
four implementation tests,52-source freeze,CUDA preflight and actual fits).
Read ROUND64_DEPTH_FIELD_PROTOCOL.md and docs/round64_depth_field_plan.md.
The plan's unimplemented sentence and preparation flags are historical.

R64 root:results/round64_depth_field/attempt1. Lock SHA256:
cddd4c260d1a0b567754c69fdd7bcb5c62629a4eb474129e2ecd9f6ad1b28cfe.
All52scientific sources match. Freeze session33111 exited0 at1bd0d4;
freeze/tests outputc4c2d6. Four tests passed in22.822seconds;standalone first
pass52f9ce took20.762seconds. Tests cover analytic surface/differences/boundaries,
full-grid contract,positive floor,finite geometry and network gradients,
independent q/k permutations,exact initial surface/footprint equality,
input exclusions and shared common weights. Equal-focal initial eigvalsh
backward is finite; no arbitrary anisotropy/old static-integrator gate added.
Independent NumPy synthetic operator f9b74f: max relation errors3.20e-6,
1.48e-6,6.59e-7 for surface/footprint/localMLP;new geometry max1.74e-7.
This is implementation evidence,not task utility or independent lab replication.

Original LIVE controller:session79308,hostPID122444,launched52a982.
Direct poll2ddae0 and hostprocesscheckb75517 confirm live;do not restart.
At launch all8GPUs were occupied;controller waited,then twice-observed idle
GPU6 andGPU7 before launching owned tasks. It never stops other users' jobs.
Preflight child16971 exited0 onGPU6,RTXA6000;all3roles had finite gradients,
losses and changed PEweights. Peakallocated8.997/8.997/8.289GiB,respectively;
shared-host timing/memory are not a speed benchmark. Actual PEparams10465/
10465/10369. Main training now running onavailable GPUs,up to6ownedworkers.
Snapshot4ad5bb:surface/11 PID27485 step100 onGPU6;footprint/22 PID32611 step0
onGPU7. These are transientprogress values,not selected task results.
Original preflight/fit processes and fresh queue_status are authoritative.

The frozen experiment:one scalar depth pertoken/layer;center=z*r;
Sigma=5.25*(J_uJ_u^T+J_vJ_v^T)+(1/300)^2I;pixel derivatives use canonical
18x24 grid reconstruction frompositions. Retain original11->96->96relation
andcalibratedcrossattention. depth_surface primary,depth_footprint without
J's depthgradients,depth_local_mlp same localfield information via10->96->96.
No newbackbone,depthsupervision,pretraining,renderingloss or physical-recovery
perfection gate. Gaussian coupling and attainable range change together.

Nine new fits,three roles x11/22/33;exact5000steps andold576/64/480RGBpairs.
Allnine selected beforeanynewoldevalexports.24exactR55/R62RGBreference models
and48predictions reused byidentity,notnewindependent evidence.18new+48reused
=66independently recomputed metricgroups;retainallnegative outcomes. No fresh
confirmation or actualGSdata accessed. Strongestcontrols/sameinformationMLP
mustbe beaten; beatingfailedR62alone isinsufficient. OriginalRGBfeatureextractor
remainsfrozen;head/PE/matcher learnjointly.

Postprocessingexperiments/round64_report.py prepared,notpartoffrozen training.
It covers33models,66groups,16pairedcontrasts,15/60gaps,validationcurves andcost.
Incomplete-run guard passed00f214 before reading taskmetrics;no report1exists.
It still needs actual completed outputs,rendering,visual/scientific review.
Executionmetadata:results/round64_depth_field/attempt1/execution.json.
NEXT:continueoriginalsession79308;inspectpreflight/fitfailures ifany,otherwise
finishallninefits,selectionaudit,exports,independentmetrics,thenactualreport
andinterpretation. Do not rank methods fromshortpreflight or unfinished fits.
ActualGS,cross-task,applicablegeometry/robustness/generality andunusedconfirmation
remain required bythefullgoal. R62/R63andearliernegativeevidence remainvalid.
The following entries are HISTORICAL and do not supersede this status.

Latest 2026-11-02: R63 fixed-weight RGB diagnostic is COMPLETE. The full goal
remains ACTIVE and UNACHIEVED. Previous goal turn was PROGRESS (R62 closure
and R63 preparation). This turn is PROGRESS: R63 implementation,four tests,
freeze,CUDA execution,54diagnostic groups,independent metrics,actual image
review and interpretation;then primary-source review and concrete R64 plan.
Read docs/round63_rgb_diagnostic_report.md and docs/round64_depth_field_plan.md.

R63 root:results/round63_rgb_diagnostic/attempt1;50frozen sources unchanged;
lock1243ba05673b6ab3a208542a35d31424fa678e696868e3c2f76a11cb50158e70.
Original controller session13530/PID93526 exited0 at89af48;all6children0.
Session13530 was REUSED after R62 terminal;do not confuse it with R62PID105183.
Report session12307 exited0 ate5c0a4. No R63 process remains to restart.
Completionafc0a5 SHA256:
36d49da4802b31d36ef44970ab99dfb36c7ac4d670c6e97b06248b9877fcc0e6.
Chinese report SHA256:
2100a4a3d8ea4f2689a6dd56d4863b72fdca839d22616562673f39969c05e7d2.
Both PNGs ACTUALLY VIEWED in call containing3ce45e/e9ff9b. Generated pending
flags are historical;completion records actual scientific/visual review.

R63 scope:48oldfitpairs(16per3scenes),64oldvalpairs(1scene),9selectedR62models,
54fixed-weight groups,no fitting,no old_eval/confirmation numeric access.
Nine original val replays are BIT-IDENTICAL toR62 logits andCE;allweights
unchanged.Independent metric maxerror3.42e-8.CUDAA6000passedprobe;forward
cumulative42.89seconds(sharedhost,notbenchmark).Four unit tests passed.

R63 main result: all six forward interventions worsen EPE/CE on fit andval,
all3seeds andeachscene mean. ValEPE original3.08519;observedcenter26.44074;
observedshape10.37488;both25.23301;initialshape3.98137;sphere4.24490;
zerorelation28.43777. Independently trained observed_full baseline2.67133.
Fit original1.41782,both19.93356,zero18.99084,observedbaseline1.25153.
Evidence supports investigating co-adaptation/interface distribution,not an
instant oracle replacement remedy. Interventions shift distributions andlater
features,so no unique-cause claim;zeroingPE is not retraining aNoPE/PRoPEcontrol.

Native-center distance medians averaged overseeds:fullfitL1/L2 .82068/.71939m;
val1.07465/.85293m. ValL2predicteddepthmedian2.49103m vsobserved3.178m.
Shape log-error medians valL1/L2 3.89996/3.39413;fixedsphere3.49128.
Sensor supports are references,not trueGS. LearnedPE neednot recover physical
parameters exactly;do not add geometry-reconstruction perfection as a gate.
All declared strata have0fraction above95%tanhcoordinate saturation.
Observedpatch16covariances outsidecoordinatebounds:fit46.730%,val34.616%;
clippingrelativeFrobeniusmedian~1.27e-8/5.45e-9,p90 .85659/.75746.
Retain tailrange limitations,but no broad prediction saturation diagnosed;
clippingis notoptimalprojection. Masks onlystratifyanalysis,notmodel/evaluation.

NEXT R64: test a compact tied depth-field Gaussian generator while retaining
R62's11->96->96relation andcalibratedcrossattention. Predict one depth pertoken;
mu=z*r;Sigma=5.25*(J_uJ_u^T+J_vJ_v^T)+(1/300)^2I,whereJ includesdepthgradients.
Fixed8pxcentral/one-sided derivatives;reconstructgridbypixelcoordinates to
preserve token permutation behavior. RGB frontend assumes acompletegrid.
Three newroles:depth_surface(primary),depth_footprint(no depthderivative),
depth_local_mlp(samez/derivative/rayinformation via10->96->96directrelation).
No addedbackbone,geometrysupervision,pretrainedfeatures orrenderingobjective.
Same5000steps/3seeds,data/selection asR62;9newfits,24exactR55/R62RGBreference
models,66predictiongroups(18new/48reused). Allnewselectedbeforeoldevalexports.
Mustbeatstrongcontrols,notmerelythefailedR62fullmodel. Preserveallnegatives.

R64 preparationd8bf89:results/round64_depth_field/preparation/availability.json
SHA256 a5df491afd5ebc53f6d9ca07acff5b574bbd074a3a127fc8b654376cf7413dde.
Allreferencecommoninit/samplingidentitiesmatchwithinseed;metadataonlyread.
R64 is NOT implemented,frozen orlaunched. Implement/check/freeze concrete
sources andrun onverifiedidleGPUs. The primary-source references inR64plan
boundclaims:Gaussianregionalencoding andsurface-awarePEarepriorart;no novelty
claim,physicalparametersneednotbeexact. Ournewgenerator remainsuntested.
ActualGS,cross-task,applicablegeometry/robustness/generality andfreshconfirmation
remain required bythe full objective. The following entries are HISTORICAL.

R62 completed background:

Latest 2026-11-02: R62 RGB relation bridge is COMPLETE. The full research goal
remains ACTIVE and UNACHIEVED. Previous goal turn made bounded fit-ray input
checks and waited on live jobs. This turn is PROGRESS: remaining fits,18 model
selections,60-group evaluation,independent metrics,actual image review,Chinese
interpretation and formal R62 completion. R63 diagnostic samples are prepared.
Read docs/round62_rgb_relation_report.md and docs/round63_rgb_diagnostic_plan.md.

R62 root: results/round62_rgb_relations/attempt1. All27 child jobs exited0.
Original controller13530/PID105183 exited0 atfe74d2; observer17714/PID6934
exited0 atd37989. Do not restart any R62 process.44 scientific sources unchanged;
lock d5c7319b0aafb3f8b4d81ee442d7244d549cabdea353561229bce92381af5aa9.
Completion4d13a7 SHA256:
4ef41adad9bca5363d60612ad343ec0f9be86df4d5b284a10601e83f29c26bb9.
Chinese report SHA256:
28bd3e404f4d441d1fd9a196517dc05881cb9b4ea9d466d75a39bcfdb5c3393d.
Both report1 PNGs actually viewed in call containing69abfa/e9ecc9. Generated
pending flags are historical;completion.json records completed review.

R62 scope:18new fits,sixroles x11/22/33;12exactR55 camera/plane/ray-point/ray-affine
models reused.5000updates,identical common initialization and actual streams.
All18new selections preceded new old_eval exports.60 independent metric
recomputations:36new and24reused prediction groups. Old development scenes;
no fresh confirmation. New fit/selection cumulative16438.75seconds on shared host.
Metric maxerror1.98e-9;selectionCE3.99e-8;CPU replay4.98e-5;operator2.53e-6.

R62 decision: RGB PRIMARY FAILS STRONG CONTROLS. Scene-macro softEPE:
predicted_full3.99521,predicted_point4.09188,free_relation3.86734,
camera_mlp3.92554,PRoPE3.86360,calibrated-plane3.87887,ray_point3.70651,
ray_affine3.66693. Full is worse than all camera/plane/ray/phase controls for
all3seeds;versusfree_relation only1/3wins. Versusray_affine +.32829px,all3seeds
and5/5scene means worse. Do not promote this RGB candidate to confirmation.

Full versusits own point control: -.09667px,+.483ppPCK8,3/3seeds5/5scenes,
butCEmean+.00370worse. Initial full covariance is anisotropic and point is a
fixed sphere;this does not alone isolate learnedshape. See interpretation note.
Observed_full3.12407 versusobserved_point3.38778: -.26372px,+1.089ppPCK8,
CE-.05181;all3metrics3/3seeds5/5scenes better. These are PRIVILEGED measured
geometry diagnostics,not fair RGB competitors. They show this center-matching
task can expose shape value;the failure cannot simply be blamed on its metric.
The .87115px observed/predicted gap does not alone diagnose prediction quality.

Gaussian maps retain the R58 eleven relations,but move from within-view
selfattention to camera-calibrated crossattention. RGB extractor frozen;
parameter head/PE/matcher jointly trained,no depth supervision. Observed
covariance is patch second moment,not optimized3DGS. RGB permutation helps
locate dependence but is not a real-world robustness pass.15/60gaps bothretain
RGB failure. Preserve all R55/R57/R58/R59/R60/R61 negatives.

NEXT: R63 bounded frozen-weight diagnostic on old fit/val. Seven fixed
predicted_full interventions (baseline,observedcenter,observedshape,both,
initialshape,sphere,zero relation),plus predicted_point/observed_full baselines.
Three seeds;48fitpairs (16/3scenes) and64valpairs;54summary groups;no newfitting.
Inspect center/shape differences,parameter saturation and practical range limits;
retain fallback/depth-coverage strata. Interventions are distribution shifts,
not standalone causal proof or a replacement for matched retraining.
R63 metadata preparation22b239:192unique rawframes,9selected checkpoints,
results/round63_rgb_diagnostic/preparation/availability.json SHA256
891989e6b18eaa158d5a5d705aea8dee7357a495f10d3cd8bb261fa77541c295.
Onlymetadata/checkpoint identities read. R63 is NOT implemented,frozen orlaunched.
Implement,check the declared interventions,freeze exact sources/identities,and
run short forward diagnostics on verified idle GPUs. Do not retune R62 or read
old_eval/fresh confirmation for this diagnostic. ActualGS,cross-task,applicable
robustness/generality and unused confirmation remain in the full objective.

R61 closure background (the current R62 execution state above supersedes the
old no-R62-launch sentence below):
Latest 2026-11-02: user resumed the full research goal; ACTIVE and UNACHIEVED.
R61 is COMPLETE, including all fits, metrics, actual visual review and Chinese
interpretation. Read docs/round61_normal_relation_report.md and its completion
record. Previous experimental goal turn was PROGRESS (R61 terminal evaluation);
the intervening discussion was planning only. Current turn is PROGRESS (R61
closure and preparation of the explicitly chosen RGB bridge). Do not restart
any R61 or earlier controller. No R62 experiment has been launched yet.

R61 root: results/round61_normal_relations/attempt1; 32 unchanged sources; lock
 de370b0d994d7d045b26eeb9d21fc38b4a524c411623337ba141130b2cb02027.
Controller92910/PID6062 exited0 at ab71da; watcher10962 exited0 at2d3b33.
All20children exit0. Six new source fits and six new real adaptations;15source
and15target models reused.48new independently recomputed prediction groups plus
120exact reused groups.12new selections precede all new evaluation exports.
Five implementation tests passed;metric maxerror1.78e-15;both report1 PNGs viewed.

R61 decision: retain the exact compact11->96->96 relation MLP. Real-adapted
region improvements over normal+spectra are+1.334/+2.572pp(base/sparse),3/3seeds
and5/5scene means positive. Versus axial:+.791/+1.174pp,only2/3seeds each.
Synthetic low-overlap vsnormal/axial:+1.457/+.947pp,3/3region seeds positive.
Synthetic sparse vsaxial:region-.419pp;center-.466pp withall3seeds worse.
All three full/normal/axial source-direct real averages remain belowdistanceMLP.
No universal triaxial necessity,source-transfer,actualGS or fresh-confirmation claim.
Target models start from own source weights; sparse eligibility populations change.

NEXT: user explicitly chose to reconnect the candidate to RGB before prioritizing
actualGS object classification. Test identical RGB matching task/features/cameras:
direct learned PE,RGB-predicted Gaussian relations,and matched PRoPE/calibrated
HRoPE controls. Observed geometry is PRIVILEGED diagnostic input,not a fair RGB
competitor. Preserve known camera relations equally in the relevant comparisons.
If observed works but predicted fails,investigate prediction/interface adaptation;
do not infer RGB prediction quality from cross-task scores. Do not expand local
point-cloud attribution indefinitely. Fix hypothesis,competing explanations,input
contracts,controls,budget and decisions before launching suitable idle GPUs.
Full original geometry,robustness,generality,actual-task/primitive and untouched
confirmation requirements remain. R55/R57/R58/R59/R60 negatives remain authoritative.
The following entries are HISTORICAL and do not supersede this latest status.

R60 completed background (authoritative outcomes retained):
Latest2026-11-02:R60 is COMPLETE; the full research goal remains ACTIVE and
UNACHIEVED. Read docs/round60_real_scene_report.md and the completion record in
results/round60_real_scene/attempt1. Current turn was PROGRESS:all30 target fits,
120evaluations,independent metrics,actual visual review and scientific interpretation
completed. PRoPE branches(3)/(4) were re-read;their latest messages do not undo
the completed R54/R55 comparison. Do not restart any completed controller.

R60 lock:fb24317caebd29419e117b6f248cd3b1e7eb3d7aab53ab070569ff2873ea0639;
24 scientific sources unchanged. Controller69037/hostPID10878 exited0 at46fc26;
report watcher4685 exited0 at14eb25. All38 child jobs exited0. Four implementation
tests and bounded pose/input/stream/selection checks passed. Independent metric
maxerror1.33e-15. Both report1 PNGs actually viewed during this review.

Same10 R59 roles x seeds11/22/33.30 exact source models plus30 NEW target fits;
60epochs,1080updates each. Target starts from each role's OWN source checkpoint;
initial target backbones differ across roles. Compare complete source-plus-target
pipelines,not a target-only PE reset. New target fitting/selection cumulative
1323.42seconds on shared host,not a controlled speed benchmark.
Old native SUN3D3/1/5scene split,576/64/480pairs. Sensor depth is PUBLIC INPUT
for this task;pose only supplies labels/scoring. No RGB-only/optimized3DGS/full
registration claim. Pose-independent centering/shared RMS scale;64regions/32NN.

## R60 evidence and immediate next step

Both source transfer and fixed target adaptation were planned before outcomes.
All30 target selections precede all exports.120groups,all5/5scenes evaluable.
Scene-macro regionR@1 by source-base/source-sparse/adapted-base/adapted-sparse:
candidate35.629/40.852/46.280/48.782;
distanceMLP36.742/42.545/44.469/46.700;
descriptorMLP33.384/39.290/42.583/45.853;
angleDescriptor33.511/39.348/42.737/46.022;
matrixMLP33.215/39.799/40.329/45.002.

Source candidate versus distanceMLP:-1.113pp base,-1.693pp sparse;all3sparse
region/softEPE/CE comparisons worse. Retain this transfer limitation.
Adapted base versus distanceMLP:+1.810pp,paired+4.172/-.254/+1.512;5/5scene means
positive. Center+1.135pp,2/3seeds;softEPE-.020704/CE-.068383,3/3seeds and5/5scenes.
Versus descriptor/angleDescriptor/matrix region+3.696/+3.542/+5.951pp;all4metrics
3/3seeds and5/5scenes better. Adapted sparse versus distanceMLP:+2.081pp,paired
+2.359/+.907/+2.978;5/5scene means better,center+.451pp. Sparse region versus all
three descriptor/matrix controls also3/3seeds and5/5scenes better;other metrics
have some scene/seed exceptions documented in the report.

Eligible queries:base20422/61440=33.239%,25/480zero-validpairs;
sparse8811/61440=14.341%,117/480zero-validpairs. No native-input exclusions.
Sparse changes supports/query populations;higher absolute sparse recall does
not establish fixed-query robustness. Units are normalized scene units,not metres
or pixels. Old scenes and reused candidate models are not fresh confirmation.

DECISION:retain exact11->96->96 invariantMLP as promising DEVELOPMENT candidate;
no general transfer/uniqueGaussian/novelty/global success claim. Current controls
do not isolate known normal-PPF relations. Read docs/round60_relation_literature_note.md
and docs/round61_normal_relation_plan.md. RoITr already uses normal point-pair
features;the current center-triplet angle control is not that control.

Next plan adds only two same-backbone controls:distance+spectrum+unsigned normal
PPF,and original spectrum with axisymmetrized directional covariance relations.
Six new source fits plus six fixed real-target adaptations;reuse existing exact
references. Keep source direct transfer separate from adaptation. No new module
stack,no additional pose input,no post-result budget rescue. R61 is NOT yet
implemented/frozen/launched. Fix concrete sources before launching on idle GPUs.
Original cross-task/actual-primitive/generalization/robustness/fresh-confirmation
objective and R55/R57/R58/R59 negatives remain. No reserved data access.

R59 completed background (authoritative findings retained):
R59 is COMPLETE. Read docs/round59_relation_attribution_report.md
and results/round59_relation_attribution/attempt1/completion.json. The full
research goal is ACTIVE and UNACHIEVED. Previous goal turn was PROGRESS (report
watcher/execution metadata and live queue); this turn is PROGRESS (remaining
fits,30 selections,120 main+30 diagnostic groups,independent metrics,actual
visual review and scientific interpretation). Do not restart completed jobs.

Controller40218/hostPID56148 exited0 atf5d1f6;report watcher42158 exited0 atbb31a1.
All18children exited0.15 frozen scientific sources unchanged;lock:
260319452a918ebbbd1ffff96df63e0a5c54eb9cf24fb773691168bead9706cf.
Both PNGs actually viewed in tool call containing74d7be. New fitting/selection
cumulative7590.89seconds; shared-host timing is not a controlled speed benchmark.

## R59 evidence and next scope

Primary is EXACT R58 invariant_mlp,adaptively selected AFTER development results.
Reuse18 R58 fits/72predictions;12NEW fits are distance_mlp,descriptor_mlp,
angle_descriptor_mlp,matrix_mlp x11/22/33. Same7caches,60epochs/14880updates,
backbone,common initialization and actual stream.78NEW prediction sets include
30base-reframe diagnostics. No official test/reserved confirmation access.
All30 selected before new predictions. Independent metric maxerror1.33e-15.

Base regionR@1/centerPCK/softEPE/CE means:
candidate38.499/22.504/.316488/3.352861;
distanceMLP35.078/19.996/.330808/3.443439;
descriptorMLP37.108/21.437/.323537/3.387721;
angleDescriptorMLP37.247/21.358/.323636/3.380668;
matrixMLP34.278/19.368/.332969/3.464639.
Candidate improves base region/center/softEPE for ALL3seeds against ALL4new
controls. Versus angle region+1.252pp,paired+2.767/+.585/+.404;object-conditional
CI[+.721,+1.813]pp. Versus descriptor+1.391pp,CI[+.870,+1.944]. CE versus angle
only2/3better. Old data/reused candidate and conditional object intervals are
not fresh confirmation or an estimate of training-seed uncertainty.

Low-overlap versus descriptor/angle region+2.680/+2.374pp,all3seeds positive;
center/softEPE alsoall3;angleCE only2/3. Noise versus descriptor/angle means
+.958/+1.130pp butonly2/3regionbetter;seed33-.595/-.024pp. Center3/3better,
softEPE/CE mixed. Sparse versus descriptor/angle region+.366/+.198pp,intervals
crosszero;angleonly1/3wins,centermean-.012pp. No stable sparse strong-control
advantage. Candidate versus pure distance/full matrix improves region andcenter
onall4conditions x3seeds. Keep these distinctions;no blanket robustness claim.

Fixed reframe: candidate region/center unchanged all3,maximum probability1.34e-6,
but446A-to-B winner-index changes overALLqueries including ineligible ones.
Do not claim every winner unchanged. MatrixmaxP.200,62306winnerchanges;mean
region+.043pp/center-.112pp/CE+.00110. Declared frame sensitivity is behavior,
not an implementation error;its causal contribution to baseline gap unisolated.
Angle controls have some local probability changes(.057/.113max) despite bounded
val checks;region mean changes+.002/-.006pp. Retain the full diagnostic,do not
convert small aggregate effects into new numerical gates or retroactively fix
the frozen comparison. No confirmed cause of those local changes is claimed.

DECISION:promising DEVELOPMENT candidate,not a generally validated method and
not promoted to untouched confirmation. A single11->96->96 map of Gaussian
invariant relations has useful base/low-overlap evidence beyond learned distance,
direct local descriptors,angles and raw complete matrices. No new module stack.
Sparse uncertainty remains. R58 scalar-form failure and R57 classification
limitations are retained. This is observed-support parameterization,not optimized
3DGS or a recovered texture-transport map; no novelty claim.

Next keep this exact structure and prepare one real-scene native3D correspondence
development experiment with matched strong controls and explicit density scope.
Read docs/round60_real_scene_plan.md. Metadata/file availability is now checked:
9 old SUN3D scenes,576fit/64val/480old_eval pairs,1605required sensor depth files
present. Saved results/round60_real_scene_preparation/availability.json;no new
numeric arrays/model outputs read. The plan declares sensor-depth point-cloud
input,a pose-independent shared normalization,and unchanged R59 candidate/all9
controls. Both fixed source transfer and60epoch target adaptation are planned
BEFORE any target outcome;512/256density conditions and per-scene reporting.
Pose belongs only in labels/scoring,not model input. Implement and freeze concrete
sources/data/configuration before launch. No next source lock/job exists yet.
No post-result budget/temperature/width rescue.
Base/low-overlap evidence justifies this development step without requiring all
old conditions to be wins. Generalization,cross-task evidence and untouched
confirmation remain necessary before the full research objective can be achieved.

R58 completed background (authoritative outcomes retained):
R58 is COMPLETE. Read docs/round58_observed_support_report.md
and results/round58_observed_support/attempt1/completion.json. The full research
goal remains ACTIVE and UNACHIEVED. The R58 primary is NOT promoted. Previous
goal turn was PROGRESS (repeatability diagnostic and completed fits); this turn
is PROGRESS (remaining fits,18 selections,72 evaluations,independent metrics,
actual visual review,scientific interpretation and completion record).

All R58 jobs are terminal. Controller93987/hostPID17782 exited0 atc39775; report
watcher73520 exited0 atc8766a. All26children exited0. Completion record created
at1bd93e. Do not restart any R58 worker/controller/watcher or historical R57 jobs.
Eight frozen scientific sources unchanged;lock:
0f843169bf465bb3202084cfd4f7f56d5846cba6ed34958152be3b128d4f993a.

## R58 evidence and decision

Observed Gaussian supports from32 input neighbors at64tokens/view,not latent
covariance predictions or optimized splats. Disjoint raw point-ID sampling,
independent partial views/noise/SE(3),no ground-truth pose in forward. Two
self/cross Transformer pairs;Gaussian sqrt(2J) sinusoidal relative embedding
inside self-attention only. Cross/final matching use content. Region labels
come from actual point-set overlap. No overlap-detection/full-registration claim.

Six roles x seeds11/22/33,18new fits.60epochs/14880updates each; common backbone,
shared initialization,actual order and validation-CE selection. Old official-
train-only ModelNet7910fit/965val/965old_dev_eval. All18 selections precede new
exports. Official test and reserved scenes remain untouched. Old objects do not
become fresh through a new task. Source/config:ROUND58_OBSERVED_SUPPORT_PROTOCOL.md.

Base means (regionR@1%,centerPCK%,softEPE,CE):
fullGaussian34.458/19.374/.331958/3.465510;
Gaussian-distance34.274/19.221/.331812/3.470909;
Euclidean35.135/19.970/.330620/3.450592;
distance+angle34.951/19.808/.332029/3.455356;
invariantMLP38.499/22.504/.316488/3.352861;
NoPE17.683/8.789/.416250/3.940711.
Full versus MLP region-4.041pp,paired-4.762/-3.926/-3.435;center-3.129pp.
Object-bootstrap region interval[-4.614,-3.450]pp is conditional on these three
fits,not seed uncertainty/independent confirmation. Full versus Euclidean-.677pp
and versus distance+angle-.493pp,all3seeds lower. Full versus metric-only+.184pp,
only1/3seeds better. No stable trace-component benefit.

Full/MLP regionR@1 by base,noise02,sparse256,low_overlap:
34.458/38.499;32.343/36.546;35.065/39.140;33.052/35.692.
Across all4conditions x3seeds,full loses to MLP in region,center,softEPE andCE.
Sparse full versus Euclidean/angle-2.161/-2.539pp,all3negative. Preserve the
limited positive:low-overlap full versus Euclidean+.365pp/angle+.563pp,all3
region signs positive,but object intervals cross zero,center signs mix,and
full remains below MLP and below metric-only. No cherry-picked promotion.

The MLP control has positive evidence worth following:versus both Euclidean and
angle,all4conditions x3seeds improve regionR@1,centerPCK andCE. This control
ALREADY consumes Gaussian-related invariant statistics,so its success is not
proof that Gaussian parameters are useless. Nor does it isolate their benefit:
nonlinear distance encoding or extra pairwise local-feature comparison could
explain gains. The original primary has failed;retain that fact when choosing
the successful development control for follow-up.

Effective query fractions across conditions63.49/62.99/45.71/48.61%;zero-valid
pairs58/59/161/58 of965. Sparse absolute recall increases do not prove robustness
on an unchanged population. All comparisons within each condition are matched.
Validation oracle diagnostic:4.407%eligible queries cannot both maximize overlap
and satisfy.1center threshold,even allowing all overlap ties,although another
available center satisfies it. This limited conflict does not explain away the
simultaneous region/center improvements of MLP. Base nearest-centerPCK bound82.34%.

A prespecified old-val repeatability diagnostic completed atfe9d19:
107710 relations,893analyzable objects. Mean symmetric relative changes:
Euclidean.094429,Gaussian metric.179075,full.174191. On99.44%of objects Gaussian
forms change more. This mixes support,center,sampling and nonlinear sensitivity;
not isolated covariance noise or a task-performance bound. Full is slightly
less variable than metric-only,so trace is not uniquely implicated. No new gate.
See docs/round58_development_diagnostics.md and results/.../diagnostics/repeatability.
One missing-bracket parse error occurred before data access;fixed/source-retained
in diagnostics/preparation_failure1. Main frozen experiment was unaffected.

Checks:18validation CE replays<=8.89e-16;72metric groups<=8.89e-16;actual direct
Gaussian scalar3.55e-15;trained rigid probability<=8.20e-7,zero winner changes;
permutation logit<=1.15e-5. Same-author independent arithmetic,not second-lab
replication. Both PNGs viewed in tool call containing341ffc. New fit/selection
cumulative10077.99seconds,not controlled speed benchmark. Full selected45/39/33,
MLP45/57/45;no global convergence claim,no budget/frequency/temperature rescue.

## Next authorized research scope

Stop this scalar Jeffreys/sinusoidal candidate;do not open fresh confirmation.
Continue from the successful R58 invariantMLP as a DEVELOPMENT discovery. It is
one two-layer map of11Gaussian-related invariants,not an added module stack.
Question:does its benefit require directional Gaussian relations,or are generic
learned distance and local-content comparisons sufficient?

R58 completion authorized one matched attribution comparison on the same task/data/backbone/
budget/seeds,exactly reuse R58 references,add distance-MLP,distance+local-descriptor
MLP,and distance/angle+local-descriptor MLP controls. Explicitly state features,
parameter counts,initialization,how angle neighbors are aggregated,and decisions
before launch. Common local descriptors are the same11input features all R58
models already received;geometry/pose labels must never enter forward. Reuse is
not a new independent fit or fresh confirmation. R59 above now implements that
scope,including a complete matrix-input control. If generic controls match,do not claim Gaussian-specific
utility. A reliable directional-shape gain would motivate separately frozen
real-scene and cross-task validation before untouched confirmation.

The full original objective remains:compact general PE with reliable strong-
control gains,appropriate geometry/robustness/generalization/actual-task evidence,
and untouched confirmation. R57 classification negatives remain part of evidence;
a local learned-control win cannot establish the full objective or novelty.

R57 completed background (its outcome remains authoritative):
R57 is COMPLETE. Read docs/round57_gaussian_pair_report.md
and results/round57_gaussian_pair/attempt1/completion.json. The full research
goal remains ACTIVE and UNACHIEVED; the candidate is NOT promoted. This turn
made PROGRESS: all remaining fits, selection checks, exports, metrics, reports,
visual inspection and the experiment-completion record finished.

All R57 jobs are terminal. Controller77259 exited0 at a2dd48; report watcher87784
exited0 at2eb30b; completion62192 exited0 at1416fa. All15children exited0.
Do not restart these sessions, completed workers or historical controllers.

## R57 result and interpretation

Nine new120epoch/29760update fits: pair_full, pair_distance, pair_center x
seeds11/22/33. Same R48 ModelNet official-train-only7910fit/965val/965old_dev_eval,
backbone, initial shared weights, actual sampling/augmentation stream and
validation-CE selection.39 exact R56/R48 references at the SAME seeds reused;
not39new independent runs. All48 selections preceded new dev_eval exports.
29 frozen scientific sources unchanged; lock:
3e1f37a49b82a07d48a5ac0432b9c803d84bf4563ba2b928aa94a5ec0507de0a.
Official ModelNet test and the reserved cross-scene confirmation data remain unused.

Full Gaussian-pair OA89.292%,mCA84.486%,CE.434493. Same-capacity distance:
89.223/83.683/.445731; center-only88.877/84.019/.433895; MLP89.154/84.457/.437119;
free quadratic joint89.154/84.192/.434653; R56 single-query joint89.223/83.749/.431164.
Full versus distance OA+.069pp, paired -1.658/+1.036/+.829; mCA/CE also2wins1loss.
Full versus MLP OA+.138pp, paired -.415/+1.554/-.725; mCA+.029pp, CE only1/3better.
Versus free quadratic OA+.138pp,1/3wins. These are not stable strong-control gains.
Full versus center-only OA+.415pp,mCA+.467pp and versus fixed distance OA+.484pp,
mCA+.902pp:all3seeds positive. Preserve this limited component evidence.

Prespecified positives: jitter02 versus MLP+.553pp,all3seeds;1024points versus
adaptive distance+.794pp,all3seeds;Y90 versus R56 joint+.933pp,all3seeds.
Limits: jitter02 remains-.173pp versus free quadratic;256points is-.484pp versus
center-only,all3seeds worse,and also below MLP/free quadratic. Tilted rotation
still catastrophically low at6.425%; no whole-network SO3/affine-invariance claim.
Removing only the trace compatibility term at fixed full checkpoints changes
OA-.242pp on average (two losses/one gain);CE increases in all3seeds. This is
component dependence,not recovered physical covariance or unique task utility.
The adaptive-distance term still uses shape during this intervention.

All48 formula samples:maxP6.93e-6,new9max7.35e-7,winner changes0,warnings0.
Saved validationCE replay<=1.33e-8;336metric correct counts and3diagnostic correct
counts exact,CE errors small. Same-author independent arithmetic,not a second
lab/full-network rewrite.39 reference prediction copies exact. Both result PNGs
viewed. New fitting/selection cumulative24645.52sec; not a formal speed benchmark.
The19 positional Q/K channels use FP64 geometric products; no fused-kernel claim.
Full selections100/95/100;distance100/115/120;center80/105/120. No convergence claim.

## Scope recorded at R57 completion (now concretized by R58 above)

Do not promote this classification candidate or open fresh confirmation. Do not
rescue it by temperature/frequency/budget scans. The conditional ordinary-shape
similarity reserve in docs/round57_attribution_note.md remains UNEXECUTED; its
stable-utility trigger was not met. Original scientific objective is unchanged.

Next make one native3D local-correspondence development experiment concrete:
state whether a Gaussian is learned comparison geometry or observed region
support, how its center/shape transforms, what relationship is being predicted,
and which competing explanation each matched control removes. Use the same
information/backbone for all controls, including appropriate generic and geometric
strong controls; preserve center-localization and region-correspondence measures.
Known-frame matching and unknown-pose registration are distinct tasks. For the
latter, ground-truth relative pose belongs only in supervision/scoring; it cannot
be used to pre-align forward inputs. Specify data/configuration/decisions before
training. At R57 completion no next-round candidate or budget was frozen; the
R58 state at the top of this document supersedes that execution status.

A classification tie does not prohibit a separately motivated geometry diagnostic;
a local-task pass does not erase the classification and rotation limitations.
General utility,appropriate geometry/robustness,cross-task evidence and untouched
confirmation are still required. Do not substitute one convenient task for them.

PRoPE branches(3)/(4) were read. Their same-camera calibrated comparison and
bounded budget diagnosis are already completed in R54/R55:plane and PRoPE are
near ties at5000steps,while full Gaussian lacks stable extra utility over point/
affine. For future calibrated multi-view studies preserve known camera operators
and feature inputs across PRoPE,plane and Gaussian variants. R57 latent Gaussian
similarity is not a texture-transport transform. Its mathematical construction
alone does not establish usefulness or novelty.

The attribution note also records the R46/R5/R41/R57 input semantics and checked
GeoTransformer/REGTR primary methods as task/control references. They have NOT
been implemented or reproduced. The following entries are historical background;
latest execution state and next scope are given above.

The following R56 findings remain authoritative background and negative evidence.

R56 completion state (its jobs are terminal; R57 jobs are listed above): read
[the R56 report](/home/liwt/GRoPE/docs/round56_native_joint_report.md) and
results/round56_native_joint/attempt1/completion.json. R52--R55 also complete.
The full research goal remains ACTIVE and UNACHIEVED. All R56 jobs are terminal.
Do not restart any completed queue or historical PID.
This goal turn made concrete progress: all nine fits,39 selected-model checks,
39 exports/273 condition groups,12 diagnostics,analysis and visual review finished.
Previous goal turn was a verified wait on the original live35361/18034 handles.
No confirmation dataset or official ModelNet test was accessed.

## R56 outcome and decision

Same old ModelNet official-train-only split7910fit/965val/965dev_eval; same
120epochs/29760updates,backbone,initial shared weights,sampling/augmentation,
optimizer and validation-CE selection. Nine new fits (three joint roles x seeds
11/22/33),30 exact R48 references reused at the SAME seeds. Do not mix these
three-seed means with the old five-seed summary or count reused fits as new data.
All39 selections preceded new evaluation. Sources26 unchanged at lock
269ca1e9530ea4707fc0bc2850b40ea1b5a232e3b0b4883ccca7b1c9534898c4.

Base OA:joint_conditional89.223%,joint_fixed88.808%,joint_free_quadratic89.154%,
conditional-alone88.325%,diagonal-alone89.223%,centerRoPE88.774%,MLP+RoPE89.154%.
Joint full vs Gaussian-alone +.898pp,3/3 seeds; CE improves in all3. This supports
center-phase complementarity. Free quadratic/fixed distance also improve when
center phase is added; the effect is not unique to Gaussian parameterization.
Full joint vs free joint only+.069pp,1/3 seeds,mCA-.444pp; versus MLP only+.069pp,
2/3 seeds,mCA-.709pp. Mean CE slightly better but only2/3 seeds in each comparison.
These strong-control results are near ties,not a Gaussian-specific gain.
Y90 and jitter02 OA are .898pp/.760pp lower than free joint,ALL3 seeds worse.
Tilted rotation remains catastrophic (full joint5.803%); no SO3 robustness claim.
Point reversal changes no class winner in any39models; tiny logit differences
(max1.93e-5) do not change the outcome.

Fixed-checkpoint shape interventions on full joint decrease OA21.071pp for
trace-preserving isotropization and53.610pp for fixed Z90 covariance rotation.
The models DO depend on shape parameters; lack of unique utility is not evidence
that the predictor is ignored. These interventions shift learned attention and
are not proof of recovered physical covariance. Each layer retains its mean
predicted from its CURRENT input; early changes can alter later features/means.
They do not clamp all baseline-trajectory centers. All12 diagnostics retained.

DECISION:no promotion,no fresh confirmation,no diagonal/isotropic joint sweep,
no budget/temperature/frequency rescue. Preserve positive component effects and
negative strong-control/robustness results. Full objective remains intact.

## R57 rationale, recorded at R56 completion

Current R56 Gaussian defines a query-conditioned region over point keys. Consider
whether Gaussian parameters of BOTH tokens should define their relative relation.
The existing compact_gaussian_dual_pe_reserve_20261101.md derives Gaussian
Jeffreys similarity with19position Q/K channels. Its width/additivity
counterexamples remain valid for use as probability mass; they do not reject
all learned token-similarity uses. Neither closed form nor invariance establishes
actual task usefulness. Do not restart its old static-integrator gates.

Before training, formulate one prospective comparison:full Gaussian-pair score,
same adaptive-center distance with the covariance-compatibility term removed,
and learned centers with fixed covariance; retain RoPE/MLP/free-quadratic strong
controls. The same-capacity removal control separates shape compatibility from
generic adaptive distance; the center-only control separates moving both centers.
Use one positional relation,not extra branches/losses/rendering. Explicitly assess
whether this is a meaningful structural change rather than another parameter
scan. The prospective R57 plan is now implemented and frozen as recorded above;
no R57 task results are available yet.
R56 does not prove this next hypothesis. Old development evidence first;
strong controls,appropriate robustness/cross-task and fresh confirmation still
required for any eventual generally useful compact scheme.

## Execution evidence and scope

Controller35361 TERMINAL exit0(fb7c88),all15children exit0. Report watcher18034
TERMINAL exit0(f7032b). Completion47178 TERMINAL exit0(80467b). Both result PNGs
actually viewed. Four bounded tests,30 reference-function checks and real-input
gradient checks passed before fitting. All actual sample streams equal originals.
39 independent first-layer formula samples:maxP6.93e-6,winner changes0,warnings0;
saved validation CE replay<=1.33e-8. All273 metric correct counts exact,CE error
<=1.13e-7. Same-author independent arithmetic,not another lab/full-network rewrite.
New fitting/selection cost16492.57sec with checkpoint overhead; shared-host
wall time is not a formal throughput benchmark. Common338508params; full/free
PE1746,full model340254. Some selected models at120epochs; no convergence proof.
Two transient host NVIDIA cleanup/NVML waits resolved without restarts,kills,
driver edits or source changes; retained in operational_delay1/2.json. The first
report-watcher sandbox invocation could not see hostPID5539; its preserved
watch_attempt0_failure.json predates the successful unchanged host invocation.

## Literature and interpretation retained

PRoPE branch(3),01a199fa-0796-7a40-9ba5-c084c53d3891,and branch(4),
01a19a69-896a-7423-8fff-3f54c97288d0,were read. Camera-relative structure is not
our invention; covariance shape alone does not specify internal texture transport.
PRoPE https://arxiv.org/html/2507.10496v2#S3.SS4 ;
HRoPE https://arxiv.org/html/2610.28284v1#S3.SS3 .
R56 weights spatial preference without attenuating unit-amplitude rotary content;
it does not perform homography alignment or measure real GS support covariance.

Riva et al., https://arxiv.org/html/2409.13291v1 ,sections4/5.2,combine separate
Gaussian and RoPE heads; clean/noise tradeoffs prevent a general robustness
inference. Publisher:https://doi.org/10.2312/stag.20241345 . No reproduction.
MoPE https://arxiv.org/html/2606.01258v1 ,definition1/section3.1,uses Gaussian
amplitude around an absolute origin and discusses its origin prior; this differs
from R56's conditional relative quadratic. No novelty claim for our combination.
He et al., https://arxiv.org/html/2601.02339v1 ,sections3.1.1/3.1.2,use measured
Gaussian covariance in regularized graph metrics,spectral shape descriptors and
center PE. That multi-component pipeline does not validate R56's learned region.
Point-GN https://arxiv.org/html/2412.03056v1 ,section3.2,uses fixed axiswise
Gaussian coordinate features in a non-parametric classifier,not full anisotropic
Gaussian Transformer PE. No paper scores are compared with our different split.

The following R55/R54/R53/R52 status entries precede R56.

Updated 2026-11-02: R55,R54,R53 and R52 are COMPLETE.
Full goal remains active and unachieved. No owned research job remains live.
Do not restart these completed rounds. Next scope:native3D center/region encoding.
R51 real visual matching COMPLETE, including the explicitly requested HRoPE
versus MLP comparison. The user explicitly resumed after the PRoPE discussion
and approved the previously described next plan through the active goal.
Full goal remains unachieved. The earlier pause entries below are historical.

Previous goal work made progress:16 R53 fits, matched references, bounded checks,
and the PRoPE/calibrated-plane comparison gap are retained. All old interrupted
processes were authoritatively terminal before resumption. R53 resume14324 is
now TERMINAL exit0(4f37e3):18 new fits,12 reused references,60 exports,30 checked
models complete. Analysis/plot/reference27272 TERMINAL exit0(bf4ada). No R53 job
is live; do not restart. Interrupted history is archived, no completed run replaced.
Read docs/round53_ray_frame_report.md and completion.json. SoftEPE:PRoPE4.2069,
ray-point4.0813,ray-iso4.1042,ray-full4.0611,ray-affine4.0635,hybrid4.1675.
Full vs PRoPE improves3.464%,3/3seeds/5scene means (two scenes nearly tied).
Full vs point only-.0201px,2/3seed and2/5scene wins; versus affine-.00236px.
No unique Gaussian benefit/promotion. All30 checks/60 metric replays passed
bounded impact review; warnings0, no winners changed. Unused confirmation remains untouched.

R54 sources: ROUND54_CALIBRATED_PLANE_PROTOCOL.md, models/round54_calibrated_plane.py,
tests/test_round54_calibrated_plane.py. Three bounded tests passed45786 exit0
(b0a308), including official PRoPE arithmetic with explicit chart/phase adapters.
This is a matched implementation, not an official training-pipeline reproduction.
Two new plane/offset roles x3seeds;21 identity-checked R52 references reused.
Nine real-input checks and21 reference identities verified88928 exit0(cf29ac).
Lock f0c936b2c16b80305c1e6ee7ecefe7997f80d51e41d6654d1a13d4bda869f94a.
GPU queue86598 is TERMINAL exit0(8105a9), all7 children exited0; CPU checker30710
is TERMINAL exit0(f0e9c3). Analysis/plot/reference84445 TERMINAL exit0(a28002).
All21 scientific sources,27 selected model identities and54 prediction exports
verified;42 reused reference exports equal R52 exactly. Read
docs/round54_calibrated_plane_report.md and completion.json. SoftEPE:P4.2069,
offset4.2781,plane4.5467,iso4.3951,GS4.4012,affine4.4123. Plane is worse than P
by8.079%,3/3 seeds and5/5 scene means; GS is worse by4.619%,also3/3 and5/5.
Filling the known-camera comparison gap did not reveal local-shape utility.
This does not establish PRoPE's universal superiority over HRoPE; the pipeline
and task differ from the original paper. All27 operator/CPU-validation checks
and54 metric replays passed bounded impact review,warnings0,winner changes0.
New6 fits/selection cost387.54sec. Source/data/model comparisons remain frozen.

R55 is COMPLETE under ROUND55_BUDGET_DIAGNOSTIC_PROTOCOL.md. Read
 docs/round55_budget_diagnostic_report.md and
 results/round55_budget_diagnostic/attempt1/completion.json.
Previous goal turn made progress:implemented/froze the fixed budget intervention
and launched GPU fits. This turn verified the live handles,completed24 fits/48
exports/24 selected-model checks,analyzed results,and closed the experiment.
No fresh confirmation accessed. The full research objective remains unachieved.

One matched extension only:eight unchanged R52/R53/R54 roles x3 paired seeds,
5000steps. The original1000-step init,batches,schedule,selected weights/functions
and validation histories reproduced exactly for all24 models. Subsequent4000
updates used constantlr.0002 with continuous optimizer state. All24 selections
preceded evaluation;shared initialization and complete sampling hashes matched.
34 scientific sources unchanged at lock
6bb56fc73932ef32b058c08612381008247f06759c84a3dd194d1701b1a84e7c.
All selected steps were2600..4900;none at5000,without claiming global convergence.
No5000-step NoPE/MLP run was performed;their old1000-step scores are not a fair
ranking comparator for this diagnostic. These are correlated development runs.

SoftEPE/PCK8%:P3.8636/87.3356,offset3.9148/87.0646,plane3.8789/87.3371,
forwardGS3.9185/87.4501,ray-point3.7065/88.0617,ray-iso3.7482/87.8231,
ray-full3.7344/88.0644,ray-affine3.6669/88.3119. Ray-full vs point changes
from-.0201px at1000 steps to+.0279px at5000;1/3 seed wins and0/5 scene wins.
PCK8+.00272pp is a tie. Full vs affine+.06748px,1/3 seed wins,0/5 scenes;
soft-tail+.12694pp is worse inall3 seeds. Full vs iso retains+.2413pp PCK8
(3/3seeds,4/5scenes),but point/affine prevent a unique-shape claim.
Full vs P retains-.1292px/+ .7288pp,3/3seeds,4/5scenes;point/affine also gain.
Do not assign the cross-route gain solely to Gaussian shape or inverse frames.

Plane vs P gap shrinks from+.3399px to+.0153px:the large short-budget penalty
mostly disappears;these are now near ties,not universal PRoPE superiority.
Plane beats offset by.0360px/+.2725pp PCK8 in3/3seeds,but not P consistently.
ForwardGS vs P gap shrinks to+.0549px,still0/3 softEPE seed wins.
RGB-permuted PCK8 rises with longer training:P75.605%,ray-point74.399%,
ray-full74.413%,affine75.149%. Full still loses13.652pp under permutation;
local appearance matters,but the test cannot distinguish stronger geometry,
retained image-level features or development-scene statistical memory.

Queue64824 TERMINAL exit0(d1c8f4),all16children exit0;CPU checker98385 TERMINAL
exit0(879491). All24 operator checks:maxP5.28e-8,maxV2.76e-6,winner changes0,
full CPU valCE error<=1.51e-7. All48 metric replays:PCK/tails exact,EPE<=2.22e-9,
warnings0. Same-author independent arithmetic,not another lab/full-network rewrite.
New training/selection cost7676.82sec,including checkpoint/prefix-check overhead.
Initial analysis87058 exited1(7cf399) due to stale MLP scene index;failed source
and record retained in postprocessing/attempt0. Corrected analysis91461 exit0
(63a9f6),final plot96297 exit0(1dbc45);both plots visually inspected. No science
source,selection,prediction or training changed. Completion72758 exit0.

DECISION:end tuning rescue of this local-frame route;do not scan more budgets,
features or matrix factorizations to manufacture a winner. Preserve its positive
center/generic-frame signals and negative full-shape result. No candidate promoted.
Next structural hypothesis:in native3D,retain center RoPE on content and express
conditional Gaussian region preference as extra positional Q/K channels in the
same dot product. No physical covariance attenuation of the center phase.
Use the R48 native point-cloud setting to remove camera/predicted-depth dependence;
compare each part alone,joint full/simpler/fixed-distance/free-quadratic forms,and
RoPE/MLP strong controls with identical data/backbone/budget. Only exact unchanged
references may be reused. R48's severe tilted-rotation failure remains a limitation.
The unexecuted R49 draft is background,not permission to resume its older projected
matching setup,60-fit plan or1pp guide. A new native3D protocol and identity checks
must precede training. No next-round code/protocol has yet been frozen or launched.

Objective: a compact, generally useful position encoding derived from 3DGS
parameterization, supported by matched strong controls, actual tasks and
untouched confirmation. No candidate has met the complete objective.

2026-11-02 archival PAUSED planning update (superseded by completed R54 above):
user requested reading the PRoPE discussion
in the branch titled "检索3DGS与RoPE结合进展 (3)" (thread
01a199fa-0796-7a40-9ba5-c084c53d3891). Read its latest two exchanges and checked
PRoPE arxiv2507.10496v2, official prope/torch.py (blob
1c450ccc05bca91a6b9ddbd19971437896ef83b5), and HRoPE arxiv2610.28284v1
sections3.3/4.2.2. No research job was resumed.

R52 camera_only is a PRoPE-style adaptation, not a new project contribution or
an exact reproduction of the official pipeline. Current HRoPE predicts its
R/t/plane transform; unlike camera_only, it does not explicitly insert the known
camera extrinsics into that transform. Identical available metadata does not
isolate the incremental benefit of local surface geometry. The missing matched
control is calibrated HRoPE: H=P F, F=[R_n,z*n;0,1], with known P retained.
The paper reports this P*F comparison separately from predicted3x3 HRoPE.
Common F cancellation is explicitly its reduction to PRoPE; R52's common-L
cancellation limits the whitening interpretation, not a proof of implementation
failure or a sufficient reason to prefer inverse frames.

The priority then proposed for resumption was to align/document the PRoPE camera chart,
units, rotary conventions and Q/K/V/output transport; fill the calibrated-plane
control under the same inputs, predictor feature source and training budget.
Compare P, P*plane (plus offset-only control), P*Gaussian (isotropic/full), and
P*affine; retain NoPE/MLP references. Reuse old checkpoints only if scientific
configuration and functions are unchanged. First question: can local plane or
Gaussian geometry add utility beyond the same known camera operator? HRoPE
winning is a mechanism reference, not a universal prerequisite or new gate.
Use paired seeds/scenes, continuous localization and tails; geometry strata must
be fixed independently of model errors. If needed, permit one bounded matched
budget/feature-source diagnosis rather than indefinite tuning. R53 partial fits
remain an independent frozen candidate for eventual original-protocol closure;
do not splice new controls into its frozen sources or let it dictate the route.
Native3D/cross-task utility and untouched confirmation remain required for the
general claim. This paragraph is a plan, not an executed experiment or approval
to resume the paused goal.

Latest scope clarification: R41/R42 rendering readout/loss results are task-specific
evidence and cannot establish general Transformer PE utility. R43--R45 test PE
inside the full two-block Transformer in projected 2D coordinates; R46/R47
test native 3D coordinates with matched anchor-geometry controls and no renderer.
The main question is whether the GS-form positional parameterization improves
the same Transformer against strong PE controls at a matched training budget.
Rendering is optional downstream evidence, never a prerequisite or substitute.
Cross-view matching is also one task, not proof of generality. Distinguish
generic Gaussian/RBF effects, learned center/orientation/scale benefits, and
effects of changed input geometry. Learning a PE covariance does not require
using a physical rendering covariance. Native 3D utility and cross-task utility
are not established as general Gaussian advantages; R46/R47's completed evidence
is recorded below. No historical result or criterion changed.

R48 is COMPLETE and NEGATIVE for its fixed primary development claim. Native
point-cloud classification OA:conditional88.622%,RoPE88.518%,MLP88.891%,free
quadratic88.642%; only the comparison with NoPE gives>=1pp and5/5seed wins.
No unique full-Gaussian benefit or promotion. The severe shared tilted-rotation
failure was checked with a bounded independent input/replay path; see
docs/round48_native_point_report.md. Controller24005 is TERMINAL exit0; do not
restart R48 training/export. All official-test/confirmation data remain unused.
R49 joint center-RoPE/Gaussian-prior code/protocol/tests exist as UNEXECUTED
drafts. They were not frozen, unit-tested or trained. Historical statements
below that say no R49 files exist predate those drafts and are superseded.

R50 local-frame mechanism experiment COMPLETE, ROUND50_LOCAL_FRAME_PROTOCOL.md.
The user discussed HRoPE's surface-deformation explanation and resumed the goal.
Test identical-input compact GS-frame, HRoPE operator adaptation, MLP, affine,
RoPE, isotropic/identity frames and Gaussian-bias controls inside the same
Transformer. Procedural multi-plane correspondence is a bounded development
test, not a renderer objective, real-data result or fresh confirmation. Both
exact and identically corrupted geometry are supplied to all methods; corruption
is a synthetic proxy, not a trained predictor. Full goal ACTIVE/unachieved.

R50 four bounded CPU tests passed: physical homography/differential labels,
independent nonzero matrix/transport path, shared initialization/label exclusion/
permutation, and shape gauge/finite-difference gradients. Generated fixed data
512fit/64val/128paired scenes for each of4 development conditions. Sources frozen
at lock SHA1c86fb2db11f38963441e23c3e8b162f6b986fdedb8a63cc8bf3922e45f7807a.
All48 trainings/384 evaluation groups complete; controller75996 TERMINAL exit0
(d95300), all6train+3eval children exit0. No owned live research jobs remain.
Postcheck49758,neighborhood30940,plot30171,zoom diagnostic21607 all exit0.
GPU devices are not exposed inside the sandbox; use host execution for CUDA,
observe current occupancy rather than reserving fixed device numbers, and never
modify drivers or interfere with another user's processes.

Read docs/round50_local_frame_report.md before continuing. R50 oracle AUC
four-condition means:GS57.311%,MLP59.561%,affine57.494%,HRoPE59.343%,
isotropic57.309%,identity56.811%,RoPE58.982%. Exact values in summary.json.
No unique full-GS advantage; zoom is a material weakness. Fine localization and
the39-scene/166-query severe-deformation subset do show local frame advantages,
but these are not general GS success. Base PCK8 is~100%; grid AUC ceiling77.998%
versusMLP77.510% makes the proposed+1pp base guide unattainable. Preserve the
old guide, explicitly stop treating it as a valid scientific rejection rule.

Fixed-checkpoint diagnostics changed the next decision: RGB token permutation
hardly changes metrics, while changing phase units/removing rotary V transport
damages fixed models. Exact centers+cameras let the task be solved geometrically;
R50 cannot independently verify appearance alignment. No trained operator bug:
all48 independent formula samples maxP3.28e-7,maxV1.57e-6,zero winners changed;
CPU validationCE max1.05e-7. GS axes did differentiate (mean ratio1.074-1.398),
so spherical-init zero orientation derivative is not proof of training failure.

R51 is COMPLETE: ROUND51_VISUAL_ALIGNMENT_PROTOCOL.md.
Ten methods x three seeds; 1000 updates each. Original SUN3D576fit/64val/480old
evaluation pairs; same RGB ResNet features/native positions/cameras for all,
without per-token3D centers/depth/normals/covariances. Full GS remains primary;
requested HRoPE/MLP comparison is explicitly reported. Two HRoPE forms: inherited
R50 adaptation and paper-form hard bounds/rotary convention. Official repository
still says Coming soon; these are our operator implementations, not the official
LVSM/VGGT pipeline or its reported benchmark. Shared backbone/features/frequencies
and QK normalization are declared adaptations. Full goal unachieved.

R51 lock812c939ec3d824f5e17da5bf1ec08147f08a782d40e2d6f332ed591beecac0df,
results/round51_visual_alignment/attempt1. All30 fits/60 exports complete; queue
9626 TERMINAL exit0 (9ee31b), all10child tasks exit0 on initially-only-idle GPU5.
CPU postcheck9616 TERMINAL exit0 (9fad4e); analysis/plot12476 exit0 (96081c).
No owned live research jobs. Do not restart these or any terminal R50 jobs.
Three bounded unit tests, all10real-input CPU checks and GPU smokes passed.
All30 selected operator samples: maxP1.63e-7,maxV9.01e-7,zero winners changed;
CPU validationCE max1.50e-7; all60 metrics replayed, largest EPE difference8.53e-9.
Shared initial states and actual batches matched; all learned PE tensors changed.
No fresh confirmation accessed; full scientific goal remains unachieved.

Read docs/round51_visual_alignment_report.md. PCK8 / softEPE: MLP79.331%/6.192,
HRoPE adaptation80.758%/5.993, paper-form79.996%/6.239, fullGS80.292%/6.107,
affine80.225%/6.108, iso80.210%/6.128, identity80.270%/6.079,
NoPE80.564%/5.845. HRoPE adaptation beats MLP in3/3seeds and5/5scene means
on both metrics (+1.427pp PCK8,-3.218%softEPE), with small extra gains over
identity; paper-form is unstable, and NoPE has better mean continuous error.
FullGS is near-tied with generic/simple controls, not a unique shape success.
RGB permutation reduces PCK8 by~63--72pp: genuine visual dependence in this
limited sense, not proof of physical plane/covariance recovery. HRoPE's gain is
mostly at15frame gap; at60frames softEPE10.253 versusMLP10.235,NoPE9.714.
All focal ratios1; no zoom test. Three seeds/five reused rooms are developmental.

R52 is COMPLETE under ROUND52_CAMERA_FRAME_PROTOCOL.md. The earlier
docs/round52_camera_frame_plan.md draft is superseded by that executable protocol.
P=diag(K,1) T^-1 retains camera depth in a homogeneous4D lift, with a fixed3m
unit; A=P G with canonical Gaussian root G. It is not an inverse2D projection.
Eight roles x three seeds: NoPE,MLP,smoothHRoPE,localGS(A=G),camera-only,
camera-isotropic,camera-fullGS(primary),camera-affine. The localGS control
isolates adding P; R51 used A=G^-1, so comparison to R51 alone would confound
the convention. Same R51 cached inputs/labels, common initialization/batches,
1000updates; no unused confirmation opened. Three bounded tests and all eight
real-input CPU gradient checks passed. Lock:
a92cb1b3d86db8e9a49bd7b78c136a64cfef8258fea9f09759c591ac256fe5ed.
Evidence results/round52_camera_frame/attempt1. All24 fits/48 exports complete;
controller16683 TERMINAL exit0(ba832b), all10child tasks exit0; CPU postcheck5902
TERMINAL exit0(1929cd), analysis/plot89375 exit0(a233fa), reference replay51439
exit0(17e7df). Do not restart these terminal jobs. No owned live research jobs.
Training used initially-only-idle GPU5; eval used later-idle GPU3/5. Future GPU
selection must inspect current occupancy. All24 selections sealed before export.
All24 independent NumPy operator samples: maxP1.63e-7,maxV2.23e-6,zero winner
changes; CPU validationCE error<=1.39e-7. All48 metric groups replayed; EPE error
<=7.58e-9px, PCK/tails exact, warnings0. All PE tensors changed. Nine repeated
NoPE/MLP/HRoPE checkpoints equal R51 tensor-for-tensor; not new independent trials.

Read docs/round52_camera_frame_report.md. PCK8/softEPE: camera_only87.841%/4.2069,
cameraGS87.457%/4.4012, cameraIso87.298%/4.3951, cameraAffine87.218%/4.4123,
localGS80.352%/6.0824, NoPE80.564%/5.8445, MLP79.331%/6.1923,
HRoPE80.758%/5.9931. Camera-only beats NoPE by7.277pp and28.021%softEPE,
with3/3seeds and5/5scenes positive. FullGS is worse than camera-only in3/3seeds
and all5scene softEPE means (+4.619%); iso/affine differences are small ties.
Do not rename camera-only as a Gaussian success. Camera benefit extends to60gap:
softEPE7.009 versusNoPE9.714,GS7.334. RGB permutation drops camera-only32.439pp,
GS36.244pp; visual dependence plus a strong camera prior, not physical-GS recovery.
No unused confirmation accessed; full goal remains unachieved.
Data-free algebra witness13930 TERMINAL exit0 (55ea94) records that common L
cancels from G_i G_j^-1 (max1.67e-16), whereas inverse-frame translation retains
whitening. Read docs/round52_operator_scope_note.md; this limits physical claims,
does not alter the frozen trial and is not a proven cause of task performance.
R51 already shows the task/metrics can register a
limited HRoPE adaptation benefit, but not difficult-geometry superiority or GS
generality. The SPD-frame pure-rotation limitation is a mathematical observation
(docs/round51_operator_scope_note.md), not an established cause of task errors.
Do not discard NoPE, inflate tiny ties into failures, or return to static repair.

R53 archival execution notes (completion above supersedes the pause state):
docs/round53_ray_frame_draft.md records the design origin; the executable
ROUND53_RAY_FRAME_PROTOCOL.md is now FROZEN and GPU training has started.
Three bounded tests passed again (48127 exit0,839afb); ten real-input checks
and twelve immutable reference-function/initialization/batch checks passed
(38056 exit0,058a77). Lock c19be86d01aae2f0928299da65b9414bca8ec40e5d4c322e6e9dbe9a31c46b5c.
User subsequently PAUSED the goal. Queue51887 TERMINAL exit143(4c5f73) and CPU
postcheck36170 TERMINAL exit143(dcd3c7) were intentionally stopped with SIGTERM,
along with owned training child65399. All three process exits were verified.
At that pause16/18 new fits and12 references were saved, without complete evaluation.
The subsequent authorized closure is recorded above. Preserve the interruption
history and do not restart completed jobs.
See results/round53_ray_frame/attempt1/user_pause.json; queue_status_before_pause.json
preserves the interrupted queue state. This is user interruption, not model failure.
Only currently idle GPUs0/1/3/5/7 were used at launch; future launches recheck.
A=(C G)^-1,mu=d*camera_ray,6D
Cholesky covariance+one depth; full-head4D transport primary, half-head native
rotary hybrid as a fixed control. This retains shared-scale whitening of center
differences and uses one main operator, but no utility is established. Include
the newly strong frozen R52 camera projection operator, not just NoPE/MLP/HRoPE.
The model factory has10roles:18new fits (pose-only,ray-point,ray-iso,
ray-full,ray-affine,ray-hybrid)x3seeds and12 explicitly reused reference models
from R52. Original training locks and records are retained; reuse is not new
confirmation. Scientific inputs/comparisons and bounded postcheck are frozen.
Preserve unused confirmation and the native3D/cross-task goal. The completion
report above supersedes these historical execution details and the former pause.

The user's route review exposed an unsupported dependency: success of an
untrained static region kernel had become a prerequisite to learning/GS tasks.
Early decoupled-shape synthetic gains are not established real-data success:
R2's shared-random-identity control substantially weakened that interpretation.
The R38 rank-one anchor correction vanishes for a true GS with a=mu, while the
underlying shape covariance remains nonzero. R13 documents region/center target
conflict but does not prove the finite tolerated empirical gates impossible.

Completed R40 experiment: `ROUND40_DECOUPLED_GPU_PROTOCOL.md`. Center3D RoPE plus one
scalar-weighted seven-dimensional CENTRAL covariance descriptor distance, with
center/expected/normalized, same-capacity center-distance, isotropic/shuffled
and all six R7 controls. Both views use the same predicted geometry. The primary
is fixed `decoupled`, not whichever ablation wins. 576/128 old TRAIN/VAL pairs,
13 roles x five paired seeds,1000steps. No fresh data authorization.

Evidence directory: `results/round40_decoupled_gpu/attempt1`. All65 GPU training
runs finished, both shards exited0. Independent full depth metric/identity
replay completed under `audit_impact/`: decoupled AbsRel .164532846 versus best
old point .161017277 (+2.1833%), center_rbf .161791281 (+1.6945%) and singleview
.162353858 (+1.3421%). Zero of five seeds beat any of those three controls.
Turning off shape at the fixed primary checkpoints slightly improves AbsRel.
No primary promotion and no result-driven frequency/temperature rescue.

The original audit stopped at a 2.1193e-5 P error versus2e-5 tolerance for a
point_pck sample. Retained it and used a separately recorded impact version;
only point_pck seeds11/44 warned, no winner changes, sampled depth change at
most1.399e-6m. Primary and all three decisive controls had no warning. See
`audit_impact/numeric_impact.json`; no retraining or prediction replacement.

All60 selected cross-view checkpoints completed old q24->k8 diagnosis at
`results/round40_region_diagnostic/attempt2`, with full metric replay. First
attempt's return-unpacking error and source/log were retained; corrected only
that interface. Primary KL2.11244/PCK44.579% versus full .544402/82.777%.
On773 target-consistent queries, pooled PCK47.684% versus full88.202%; failure
cannot mainly be blamed on center/region conflict. Full2304-query population
and1096center labels preserved. No source numerical warnings in region audit.

An additional warning about interpretation: point depth AbsRel improves when
V order is reversed (.1610173->.1598165). Do not equate lower residual-depth
error or nonzero gradients with correct spatial correspondence.

All R40 job handles are terminal. Complete report:
`docs/round40_gpu_research_report.md`. A matched short synthetic train benchmark
found CPU4 34.23ms/step versus GPU10.90ms/step (3.14x), excluding data/validation.
This is not the final cost gate or a general acceleration claim.

R41 actual-GS experiment IS NOW COMPLETED. Protocol:
`ROUND41_ACTUAL_GS_PROTOCOL.md`, evidence `results/round41_actual_gs/attempt1`.
Implemented direct RGB attention (no residual color head), actual R/S covariance,
context-only ResNet features/GS depth supports, same visibility/fallback as R9.
Scene-normalized units explicitly differ from metres; all Q/K retrained.
Eleven roles x five seeds x two scenes =110 runs,500steps, all predictions sealed
before target RGB. GPU4/5 shards exited0 in300.76/256.59s; peak837.875MiB.
Preflight4checks,CUDA smoke,110 independent score/RGB samples and full1328 frame
metric rows passed. Max P error8.93e-7,RGB5.04e-7,MSE replay9.72e-17; no warning.
All110 selected Q/K changed. Both main training handles14330/80742 are TERMINAL;
audit79756 and metrics92979 are also TERMINAL. Do not restart them.

R41 primary FAILED again: gate MSE .024874800 versus learned point .022275296
(+11.6699%), original simple point_h.25 .022241047 (+11.8419%) and original GS
.022219161 (+11.9520%),0/5 winning seeds for each comparison; both scenes worse.
Shape-off .024185743 improves2.7701%; reversed V .046824345 worsens88.2401%.
Learned full .022274579 is virtually tied with point, not meaningful success.
Complete report: `docs/round41_actual_gs_report.md`. R41 sources are frozen;
do not change them when implementing a successor.

R41 component diagnosis IS NOW COMPLETE. Queue91902 exited0 on GPU4; do not
restart it. Frozen ten-checkpoint, no-training scale-only / shape-only gate MSE:
.025186092 / .024093993. Shape-only improves3.13895% relative full but remains
8.16464% worse than learned point. Complete record:
`docs/round41_component_diagnosis.md`, `results/round41_shape_components/summary.json`.
Across481160 source observations (seed repetitions, not new scenes), full/off/
scale-only/shape-only mass on source-specific top-trace-decile keys is
.10625/.07394/.13373/.05347, and RMS projected-center displacement is
47.12/47.33/47.54/47.17px. Trace bins are not true uncertainty/edges; RMS is
outlier-sensitive. Raw scale similarity hurts these fixed models, but removing
it is insufficient. Max independent probability error9.63e-8, zero winner
changes. GPU full-context replay exactly matches saved selection MSE.

CPU context attempt2 also complete: full/off/scale-only/shape-only means
.02539809/.02502613/.02634465/.02451141. Waldo full shape helps center loss
(.0297972 vs off.0304115) but hurts image MSE (.0313877 vs off.0309577).
First CPU attempt stopped at too-tight1e-7 threshold; source/failure retained.
Maximum CPU/GPU MSE difference3.39e-7,72.5x below smallest shape-only/full effect;
retained numeric note, not a scientific failure. Both CPU handles terminal.

R42 matched objective control IS COMPLETE. Frozen protocol
`ROUND42_RENDER_LOSS_CONTROL.md`; full report `docs/round42_render_loss_report.md`.
Evidence `results/round42_render_loss/attempt2`:60runs,10seed shards, all training
process exits0. Three unchanged R41 models point/full/decoupled, center/image
losses,2oldscenes,5optimization seeds,100updates. Same multi-source fusion,
initialization, target schedule and four-context-image selection within R42.
Only within-round factorial isolates loss, not R41-to-R42 differences.

Attempt1 failed on Waldo view1 (one visible GS, zero jointly supported center).
Own Teatime process stopped; all sources/logs/three outputs retained, no metrics.
Attempt2 samples COMMON supported training targets for BOTH losses:
20Teatime/19Waldo. Original source pool, selection views, all8 evaluation views
and all pixels retained. Control is conditional on common supervision support.
Old handles25099/37946 terminal; freeze/preflight57578 and smoke95730 exited0.
CPU4 tests passed; GPU renderer directional derivative relative error1.72e-6.

Launcher43045 was stopped intentionally with exit143 AFTER nine launched shards
had exited0; pending Waldo44 was waiting on GPU0 occupied by other users.
Only that unstarted shard moved to idle GPU1; completion handle99399 exited0.
`launcher_reassignment.json` and `launcher_completion.json` preserve the chain.
All R42 training handles are TERMINAL, do not restart them. No live queue remains.
Audit8262 and metrics67324 exited0. All60 selected Q/K updated. Max independent
P/RGB errors8.80e-7/4.71e-7,zero winner changes/no warnings. All1088 frame metrics
replayed, max MSE difference9.72e-17. Peak allocated848.88MiB; sum per-run
training/selection wall time4373.40s, not exclusive GPU timing or elapsed time.

Gate means (point/full/decoupled):
center .022277209/.022275362/.024880107;
image  .022289946/.022193594/.024134251.
Image loss improves primary2.9978% vs center (5/5 paired seeds), but primary
still loses to image-point8.2742%,image-full8.7442%,old simple point8.5122%,
original GS8.6191%;0/5 winning seeds against EACH, both scenes worse.
Image zero_shape .023703017 improves1.7868%; reverseV .047161387 worsens95.4127%.
Image-vs-center closes29.14%/25.50% of primary-minus-point/full MSE gaps; this is
not a causal decomposition of R13 region/center conflict or all historical error.
Full improves only.4323% vs image-point, opposite scene directions, no promotion.
All10 image-primary best checkpoints are at100; convergence not established,
shape strength remains.953-.978. This is failure at the matched fixed budget,
not a theorem of impossibility under every budget.

DECISION: stop promoting/rescuing raw covariance-descriptor similarity. The
supervision mismatch contributes but does not remove the deficit; neither more
frequency/temperature sweeps nor renamed ablation winners are authorized as a
success claim. Keep the full scientific objective active. Fresh data untouched.

R43 COMPLETED TRANSFORMER SCREEN (negative; bounded diagnosis also complete): distinguish
using physical rendering covariance directly as matching support from using the
GS PARAMETERIZATION to construct a learnable PE. R42 tests the former's concrete
candidate, not the latter in general. Review learned frequency/basis alternatives
against direct priors before implementation. A grouped Gaussian frequency family
would need controls for fixed frequencies, isotropic/diagonal covariance, ordinary
learned affine position projection and free learned frequencies; its factorization
may just reparameterize existing learned RoPE. Do not claim novelty or positivity
from this sketch. Physical R/S utility and learned frequency R/S utility are
different claims, neither can silently substitute for the other. Any successor
must retain actual task tests and map applicability of prior geometry/robustness/
cost checks explicitly. No weakening final success criteria to fit a new branch.

The user explicitly restored active status after the pause. During the pause,
the root agent had mistakenly finished one R43 draft write, acknowledged it,
then stopped. That draft is untested, not evidence and not a frozen decision.
The inspected implementation passed four CPU preflight tests and all13 CUDA
toy roles (max independent P error2.071e-7, zero winner changes). Protocol,
sources and old data identities are frozen at
`results/round43_frequency/attempt1/lock.json`; its SHA256 is
83112dc07d8d521eddd409125a07c4662674959810d7236f3391f064e5f484a8.
This is a frequency-parameterization hypothesis with a fixed
two-block Transformer, not another RGB readout/renderer optimization. R5's
negative Transformer result remains valid for its particular formula. Physical
input-GS R/S utility and GS-style learned frequency parameterization are distinct
claims. Rendering is an optional downstream transfer test; it does not gate
all PE research. The R43 protocol records this clarification BEFORE results.
Initially all eight GPUs were occupied. After other jobs finished, a live query
verified GPUs1..5 empty (1..4MiB, no compute process). Five seed shards were then
launched:11/GPU1/handle90524,22/GPU2/65497,33/GPU3/59297,44/GPU4/72430,
55/GPU5/43086. Each executes13 matched roles,1000steps. Details and UUIDs are in
`execution_handles.json`; logs are `seed_*.log` in the attempt. No other process
was stopped. Re-poll these exact handles before concluding/restarting work.
All five training handles are now TERMINAL exit0, all65 runs completed.
Training audit handle77046 also exited0 and sealed all65 checkpoints. Maximum
independent sampled P error3.721e-7, no warnings or winner changes; matched
initialization, batches and earliest minimum-CE selection verified. All65
selected shared weights and all30 learned-frequency banks changed. Peak allocated
805.68MiB; summed per-run training/selection wall time3815.66s, not a throughput
benchmark. Source identities and the frozen experiment have not changed.
Old-evaluation exports launched AFTER this seal, one seed/GPU each:
11/99272,22/37154,33/4627,44/97316,55/52073; all now TERMINAL exit0.
Full metrics audit72668 exited0. All432 keys,110189 valid queries/model,480pairs
and primary/zero-V outputs retained. All62400 frame metric rows replayed; zero
PCK4/PCK8/valid-count differences. Max frame CE sum difference1.826e-4 and
argmax EPE sum difference.001359px do not change the PCK conclusion.

R43 main PCK8=.919973726 versus point=.923507882 (-.353416pp;1/5winning seeds,
0/5winning scenes). Against fixed_gaussian -.025840pp(2/5seeds),gaussian_iso
+.001762pp(2/5),group_affine -.033039pp(1/5),free_frequency +.003232pp(3/5).
All seven new frequency roles are within.0363pp of each other and below point.
Primary CE1.592955 versus point1.583625 (+.5892%). No promotion even without
appealing to the frozen1pp criterion; there is no positive point-control effect.
Old_eval zero-V primary PCK8=.617890309, supporting V-path use only.

One post-result bounded diagnostic was separately frozen in
`ROUND43_FREQUENCY_DIAGNOSTIC.md`, evidence `results/round43_frequency/diagnostic1`.
Only64old VAL pairs,20selected models (four roles x5seeds),50forward conditions,
no training: normal/zero_phase plus initial_bank for the two learned-bank roles.
Freeze33110 and run15696 both TERMINAL exit0. Full logits-to-NumPy metric
computation, normal replay PCK count differences0,maxCE frame sum diff5.724e-5.
Primary VAL normal/zero_phase/initial_bank PCK8:
.939920424/.485941645/.939219401. Free-frequency:
.940337249/.487286851/.940147783. The network clearly uses positional phases,
but restoring initial frequencies changes PCK8 only.0701pp/.0189pp at the fixed
weights. This is not a retrained ablation or an optimization-causality theorem.

Full report `docs/round43_frequency_report.md`; terminal handles recorded in
`results/round43_frequency/attempt1/execution_completion.json`. NO R43 job or
queue remains live. Do not restart any of the completed handles. Untouched
confirmation cohort remains unopened; the full research goal stays ACTIVE.

NEXT: retire promotion/tuning of this fixed global-frequency constraint. Review
the distinct, more direct hypothesis of learned SPATIAL Gaussian basis functions
as per-token positional vectors inside the same Transformer. Centers/covariances
would control local spatial responses. Compare ordinary RBF, learnable Fourier/
MLP and Gaussian Point Encoder prior work; no novelty or improvement presumed.
R44 is now COMPLETE and negative against the strongest point RoPE. Its local
advantages against additive-vector controls do not establish the full goal.

R44 protocol `ROUND44_SPATIAL_PE_PROTOCOL.md`, model
`models/round44_spatial_pe.py`, runner/auditor `experiments/round44_spatial_*.py`.
The preceding goal turn was concrete progress: completed and audited R43's65
training runs and50 diagnostic conditions, then retired its fixed candidate.
R44 tests spatial_full:36 learned Gaussian SPATIAL basis responses, one36x48
linear mixture, then normalize(h+p(x)) inside the unchanged two-block Transformer.
It is an application of known RBF/Gaussian Point Encoder bases, not a new basis
theorem. Learned precision is Cholesky SPD; covariance has GS R/S form. There
is no physical input-token covariance, renderer, RGB head or RoPE hybrid.

Eight new roles x5seeds=40new runs; six unchanged R5 roles'30 R43 runs/outputs
are reused with full identity hashes. New/old backbone initialization and1000
sampled batches must match; all new roles share projection initialization.
Added parameters full1908,diag1872,iso1836,fixed1728,fixed-centers1836,
Fourier-fixed1728,Fourier-learned1782,coordinate-MLP1836. A fixed native-grid RMS
normalizer matches initial basis energy. All Gaussian ablations start identically.
40 new checkpoints must be sealed before their old_eval export. Old datasets,
budget, selection and full query populations remain the same; fresh untouched.

Prospective utility criteria remain>=1pp versus strongest R5/Fourier/MLP controls,
>=4/5seeds and>=4/5scenes positive,CE<=2%worse. Component attribution is separate:
full versus fixed/iso/diag/fixed-centers is reported continuously, not forced to
independently earn an entire-method1pp margin. Any simplification requires a new
prospective experiment, never a relabeled ablation winner. This distinction was
recorded before R44 results; R43's frozen failure is unchanged.

Freeze/preflight handle40722 TERMINAL exit0. Five CPU tests passed, including
nontrivial independent NumPy formulas, all parameter counts/common initialization,
shape gradients at isotropy, input scope and 2D/3D exact Gaussian responses under
JOINT coordinate/basis transforms. This is not task robustness or3D task success.
Evidence `results/round44_spatial/attempt1`; lock SHA256
955125d409e88f410cad3c55542b9683c81a792355fcea65fcec4dc1427c5368.

GPU query initially found other processes on all eight devices. The bounded
controller `experiments/round44_queue.py` subsequently ran and is now TERMINAL
exit0 (tool handle55351; historical host PID121242). CUDA smoke exited0: all8 roles passed,
max sampled P error2.504e-7,zero winner changes. After GPUs freed, all five
training shards launched: seed11/GPU2/PID12779,22/GPU3/PID17975,
33/GPU1/PID19480,44/GPU4/PID19481,55/GPU5/PID22427. These are CHILD host PIDs,
not separate tool handles; the controller tool handle was55351. All40 runs,
training audit, allfive exports and full metrics are now complete. All13 child
exits0,active list empty,parent exit0 directly observed. Queue source, GPU UUIDs,
actual exits and logs are retained at `queue_status.json`/`queue.log`. No R44
queue/child remains active; do not restart or stop these historical PIDs.

R44 main PCK8=.898325531 versus point=.923507882 (-2.518235pp),0/5winning
seeds and0/5winning scenes; CE1.714961 versus1.583625 (+8.2934%). It exceeds
the additive Fourier-fixed/Fourier-learned/MLP controls by6.48896/5.22749/
1.64427pp,each5/5seeds and5/5scenes. Those local comparisons pass, but the
strong point comparison does not: no overall promotion. Component differences
full-minus-fixed/iso/diag/fixed-centers are +.010685/-.048260/+.036948/+.003293pp;
no stable evidence that learning full centers/ellipsoids drives the gains.
The equal-parameter isotropic role also exceeds the MLP, so a tiny parameter
count difference does not explain all observed local advantage.

All40 selected shared/projector/basis parameters updated.29/40 best checkpoints
are at1000steps: convergence is not established. New sampled probability error
max4.312e-7,zero winner changes/no warnings. All67200 frame metric rows replayed,
zero PCK4/PCK8/valid-count discrepancies. Peak664.12MiB; summed per-run
training/selection wall1716.81s, not a matched throughput benchmark.
Full report: `docs/round44_spatial_report.md`. Full goal remains ACTIVE and
unmet; the unused confirmation cohort is untouched.

The next-design review in `docs/round45_phase_warp_draft.md` has now become
the frozen R45 experiment (live status below). It uses u(x)=x+sum v_g*Gaussian_g(x), then original RoPE at u(x),
with v=0 recovering the strong original point baseline exactly. This tests
local Gaussian coordinate maps in a unitary PE instead of R44's additive
normalized vectors, not a width/budget sweep or a renamed ablation winner.
Required strong controls: ordinary affine/free-frequency and MLP/Fourier maps,
all sharing point initialization, plus Gaussian shape/center simplifications.
Review direct prior work before freezing. First-step zero geometry gradients
are expected at v=0; verify nonzero v gradient and subsequent geometry gradient.
R45 is now complete with no stable gain (details below), and it does not waive3D/cross-task, robustness,
cost or untouched confirmation requirements.

Training-time theoretical scope review is in `docs/round43_scope_note.md`.
A no-data/no-checkpoint 2D/3D counterexample
(`results/round43_frequency/frame_diagnostic.json`) shows that rotating only
frequency moments and rebuilding a canonical six-point Cholesky template is
not the same as rotating all finite frequencies. Analytic Gaussian and explicit
frequency transforms agree at~4e-16, whereas the reconstructed mean complex
phase differs by .770/.857 in the fixed examples despite correct first/second
moments. This limits automatic frame-equivariance claims; it is not a task PCK
effect or a new pre-training gate. No frozen source, template or method changed.

Related-work note `docs/round41_related_work_note.md` now includes 3D Gaussian
Point Encoders (2026), Elliptical Attention(2024), Learnable Fourier Features(2021),
and HARoPE(2026). Existing `models/gaussian_quadratic_pe.py` already implements
five-channel Gaussian attention; converting it to nine channels is not itself a
new design. Review bounded/finite-frequency limitations before more GPU runs.

Static region gates do not block these development mechanism experiments.
Actual Transformer task, full applicable scale/robustness/cost, attribution of
the proposed parameterization and untouched joint confirmation remain required
for the final claim. If claiming physical input-GS R/S utility, establish that
separately. Changes to
final acceptance standards must be explicit and prospective, not silent.

The old `round40_half_ray_pe.py` and its numeric auditor are unexecuted reserves;
they are not the active R40 definition. R38/R39 failures remain unchanged. Do not
edit README or other historical bound artifacts just to update current status.

R45 is now COMPLETE: `ROUND45_PHASE_WARP_PROTOCOL.md`,
`models/round45_phase_warp.py`, `experiments/round45_phase_warp_{train,audit}.py`.
This goal turn implements the preceding scope clarification rather than
repeating a status-only plan. The scientific question is Gaussian coordinate
maps inside the complete Transformer, with exact original-point initialization.
Eight new roles x5seeds=40 runs, plus30 identity-checked R43 legacy controls.
Added parameters full252/iso180/fixed72/fixed-centers180/Fourier126/MLP180/
affine4/free-frequency48. Same data,1000updates,paired initialization/batches,
selection and primary criteria as specified prospectively. No renderer or fresh
cohort access. The 2D screen does not prove3D or cross-task effectiveness.

Formula-level literature review found that HARoPE v1 acts on head-dimensional
Q/K CONTENT (R_x A_h q), not a2x2 coordinate warp. Our affine control is NOT its
reproduction. Gaussian/RBF bases and identity initialization already have
precedents; no new basis theorem or absence-of-prior-work claim is made.

Attempt1 retained: preflight59612 TERMINAL exit1, five of six tests passed,
one auditor role-dispatch typo caused KeyError before any training/real array
decoding. Corrected only dispatch and output directory; source/lock/failure
record retained. Attempt2 preflight62083 TERMINAL exit0, all6 tests passed,
including exact full-network identity for all40 role/seed combinations, nonzero
independent phase formulas, unit norms, mapped relative phase, gradient flow,
affine-frequency equivalence and input scope. CUDA smoke also exited0.
Completed evidence `results/round45_phase_warp/attempt2`, lock SHA256
f1276ec5e145fbcda1f19dba5979329e1c619a0db9c57ae6e74d50cc61c5426c.
Bounded queue tool handle54656 TERMINAL exit0, historical host PID11281.
All13 child exits0: smoke,5train,training-audit,5export,metrics. Active list is
empty; source and actual exits remain in queue_source.py/queue_status.json/
queue.log. Full40new/30reused model sealing and old_eval replay are complete.
Do NOT restart any of these historical handles/PIDs. CUDA toy max P error
1.589e-7 and all8 initial full-network differences0. Main independent sampled
P error3.224e-7,0winner changes/no warnings. Full67200 frame rows replayed,
zero PCK4/PCK8/valid-count discrepancies; maxCE frame-sum difference1.826e-4.
All40 selected shared and active positional parameters changed. Best-step
counts700:2,800:13,900:12,1000:13; no convergence claim. Peak677.77MiB,
summed training/selection1886.26s, not a throughput benchmark.

R45 primary PCK8=.923574821 vs point=.923507882 (+.006694pp),2/5winning seeds,
3/5winning scenes; CE+0.1990%,PCK4slightly lower. Paired point deltas in pp:
+.151354,-.005151,-.134141,+.070238,-.048831. The lack of stable useful gain
does not hinge on the1pp criterion or numerical auditing. Full-minus-Fourier/
MLP/affine/free-frequency differences +.041519/+.044458/+.042344/+.021440pp.
All new roles are within.04446pp. Components full-minus-fixed/iso/fixed-centers
are +.034240/+.031838/+.026562pp,4/5positive seeds each and5/4/4positive rooms;
report these small positive tendencies without substituting them for utility.
Primary zero-V PCK8=.596677892 versus normal.923574821.

Before old_eval exports, a bounded old-VAL phase-reset diagnostic was frozen
in results/round45_phase_warp/diagnostic1/plan.json, source
experiments/round45_phase_diagnostic.py. Freeze handle8735 and run30200 both
TERMINAL exit0. Six roles x5seeds x2conditions=60forward conditions on64VAL
pairs,3840frame rows. No training/tuning/new data. Normal replay CE and count
differences0. Restoring original point phases at fixed content weights gives
primary PCK8 .942610837 versus normal.943065555: learned-phase effect+.045472pp,
3/5positive seeds. Corresponding fixed/Fourier/MLP/affine/free effects in pp:
+.064418/+.159151/+.405457/+.164835/-.009473. Fixed-weight dependence is not a
retrained ablation, independent confirmation or evidence of baseline superiority.
NO R45 job remains active. Terminal record execution_completion.json and full
report docs/round45_phase_warp_report.md are authoritative. Fresh cohort untouched.

The review in docs/round46_native3d_design.md has become frozen R46 (status
below). Stop
promotion/tuning of R45's 2D candidate. Cover native3D coordinates inside a full
Transformer with matched point/non-Gaussian controls; do not substitute a
rendering task. Source inspection found R5 mean3 is a patch centroid whereas
native2D positions are exact token centers. A native3D experiment must control
center-lifting geometry separately (e.g. same center geometry with2D and3D
point controls); the code difference does not prove a cause of old failures.
No new arrays opened for this review. It also does not establish a second task,
general geometry/robustness/cost or untouched confirmation. Full goal ACTIVE/unmet.

R46 COMPLETE: ROUND46_NATIVE3D_PROTOCOL.md,
models/round46_native3d_pe.py, experiments/round46_native3d_{data,train,audit}.py.
The preceding turn made concrete progress: R45's40new trainings and60diagnostic
conditions completed and rejected its promotion; no R45 process remains live.
R46 tests native3D positions in the same complete Transformer, with center2d
and gaussian2d roles on the SAME newly center-lifted geometry. These isolate
the changed position anchor from the PE effects. The old point2D result remains
as a historical geometry control, not a falsely matched new-input baseline.

Center rays use existing RGB-predicted depth, sanitized/clamped before averaging
the four half-pixel-center samples. All864 points are expressed in the query
camera frame, centered and isotropically RMS-normalized, then multiplied by12.
Both attention directions share those3D coordinates. Input-only normalization
and no covariance/label mask in forward. Known cameras remain supplied metadata.
Same visual feature bytes, old labels,432keys and query population. 1605old
scene/frame records frozen; no unused cohort access. The3D PE's covariance is
a learned field parameter, not physical optimized-GS R/S.

55new trainings=11roles x5seeds plus5 identity-checked historical point controls.
Primary64Gaussian centers,full3D precision,zero64x3 coordinate mixture then
original-form unitary RoPE;768added parameters. Iso448/fixed192/fixed-centers576/
Fourier320/MLP448/affine9/free-frequency72/point3d0/center2d0/gaussian2d252.
All3D roles start exactly at a common3D point function;2D controls start at the
common2D point function. Shared neural initialization and1000sampled batches
match the prior baseline identities. Same1000updates/batch8/selection budget.
Primary criteria and continuous component reporting are prospective in protocol;
full3D/cross-task robustness/cost and untouched confirmation are still outstanding.

Freeze+preflight tool25108 TERMINAL exit0, all6tests passed. Frozen lock SHA256
45cdfe3e5d81792573a2c87964e78067a013e136bd1bf8ad539236ae1c382a69.
Current evidence results/round46_native3d/attempt1. Bounded controller tool
handle82629 is now TERMINAL exit0; exact historical host PID and launch identities
are in execution_handles.json. Do not restart it. It ran data construction/independent
geometry audit -> CUDA smoke ->5seed training shards -> checkpoint seal ->
old_eval geometry construction/audit ->5exports -> full-query metric replay.
Actual state/exits are in queue_status.json/queue.log and execution_completion.json.
All stages completed; scientific results are recorded below. Full goal ACTIVE/unmet.

R46 update: data child61213 exited0 and all576train/64val pairs' coordinate
arrays passed independent geometry reconstruction; maximum coordinate error
3.815e-6, visual input bytes identical. All11 CUDA smoke roles passed, maxP
1.681e-7,zero winner changes and exact same-dimensional point initialization.
Training launched on verified idle GPUs1..5: seeds11/22/33/44/55, child host
PIDs108770/108771/108772/108788/108789, respectively. These are historical controller
children, not tool handles; all have exited0. Parent handle82629 is also terminal.
Training launch records and UUIDs also saved in execution_handles.json.

R46 terminal result: all55new trainings+5historical controls sealed and exported,
all480old_eval pairs' new geometry passed, all57600frame metric rows replayed.
PCK4/PCK8/valid count discrepancies0; maximum CE frame-sum difference1.612e-4.
New sampled NumPy P error max3.669e-7,0winner changes/no warnings. All55 shared
and existing active positional parameters updated. Best steps500:7,600:6,700:6,
800:15,900:14,1000:7; no convergence claim. Peak774.835MiB, summed per-run
training/selection2417.218s, not a formal throughput comparison. Controller
handle82629 directly returned exit0;15children all exit0,active[]; no R46 jobs live.

Primary gaussian3d PCK8=.879149725 vs point3d .851576467 (+2.757326pp),5/5
positive seeds/scenes and CE-7.4948%. However mlp3d=.882505583: primary loses
.335586pp,0/5winning seeds,1/5winning scenes. Gaussian-minus-Fourier is only
+.060967pp,3/5seeds and2/5scenes. Thus the nonlinear-coordinate-map benefit
is real on old data but is not a Gaussian-specific advantage. Strong same-input
center2d=.924121336, so primary is4.497161pp lower,0/5seeds/scenes. No promotion;
this does not depend on the1pp margin or a numerical audit veto.

Center2d-minus-legacy anchor change +.061345pp,4/5seeds3/5scenes;
gaussian2d-minus-center2d +.006720pp,3/5seeds3/5scenes. Point3d-minus-center2d
-7.254487pp combines dimension/normalization/frequency configuration, not pure
dimension. Full-minus-fixed3d/iso3d/fixed-centers3d +.196522/-.076409/+.116501pp;
no stable full-ellipsoid advantage over isotropy. The center-lifting change does
not supply evidence that anchor/region conflict explains all prior failures.
Primary zero-V PCK8=.651677441; V-path dependency is not PE-shape attribution.

Report docs/round46_native3d_report.md. Stop promotion of this frozen global
Gaussian warp; no R47 candidate is frozen or launched. Subsequent design must
address what Gaussian centers/orientations/scales add beyond ordinary nonlinear
position maps inside a complete Transformer. Rendering remains optional; matching
is one task, not generality. Full geometry/robustness/cost, a second task and unused
joint confirmation remain outstanding. Fresh cohort untouched; goal ACTIVE/unmet.

R47 COMPLETE: ROUND47_CONDITIONAL_GAUSSIAN_PROTOCOL.md,
models/round47_conditional_gaussian.py and experiments/round47_conditional_*.py.
The preceding goal turn was progress: R46 completed all55 new trainings/old_eval,
independent metric replay and a terminal report, rejecting its promotion. No R46
process is live. R47 changes global Gaussian warps to query-conditioned Gaussian
positional channels inside the complete Transformer. One shared linear predictor
from the current normalized token outputs head-specific relative means and SPD
precisions; an exact quadratic Q/K lift uses5/9 positional channels in2D/3D.
No renderer, physical input covariance, rotary hybrid or final geometric shortcut.

This has direct Gaussian-attention/PE precedents (EMNLP2018 localness, ICLR2020
appendixE generalized quadratic PE, ICCV2021 SMCA); no novelty theorem is claimed.
The new question is whether conditional Gaussian constraints improve on ordinary
dynamic quadratic PE at the same capacity and initial first derivative. Six
families in2D/3D (conditional/diagonal/isotropic/static/fixed/free-quadratic) plus
NoPE give65new runs. Reuse30 fully checked R46 point/warp/MLP/Fourier controls.
Primary conditional3d remains fixed; secondary2D cannot replace a failing primary.
Same old data bytes, shared initialization,1000updates/batches and selection.
Separate full-vs-component attribution remains continuous; no new static gates.

Freeze/preflight handle73320 TERMINAL exit0, all6CPU tests passed, including65
paired initial full-network Gaussian functions, independent nonzero pair energies,
SPD/shape gradients, free-quadratic first-derivative match, input scope and
translation/permutation plus joint coordinate/precision identities. These are
implementation checks, not end-to-end generalization. Evidence
results/round47_conditional/attempt1, lock SHA256
0f1c841699d8f3f6b9df3f606ee4782c6aac6ebbe0e45bc1fe4d2886a438ab0a.
Controller tool handle13639 is TERMINAL exit0; historical parent/child identities
are in execution_handles.json and queue_status.json. Do not restart historical PIDs.
All controller stages completed: CUDA smoke ->5seed training shards ->seal ->
5exports ->metrics. Results below. No unused cohort access; full goal ACTIVE/unmet.

R47 CUDA smoke completed exit0 for all13roles: maxP1.563e-7,0winner changes,
all initial full-network fixed-Gaussian/NoPE reference differences0. Five training
children launched on verified idle GPUs1..5: seeds11/22/33/44/55,
PIDs40608/40609/40610/40612/40613. Controller host PID40250/tool13639 is the
historical handle; the children are not separate tool sessions. All65new runs
and the controller have now exited0; do not restart historical PIDs.

Bounded old-VAL input-geometry diagnosis was separately frozen before execution:
experiments/round47_geometry_diagnostic.py, evidence
results/round47_conditional/geometry_diagnostic1. Handle89911 TERMINAL exit0.
All64oldVAL pairs/10556valid queries/432keys, no fitting or R47 changes. Untrained
nearest-key PCK8: projected2d93.4255%,native3d55.3050%,native3d_XY54.8030%.
Dropping Z does not recover the loss. Label-weighted predicted-anchor residual
Z variance fraction87.70%,covariance condition25.70; these are not true3D errors.
Single old VAL scene, not general confirmation or a trained-model comparison.
docs/round47_scope_note.md records exact scope, direct Gaussian-attention priors,
and the algebraic fact that centered3D coordinates do not retain a fixed camera
origin's perspective projection. Same-input3D comparisons remain valid and all
frozen R47 rules remain unchanged. Native point-cloud classification is a possible
second task; only official metadata/literature reviewed, no data downloaded/read.

R47 terminal result: all65new+30reused models sealed/exported, all91200 metric
frame rows replayed with0 PCK4/PCK8/valid-count discrepancies; maxCE frame-sum
difference2.693e-4. New independent sampled P error3.309e-6,0winner changes,
no warnings. All65shared weights changed, all50models with learned position
parameters changed every positional tensor. Best steps400:1,500:1,600:3,700:6,
800:6,900:19,1000:29; no convergence claim. Peak655.376MiB, summed per-run
training/selection2597.096s. Tool13639 directly returned exit0,13children all0,
active[]. No main R47 job remains live; execution_completion.json is authoritative.

Primary conditional3d PCK8=.834798111 versus rope3d .851576467 (-1.677836pp),
0/5positive seeds/scenes; versus mlp3d .882505583 (-4.770747pp),also0/5 each.
No promotion. Full-minus-fixed/static/isotropic/diagonal3d +7.659334/+3.948869/
+3.437183/+.734369pp; all5/5seeds,scenes5/5 except diagonal4/5. Full-minus-free
quadratic3d +.744784pp,5/5seeds4/5scenes. Conditional ellipsoid structure has a
local effect, but does not beat the stronger methods. The failure is not a
floating-point audit veto or solely the1pp utility criterion.

Prespecified secondary conditional2d PCK8=.923258078 vs rope2d .924121336:
-.086326pp,2/5seeds2/5scenes,CE+.7915%. PCK4 improves+.850468pp,4/5seeds5/5
scenes; EPE improves-.168147px(-3.5837%),5/5seeds5/5scenes,with larger effect in
Harvard. These recorded metrics show a tradeoff, not blanket failure or a renamed
successful primary. Full2D beats free_quadratic2d by2.290998pp,5/5seeds/scenes.
Versus isotropic/diagonal2d effects only+.045777/+.022268pp. Full/free comparison
changes SPD plus parameter coupling; it does not isolate SPD as the sole cause.

Prospective synthetic CUDA cost probe source experiments/round47_cost_probe.py,
results/round47_conditional/cost_probe1. Freeze9095 and run44250 TERMINAL exit0,
GPU1 verified idle. Batch4,N432/1024,six selected models,five warmups+20rotating
repeats per condition. Conditional2d inference14.277/14.324ms vs rope2d9.581/
9.682ms (about1.48--1.49x); forward/backward38.744/38.596ms vs31.297/30.653ms.
N1024 inference incremental allocated244.5MiB vs90.5MiB; forward/backward
505.32vs468.38MiB. Small parameter count is not free runtime/memory. Shared-host
probe, not production throughput/task-length generalization. No probe jobs live.

Full report docs/round47_conditional_gaussian_report.md. Retire promotion of this
frozen primary; keep its genuine component and secondary fine-localization
signals. Next research direction: independent native3D Transformer point-cloud
classification, provisionally ModelNet40, with official train-only development
and official test reserved; no R48 spec frozen or data downloaded yet. Do not
keep patching matching frequencies/widths or let another task erase matching
limitations. Cost evidence is recorded above; applicable geometry/robustness,
cross-task utility and unused joint confirmation remain unmet. All existing
unused confirmation scenes untouched, goal ACTIVE.

R48 STARTED: ROUND48_NATIVE_POINT_PE_PROTOCOL.md and
models/round48_point_transformer.py. The user's concern about rendering-task
drift remains controlling: GS center/scale/orientation parameterization must
be tested inside actual Transformer positional channels against matched strong
controls. R41/R42 rendering results never establish that claim. R43--R47 returned
to that question; their negative results are preserved, not erased by a new task.

This turn made concrete progress: pinned public ModelNet40 HDF5 acquisition,
new paired four-block self-attention classifier, data/train/audit/controller
implementations, frozen prospective protocol and six passing synthetic tests.
The unchanged R47 conditional3D Gaussian positional form is primary; diagonal,
isotropic, static, fixed, equal-capacity free quadratic, RoPE, Fourier/MLP warp
and NoPE give ten roles x five seeds. XYZ content and all non-PE weights/training
streams are shared. NoPE still receives XYZ. No renderer or task-specific loss.
This global scalar-attention model is not a canonical Point Transformer replica.

Acquisition tool40422 TERMINAL exit0 (chunk28a56c), archive435335111bytes,
SHA256 f01b8189281fae5790e39deb9f3eca86e446b771bdc665c6ad05f28d039b20e7,
HF revision d5dc795541800feeb7a4b3bd3142729a0d2adf7a. The original Stanford
server timed out; pinned mirror is referenced by NVlabs research code. Published
mirror hash verified, no independent original-server digest. Acquisition opened
only ZIP metadata. Main freeze occurred before HDF5 decoding. Preparation opened
only five official TRAIN HDF5 files; official TEST members remain unopened.
Stratified declared split:7910 fit/965 val/965 dev_eval, all40classes, no byte-
identical clouds across partitions. All selected models must be sealed before
model dev_eval; its raw arrays were automatically partitioned during preparation.
Neither this development split nor five seeds constitutes fresh joint confirmation.

Main lock af09505eb0db1ea8becbe50da9857298b898f619637772a9737b63ea5a0d546f,
19 frozen sources; evidence results/round48_modelnet/attempt1. Freeze/preflight
tool46527 TERMINAL exit0 (chunk3bd625), six CPU tests passed. CUDA smoke child9316
and data child9549 exited0. All ten CUDA roles passed direct NumPy positional
checks: max probability error2.0628e-7,zero winner changes. Synthetic batch32,
N512 training-step medians:RoPE35.524ms,conditional61.894ms (5warmups/10repeats),
not task utility or production throughput. No scientific setting changed afterward.

Controller tool handle24005 is LIVE, parent hostPID9006; do not restart it.
Its actual child state is in queue_status.json and tool output; initial handles
are saved in execution_handles.json. Currently four seed shards were launched:
seed11/PID9725/GPU1,seed22/PID9726/GPU2,seed33/PID9727/GPU4,
seed44/PID9728/GPU5;seed55 pending. GPU3 was occupied by another job and skipped.
These child PIDs are not separate tool handles. Each role runs120epochs,batch32,
512points,common AdamW/warmup/cosine/augmentation/selection; eight GPU-hours per
seed shard maximum. Controller will seal50models,export prescribed development
variants and replay metrics. Read the LIVE handle rather than starting duplicates.
No R48 utility conclusion yet; all existing confirmation scenes and official
ModelNet test remain unused. Full research goal ACTIVE/unmet.

R48 continuation: controller24005 re-polled live (tool returned the same active
session, not a restart). Four seed shards are progressing through the first of
ten roles each, approximately epoch45/120 at the latest observation; seed55 is
still pending. No role-level comparison is valid from these unmatched partial
seed results. Main frozen sources/configuration remain unchanged.

Bounded literature/source check completed while training. At author commit
3e08aa732a81033f1c07c0b0533c334fe448e0d3, GaussianMamba3D replaces the patch
encoder, but retains a separate3->128->trans_dim MLP positional encoder
(Mamba3D/models/Mamba3D.py:590,604,717). Therefore its Gaussian encoder results
are not a direct test of Gaussian Transformer PE. The stored-mean covariance
gradient hook (PointNet/encoder.py:1058) also cannot simply be called the natural
gradient of our shared conditional predictor. Sources and limits are recorded
in results/round48_modelnet/preconditioning_scope1/result.json.

A data-free analytic counterexample was fixed and executed in
experiments/round48_preconditioning_scope.py (exit0,chunk8120bf). For3 scalar
Gaussian means sharing2 weights, pullback Fisher J^T F J=[[3,2],[2,6]], weight
loss gradient[3,1]. The true natural-gradient direction is[8/7,-3/14], whereas
preconditioning outputs before backprop gives[2,.75], not even collinear. At
equal exact KL1e-4, linear loss decreases are.0253546vs.0206474. This only
rejects an unjustified optimizer equivalence; it does not show that the output
hook is useless, explain R47/R48 failures, or require a new optimization gate.
No task data read, no R48 change, no new candidate or result promotion.

R48 progress: the first4/50 trainings are COMPLETE:conditional/11,diagonal/22,
isotropic/33,static/44. Each executed120epochs/29760updates; selected epochs
115/105/95/60 respectively under the frozen minimum-val-CE rule. All shared
weights and positional parameter tensors changed. Per-run training/selection
times1945--1982seconds, peak allocated1.490--1.509GB (decimal bytes), not a formal
inference benchmark. A first-completion check (tool5464 TERMINAL exit0,chunkd64b80)
verified output hashes, budget/selection and frozen19sources, then replayed the
selected validation CE and sampled first-layer positional probabilities. Maximum
probability difference1.9632e-6,zero winner changes; max val CE difference9.30e-9.
Snapshot:results/round48_modelnet/attempt1/early_training_check.json. This covers
four runs only, does not seal the full experiment or access dev_eval, and does
not rank methods across different seeds. The full planned audit is still pending.

Controller24005 remains LIVE with the same four training child PIDs; it has
automatically started diagonal/11,isotropic/22,static/33,fixed/44. Seed55 is
still pending. No R48 source/configuration change, no fresh confirmation access,
no method utility conclusion. Continue the existing handle rather than relaunch.

R48 reporting preparation COMPLETE while training continues: new standalone
experiments/round48_report.py renders all10roles, all5seed differences, all7
prespecified conditions, parameter counts and training/selection cost. Whiskers
are seed ranges, not confidence intervals; robustness plots use each method's
own base and accompany absolute OA tables. The reporter preserves the frozen
decision and requires all50sealed models plus terminal successful controller
state. It is not imported by the training pipeline and does not change any of
the19frozen scientific sources or the candidate/selection rules.

Synthetic-only rendering smoke10349 exited0 (chunkc18dd8), plots inspected;
small reporting refinements preserved that source/output and smoke21787 exited0
(chunk988fe1). results/round48_modelnet/reporting_smoke1 and reporting_smoke2
contain ARTIFICIAL FIXTURES, NOT task results; never cite their numbers as
experimental evidence. A direct incomplete-run guard check (tool25116) rejected
production reporting while queue24005 is live, before reading task metrics or
prediction arrays; report1 does not exist. When the main controller completes,
run the reporter with the existing Python and MPLCONFIGDIR=/tmp/grope-round48-matplotlib,
then inspect real plots and write the scientific interpretation. Goal ACTIVE.

R48 live progress: fifth complete run fixed/44 executed120epochs/29760updates,
selected epoch85, shared weights changed, zero learned PE parameters by design;
training/selection1305.58seconds. Total5/50 complete. Controller24005 was directly
polled live (chunkcbe4d0); seed44 now trains free_quadratic, and other shards
continue their second roles. Same four owned GPUs/PIDs,seed55 pending. No new
audit gate or model comparison from selection data; dev_eval and confirmation
remain unopened by the evaluator. Continue existing controller.

R48 data-identity check COMPLETE: experiments/round48_source_identity.py and
results/round48_modelnet/source_identity1. Prospective metadata-only plan,
exit0(chunk34b65e),0.54seconds: all9840 original mesh filename-stem IDs unique;
fit7910/val965/dev_eval965 have zero cross-partition shared IDs. All40class
counts match the declared manifest in every partition. Only five official-train
ID-list JSONs and shape_names.txt were decoded; no HDF5/prediction/test members
opened. This checks source-object names, not geometric near-duplicates with
different names, and does not change the split or add a model-promotion gate.

R48 progress:8/50 trainings complete. Four paired seed groups are now available:
11 conditional/diagonal,22 diagonal/isotropic,33 isotropic/static,44 static/fixed.
Actual initial shared-weight digests and full sampling/augmentation stream
digests agree within every group, and differ across seeds. All completed runs
used120epochs/29760updates; all19frozen scientific sources unchanged. Metadata
snapshot pairing_snapshot1.json (exit0,chunkc2656e); no predictions or dev_eval
read and no cross-method ranking. Full50-run audit remains pending.

GPU3 became genuinely idle; controller24005 observed two free samples and
automatically launched seed55,hostPID104142,UUID
GPU-bed502d0-6c7a-c66d-42b6-8aa08676eec7. The launch was directly returned by
the existing live handle (chunkc0b694); fixed/55 has started training. All5seed
shards are now active on GPUs1/2/4/5/3, pending[]. Other four child PIDs remain
9725/9726/9727/9728. Latest parent poll chunk1cf300 confirms live; do not restart.
Seeds11/22/33 now train isotropic/static/fixed,seed44 free_quadratic. Full goal
ACTIVE/unmet; no task utility claim or fresh confirmation access.

R48 verified-wait continuation: controller24005 directly returned LIVE again
(chunk87650a). Now10/50 trainings are complete; new free_quadratic/44 and
fixed/55 both executed120epochs/29760updates, with selected epochs110/70.
Their shared weights changed; free_quadratic/44 positional tensors changed.
All five owned seed shards remain active. Seeds11/22/33 continue isotropic/
static/fixed; seed44 now trains RoPE,seed55 free_quadratic. No restart, changed
scientific setting, development-evaluation ranking, or confirmation access.
Continue the existing controller through all50 runs and its planned audit/export;
the research objective remains ACTIVE and unachieved.

R48 interpretation clarification COMPLETE, before dev_eval: data-free
experiments/round48_control_scope.py, results/round48_modelnet/control_scope1,
tool57502 TERMINAL exit0(chunk6feaff). The equal-capacity free quadratic control
has precision diagonal in(-5/16,7/16) and linear coefficients in(-1/4,1/4).
Finite Gaussian raw outputs realize P=exp(2)/16 I=.461816I, outside the free
diagonal bound; free outputs realize P=-I/16, impossible for Gaussian SPD.
At a precision feasible for both, P=e/16 I, Gaussian mean(2,0,0) gives
b=(e/8,0,0)=.339785 on x, beyond the free linear range. Thus the bounded
coefficient families are not nested. The comparison changes nonlinear maps
and attainable ranges as well as SPD and center-precision coupling; parameter
count/initial-tangent matching does not isolate SPD. No trained coefficients,
task data or checkpoints were inspected. This does not show a bound is active
in practice, diagnose R47/R48 utility, change R48, or create an additional gate.
All19frozen sources unchanged. The standalone reporter now states this limit;
its calculation/plotting/selection code and earlier synthetic smoke artifacts
are unchanged. Controller24005 remains LIVE;11/50 trainings complete, seed33
has moved from fixed to free_quadratic. Full objective ACTIVE/unachieved.

R48 verified wait: live controller24005 polled repeatedly, latest chunk12187d;
14/50 trainings now complete. Additional fixed/33,isotropic/11,static/22,rope/44
all ran120epochs/29760updates and changed shared weights (metadata read chunk
cb1e90); selected epochs85/55/65/90. Current five live shards train static/11,
fixed/22,free_quadratic/33,fourier/44,free_quadratic/55. Original owned PIDs and
parent9006 unchanged. No utility ranking or additional gate; all50-model audit
and dev_eval remain pending, confirmation remains unused. Continue handle24005.

R48 verified-wait continuation: same controller24005 still LIVE (chunke1f441),
16/50 trainings complete. New free_quadratic/33 and /55 completed120epochs/
29760updates, selected epochs120/90, shared and all positional tensors changed
(metadata chunkbb01e6). A selected final epoch is not a convergence claim.
Current roles:static/11,fixed/22,rope/33,fourier/44,rope/55; five shards continue.
No model/configuration changes this turn, no dev_eval exports yet, no additional
audit or confirmation access. Keep waiting on24005; objective ACTIVE/unmet.

R48 verified wait: controller24005 directly returned LIVE (chunk3d5586), now
20/50 trainings complete. New fixed/22,fourier/44,static/11,rope/55 all completed
120epochs/29760updates, selected epochs65/85/75/90; all shared weights changed,
and learned positional tensors changed where present (metadata chunke6a0b1).
All19frozen sources still match. Current roles:fixed/11,free_quadratic/22,
rope/33,mlp/44,fourier/55. Five shards active, no dev_eval exports, no setting
change or fresh confirmation access. Full audit/evaluation and goal remain unmet.

R48 coverage review COMPLETE during training: source/report hashes and findings
in results/round48_modelnet/coverage_review1.json (chunkbe3302). R40 decoupled
AbsRel .164532846 versus point .161017277 is +2.1833% WORSE, not an improvement.
R40 already combined center3D RoPE with physical shape-descriptor distance, and
its center_rbf role combined RoPE with a learned scalar center-distance penalty.
That residual-depth model and its reverse-V anomaly do not establish full
Transformer utility. R44 replaced RoPE by normalize(content+Gaussian vector);
R45/R46 warped RoPE coordinates; R47/R48 Gaussian roles use unrotated content
plus conditional quadratic channels. These inspected roles do not directly test
unwarped center RoPE together with the conditional Gaussian spatial prior.

This leaves an UNIMPLEMENTED candidate question: tau*<R(x_i)q_i,R(x_j)k_j>
-.5*(delta-m_i)^T P_i*(delta-m_i), with the existing quadratic positional lift.
It could test complementary phase/content interaction and adaptive locality,
against each part alone, fixed/isotropic priors, free quadratic plus RoPE and
retained strong coordinate-map controls. A coverage gap is not positive evidence
or novelty; R40's simpler combination and all negatives remain. This is not a
frozen next round, new gate, or license to modify R48. Decide whether it is worth
testing only after the full current result. No new task arrays/checkpoints or
confirmation were read for the review; existing aggregate reports were read.

R48 live controller24005 confirmed again (chunk7e4838);26/50 trainings complete.
Additional rope/33,mlp/44,free_quadratic/22,fixed/11,fourier/55,fourier/33 all
ran120epochs/29760updates; selected epochs120/85/70/75/100/85. Shared weights and
learned PE tensors changed where present (metadata chunks3d531b/3ba163).
All19frozen sources still match. Current roles:free_quadratic/11,rope/22,mlp/33,
nope/44,mlp/55. Same five shards active, no dev_eval exports or confirmation
access. Full audit/evaluation and full research objective remain unfinished.

Focused primary-source literature review COMPLETE during R48 training:
results/round48_modelnet/literature_scope1.json (chunkf057bb). Four inspected
sources: BAM arXiv2505.22842v4, MoPE2606.01258v1, HeatTok2608.22485v1, and
Localized Gaussians for Point Clouds Correspondence2409.13291v2/STAG2024.
BAM already uses a generalized-Gaussian relative-index log-prior. MoPE's actual
experiment uses absolute-origin Gaussian-envelope positional vectors; Section7
explicitly leaves its proposed Morlet-RoPE validation for future work. Its small
single-seed EGA combination is not evidence for our conditional3D joint score.
HeatTok directly encodes Gaussian center/scale/orientation using blended phase
coefficients; our algebra C^2+S^2=1-2a(1-a)(1-cos(u-v)) shows this blend is not
generally unit modulus. This is not a measurement of its trained attenuation or
a rejection of its task results. The STAG paper mixes Gaussian-only geometry
heads with RoPE dot-product heads, and reports substantial noisy-data weakness;
it does not test the same-head conditional anisotropic joint prior here.
Do not claim novelty from our repository coverage gap, import these task claims
as our evidence, or conflate amplitude modulation with additive log-priors. The
review is bounded, not exhaustive; no external implementation was reproduced,
no new gate was added, and no task arrays/checkpoints/confirmation were read.
All19R48frozen sources remain unchanged. Finish R48 before choosing a new round.

R48 COMPLETE: controller24005 TERMINAL exit0 (chunkeb4af0), all50trainings and
all50exports finished; queue state complete/active[]. All14child stages exited0.
All training/export PIDs, including final train55/104142 and export55/76953,
are terminal. Observation cells1619/1623/1626/1630/1633/1636/1639/1642/1646 are
also terminal. Do not restart any old job. Report39867 exited0(chunkee8538),
plots inspected; results/round48_modelnet/attempt1/report1/report.md and
new docs/round48_native_point_report.md contain results/interpretation.

All50models ran120epochs/29760updates. Within-seed shared initialization and
actual sampling/augmentation digests matched; frozen selection rule held.
Shared weights changed50/50; all positional tensors changed in35/35learned-PE
models. Max sampled P error6.92553e-6,zero winner changes/zero warnings; max
validation CE replay error1.32635e-8. Full dev_eval replay50models*965objects*
7conditions gives350metric rows,zero count discrepancies,max CE error1.15086e-7.
Permutation max logits1.56164e-5,zero class flips. Independent formula paths,
not an independently authored complete network/lab reproduction. All19frozen
sources still match. Sum training+selection77257.650seconds; peak1615664640B;
shared-host timings, not a throughput benchmark. Two models selected epoch120,
so no universal convergence claim. Officialtest and old confirmation untouched.

R48 primary conditional OA88.6217617/MCA83.6360874/CE.443558689.
Means OA:diagonal89.0777202,isotropic88.5388601,static87.9378238,fixed87.9585492,
free_quadratic88.6424870,rope88.5181347,fourier88.4352332,mlp88.8911917,
nope87.4196891 (OA/MCA in percent). Conditional OA differences in pp:
vsrope+.103626943(3/5seeds),fourier+.186528497(2/5plus2ties),
mlp-.269430052(1/5),free-.020725389(1/5plus1tie),nope+1.202072539(5/5).
Vsdiagonal-.455958549(2/5),isotropic+.082901554(3/5),static+.683937824(3/5),
fixed+.663212435(4/5). No primary promotion or renamed component winner.
Near ties are not material failures, but no stable strong-control advantage
emerges even without treating the1pp development margin as a binary theorem.
Conditional mean train+selection32.78min vsRoPE19.28/MLP22.26. Generality unmet;
R47's same-3D-input matching negative remains and cannot be erased by this task.

R48 robustness:conditional OA Y90=88.725389,jitter02=88.476684,
points256=87.958549,points1024=89.139896,permuted=base. All roles collapse under
the prescribed Rz45@Rx30 condition to OA4.4974--8.0000%;conditional6.549223.
OnlyY-axis rotations were trained; no SO3 invariance claim. A large shared
failure warranted one bounded check, not a new gate or an angle/temperature
sweep. New experiments/round48_rotation_check.py and results/round48_modelnet/
rotation_check1 saved plan/source before checking task arrays or replaying models.
Tool64394 TERMINAL exit0(chunke96756),4.29seconds. All965olddev inputs match an
independent coordinate-wise rotation within1.06130e-7; orthogonality5.45877e-8,
det.9999999924, inverse error1.31837e-7, checked distance error1.63651e-7.
Up-axis tilt52.23875deg. Identity-hash-selected32objects,seed11,conditional/
rope/nope CPU replay matches saved base/SO3 predictions within6.91414e-6 logits,
zero class flips; inverse rotation restores base predictions in this sample.
No observed transform/replay defect; retain common orientation failure without
claiming its causal training explanation. No retraining,new task estimate,
fresh data or original-result replacement. Diagnostic checked the frozen inputs
and19scientific sources again; all unchanged.

NEXT: decide/freeze a bounded joint-phase/locality experiment on existing
matching development data, keeping the full objective and original negatives.
The exact question is whether center RoPE's content-relative phase and a learned
Gaussian spatial preference complement each other inside one Q/K score. The
existing quadratic lift supplies9extra channels in3D. It is a hypothesis, not
an implemented/proven method or novelty claim. Compare both parts alone,
RoPE+fixed/compact Gaussian priors,RoPE+equal-capacity free quadratic and the
strong MLP/Fourier/NoPE controls with matched budgets/initialization/data.
The full-Gaussian/free comparison includes nonlinear/range differences, not
isolated SPD. A gain shared by a fixed prior is not evidence for adaptive GS
shape. If there is a stable matched result, validate the same formula on native
point clouds and then untouched confirmation; do not relabel old dev as fresh.
No R49 protocol/model/training exists yet. No new module/loss/renderer is needed
for this proposed score change. Goal ACTIVE/unachieved, no active R48 jobs.
