# Round15：旧验证回归蓝图（不是运行授权）

2026-10-30。依据已冻结的 [R15 候选合同](../ROUND15_CANDIDATE_PROTOCOL.md)、
[继承验证清单](round11_validation_inventory.md)、
[R12 尺度/扰动合同](../ROUND12_STRESS_PROTOCOL.md) 和
[R12 实际 V 合同](../ROUND12_DEPTH_PROTOCOL.md) 整理。
本文不修改任何已冻结源码、结果或门槛，不实现、不冻结、不启动验证程序，
也不读取当前 Stage B 区域分数。当前 Stage B 仍是原定八类方法各十二配置；
本文新增的归因对照不进入其校准网格。

进入实施前必须已有完整 Stage B 独立审计通过、选参文件及来源链，随后另行
冻结旧验证协议、前向、标签适配、计时、分析和独立审计源码。Stage A 的可行
集合不是最终配置；此处不假设哪个配置会被 Stage B 选中。任何前置失败均
保留，不能因本蓝图已经写好而视为获得效果实验或新场景数据访问授权。

## 本次在新验证效果之前作出的两个设计决定

1. 在正式旧验证冻结时，**新增 zero 状态 g16→g8 的 PCK@8 保护**：相对两项
   新 point localization 参考及 legacy point_pck，分别最多下降 0.5pp。
   这是新增保护，不能写成 R12 原有门槛。对应概率已经属于原九条件，无须
   新数据、额外参数搜索或新的前向条件。
2. 增加 `quartic_isotropic_matched` 归因角色，使用 primary 相同的均值、h、T、
   内容、原生 key、积分、floor 与全部前向信息，仅把其未加 h²I 的查询协方差
   C 改成 `C_iso = trace(C)/2 * I`，再构造 `S_iso = C_iso + h²I`。
   它不独立校准、不参与 strong point/quad 选择、不替换 primary、不因验证
   结果好而成为另一个本轮赢家。它检验固定 h/T 下方向形状信息的贡献，
   不能证明优于另行充分优化的所有各向同性方法。

若 matched isotropic 匹配或超过 full 的质量/成本，不应归功于方向形状。
本蓝图没有新增一个“full 必须优于 iso 多少”的未授权数值验收门槛；归因证据
与既定主验收分别报告。即使主验收通过，缺少方向收益也限制方法解释。

## 人口、角色和严格去重

固定原 32 个旧验证 pair：Harvard c8 与 Brown bm4 各 16。它们已经反复参与
开发，只能称开发回归，不能称独立确认。不得从本次分数重新选择 h/T、角色、
条件、场景、pair、可见人口或迭代步。

共有 **18 个语义 alias**：

| 组别 | alias 数 | 来源 |
|---|---:|---|
| primary | 1 | Stage B 在 Stage A 可行集合内按固定 J 选定的 quartic_full |
| 区域控制 | 7 | quartic_point；Gaussian point/fixed/full1/full4/quad4/quad16，各自无约束最小 J |
| 新定位参考 | 2 | Stage A 分别选定的 Gaussian point、quartic point 全局定位配置 |
| legacy | 7 | 下表冻结配置，既有比较不得被新参考替代 |
| matched isotropic | 1 | primary 的 h/T，仅作归因 |

| legacy alias | family | h | T |
|---|---|---:|---:|
| point | Gaussian point | 1 | 5 |
| point_pck | Gaussian point | .5 | 5 |
| fixed | Gaussian fixed | .5 | 5 |
| full1 | Gaussian full1 | .25 | 0 |
| full4 | Gaussian full4 | .5 | 5 |
| quad4 | Gaussian quad4 | .5 | 5 |
| quad16 | Gaussian quad16 | .5 | 5 |

所有具体配置均以已审计选参/legacy 来源链为准，表格不能替代哈希核验。
new strong point 是两个已校准区域 point 控制中的训练 J 较强者；new strong
quad 是已校准 quad4/16 中较强者。它们只是已有角色的 alias，不增加搜索。

只有 **family、h、T、完整数值与内容合同均相同**的 alias 可以共享概率和
计时。所有比较名称仍保留，不把等价角色当成重复证据。不能因为某个 pair
偶然输出相同，就跨 family 去重。各向同性归因角色默认是独立 family，
即便部分输入的 C 恰好各向同性也不与 primary 合并。因此实际唯一配置数
**U∈[9,18]**；最终冻结前由真实已审计配置生成明确的 alias→算子映射。

历史 R12 iso4 是不同的高斯混合归因对照，其旧报告保留；它既不是这里的
quartic isotropic，也不是本轮新增的第九个校准族。

## 条件和标签

每个 pair、每个唯一算子仅有 **68 个不同前向条件**：

- zero 状态下 g8/g16/mixed 的 3×3 原九条件；
- 五个尺度条件：g8→g24、g24→g8、g16→g24、g24→g16、g24→g24；
- 六个单独扰动，各自原九条件：深度×.9/1.1、目标相机 Ry±1°、目标 x 平移±.01m。

zero 原九条件同时用于原始分区验收和 zero 扰动检查，不重复保存或计票。
只改变前向几何；传感器标签、评价相机、有效群体和 h/T 不变。深度缩放、
变换乘法方向、米制单位、finite fallback 等完整继承 R12，不组合扰动。
当前 key 是原生图像 cell，不把未参与算子的 key-depth 缩放说成双侧深度敏感性。

原九条件和每个扰动使用固定 common16-query→base8-key 分布；五尺度条件
使用各自原生硬 cell 计数。q24 不进入 common16 父子聚合。共同网格变换只
依赖面积/分区，不接收 GT 可见性或计数。区域指标与中心指标分别保留人口，
区域 KL 不能替代中心 PCK。

旧 R12 已独立审计的 q24 中心标签可以在全部来源、pair、半像素定义、相机、
sensor timestamp、label hash 和人口精确绑定后复用。其科学门槛失败不使正确
的固定标签失效，但不能因此继承其候选效果。g8/g16 标签继续原独立中心投影
定义，不能用投影区域均值代替。冻结时可读来源元数据和既有期望 hash；新
验证的所有前向阶段完成后才读取标签字节。

## 验收交集：原门槛与新增保护分开记录

区域 KL 在 scene/condition 内按可见 source-pixel 质量加权，再对条件和两个
场景等权。PCK 对两个场景等权，使用原 native argmax（首索引打破平局），
不换中心解码器。PCK 由整数 correct/valid 计数构造有理数，精确比较 1/200
边界，不额外添加浮点宽容。

| 检查 | 必须同时满足 |
|---|---|
| 原九条件区域效用 | 宏 common KL 对 new strong point 和 legacy point **分别至少改善 2%**，每个比较的两个场景都改善 |
| 原九条件准确率/成本 | 对 new strong quad 和 legacy quad16 **分别**满足：KL 至少改善 2% 且增量时间≤1.25×，或 KL 恶化≤1% 且增量时间≤.5× |
| 原九条件逐条件保护 | 每个条件的等场景 common KL 相对上述两个 quad 各自恶化≤2% |
| g8→g8 定位 | zero 下对两项新 localizer 和 legacy point_pck，分别下降≤.5pp |
| **新增 g16→g8 定位** | zero 下对同三项定位参考，分别下降≤.5pp；明确不是 R12 旧门槛 |
| 五尺度区域效用 | 五条件宏 direct KL 对 new strong point 和 legacy point 分别至少改善 2%，每个比较两个场景都改善 |
| 五尺度逐条件保护 | 每个条件 direct KL 相对两个 quad 分别恶化≤2% |
| g24→g8 定位 | 对同三项定位参考分别下降≤.5pp |
| 六扰动及 zero | 每个状态的九条件宏 common KL 相对同状态两个 quad 分别恶化≤2%；g8→g8 PCK 相对同状态三项定位参考分别下降≤.5pp；全部概率有限且归一化 |

R12 和已冻结 R15 的扰动硬门槛是 **每状态九条件的宏 KL**，不是每个
stress/partition 都硬性≤2%。逐条件值仍完整报告，不事后更换这一区别。
full1/full4/fixed/quad4/quad16 与 isotropic 的质量、定位和时间均报告。
不能用较弱的新参考抹去任何 legacy 比较，也不能让最后的平均值掩盖失败项。
每个预先声明的实验仍完整完成、保存失败，不按途中观察的分数删减。

若更简单的 full1/fixed/full4 在质量与成本上同时占优，即使形式化 gate 为真，
也缺少保留 quartic 闭合的证据。此科学解释不应伪装成某个历史协议已预设的
新数字阈值。matched isotropic 只影响方向形状归因，不能事后改主候选。

## 同轮 CPU 公平成本

五个旧开发 scene 各取 first metadata pair，原九条件、全部 U 个唯一角色。
CPU 四线程，三次 warmup、十五次实测；同一计时进程配对执行，角色顺序采用
冻结的循环轮换规则，记录主机负载与每次原始时长。不得多个计时进程并行
争用 CPU。每 scene/condition 先取完整时长中位数，再对 45 格等权；门槛使用
这个比值，不能把 prepare/operator 两个中位数相加替代完整时长中位数。
五尺度若额外计时单列，不混入既定原九条件的主时间权重。

从公共已投影 xy/pixel_cov 与相同 feature tensors 开始，计入实际角色所需
统计、content pooling、key geometry、积分、floor、内容路径、softmax 和
common aggregation。不能把 Stage B 一次准备所有方法的结果放到计时区间外
供候选使用，也不能用纯积分、缓存 geometry 或校准吞吐代替完整增量成本。

| 算子 | 独立执行真正需要的统计 |
|---|---|
| quartic_full / Gaussian full1 | 全部像素的均值和完整二阶矩/协方差 |
| quartic_point / Gaussian point | 均值，查询协方差为零；不计算未用的完整 scatter |
| Gaussian fixed | 均值及原生 query 面积；协方差为 area/(64×12)I |
| Gaussian full4 | 固定四个分层的均值、局部完整 scatter 与平均 pixel covariance、权重 |
| Gaussian quad4/16 | 分层均值、平均 pixel covariance、权重；不计算未用局部 scatter 或全局 covariance |
| quartic_isotropic_matched | 与 primary 相同均值及完整 C 的 trace，等方协方差为 trace(C)/2 I |

isotropic 的 trace 可以通过完整 C 后取 trace，也可以直接累计
`E[||x−μ||²] + E[trace(C_pixel)]`，数学上等价。实际采用哪条代码路径必须在
验证/计时前冻结，独立数值校验后报告；有限精度归约未必 bitwise 相同，不能
只因公式相同就假定概率 hash 等同。无论选择哪条路径，都必须计入其真实所需
统计，不能免费借 primary 的 C，也不能为使对照显得昂贵而强加未用统计。
该实现选择不改变 h/T 或引入新校准。

quartic 使用原生 key 矩形后由固定 GL24 wrapper 切成边长≤1参考单位的数值
子格；Gaussian 保持原 R12 的 native24→四个12px积分矩形规则。分块不成为
额外 token。T0 按实际冻结 forward 路径计时，不能仅在计时中偷省内容计算；
当前实现仍执行内容路径。所有角色都使用其实际配置和 dtype。

共享 RGB/depth predictor、dense projection 和 transfer 不计入这项增量门槛，
但相同五个首 pair 的公共 CPU 投影墙时、导出总时间另报。记录 tensor bytes、
CPU RSS/峰值测量方式，不能把静态 tensor 字节写成 allocator 峰值。结论仍是
增量 PE 成本，不是整套视觉系统的端到端速度。

## 存储与完整前向先行

每 pair/role 的原生概率 scalar 数为：

    7×(432+108+243)²
      + (432×48+48×432+108×48+48×108+48²)
    = 4,345,767。

float64 primary 全部 32 pair 需要 **1,112,516,352 bytes**（约1.036GiB）。
若另存 int32 argmax，仅增加 2,192,256 bytes；共同网格概率可从原生概率确定性
计算，不重复保存。所有 U 角色全存需 U×1,112,516,352 bytes，U=18 时为
20,025,294,336 bytes，超过当前可用容量，不能默认为可行。

建议 primary 全保存；controls 所有概率先生成 canonical hash commitment，
全部前向封存后精确重放才计算指标。另预先保留两个旧验证 scene 的首 metadata
pair、每个 control 的全部68条件，每 control 样本 **69,532,272 bytes**。
因此保留原生概率的上界公式为：

    1,112,516,352 + (U−1)×69,532,272 bytes。

| 唯一配置数 U | primary 全量 + control 固定样本 |
|---:|---:|
| 9 | 1,668,774,528 bytes（约1.554GiB） |
| 18 | 2,294,564,976 bytes（约2.137GiB） |

还需预留标签来源、完整 metrics/hash、数值审计、计时和来源快照空间。本文
编写时磁盘读数约9.6GiB可用，Stage B 仍在运行；正式冻结前必须按真实 U 和
当时空闲空间重算，不能承诺当前容量一直存在。压缩收益不计入保守预算，
旧失败与已冻结证据不得为腾空间被删除。

两场景、全部角色、68条件都完成首遍前向并封存后才允许解码标签。primary
从保存的 float64 数组读取；control 由同源同 runtime 精确重放，每条输出在
计指标前必须匹配 canonical dtype/shape/bytes hash。重放失败立即保留失败并
停止该结果的验收，禁止四舍五入 hash 或丢掉问题 pair。

旧预测只有在输入、角色、配置、积分/内容实现、dtype、floor、人口和逐数组
hash 都完全相同时才可复用；旧 GPU 与新 CPU 不能仅凭同 h/T 假定相同。
最简单的初版是全部重算预测，只复用来源明确的固定标签/元数据。所有 alias
到数组/计时的映射在冻结前确定，不能以结果相近为理由事后合并。

## 即使这轮通过，仍阻断通用主张的缺项

- **真实 V 深度：** R7 失败没有被修复；R8 原 source-value 瓶颈仍是风险。
  需要新候选专属、先冻结的原任务：41,345参数头、576train/128oldval、五配对
  seeds、真实 attention@V、同容量 singleview、强 position 与 compression
  对照、完整原门槛及增量 PE+实际 V/head 成本。T0 的有效 Q/K 梯度为零必须
  披露；注册参数量相同不等于有效容量相同。
- **实际优化 GS/RGB/NVS：** RGB 预测 patch 的 μ/C 不是实际 optimized primitive
  内部 R/S 收益的证据。需要物理含义明确的适配、context-only 输入、原GS与
  point/fixed/isotropic/bilinear 强控制，按 R9 的 MSE/双场景要求重新验证。
  R9 失败不能因区域分布通过而注销。
- **表示与归因范围：** matched isotropic 仅检查固定配置下的方向形状；
  log(native quartic integral) 不是点 log-density 的十四维精确 Q/K 恒等式。
  不声称精确 query 拆并一致、一般 positive-definite kernel 或低维注意力实现。
- **独立泛化：** 两个反复使用的旧场景不支持总体置信区间或新场景成功声明。
  全部必要开发/真实值/GS检查有可信证据后，才另锁未触碰场景的联合确认。
  inventory 中未来确认的 known-size point/fixed 跨尺度保护、配对场景 CI、
  各尺度和扰动非劣区间仍未完成；不能用本轮区域平均数代替。

若最终只有区域匹配成立，应明确交付有限范围的区域 PE 结论；原来“具有
一般性、结构精炼、效果可信”的研究目标仍未完成。
