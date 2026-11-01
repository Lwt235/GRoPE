# World-space compact PE 独立审查

日期：2026-10-30。审查对象是 `models/gaussian_world_pe.py`、
`experiments/region_pe_probe.py` 与对应 CPU tests；这里判断实现、归因和
论证边界，不把正在进行的 development 结果当成独立确认。

## 审查结论

1. `world_moment_rope` 确实实现了 48→48 维、逐 token、保内容范数的 Q/K
   rotary 位置变换，没有 pair gate 或附加几何 bias。它的机制就是把高斯参数
   化为一个 covariance 描述子，再对描述子使用 RoPE；不能仅据此声称新编码数学。
2. `world_isometric` 确实实现了 48→96 维、逐 token、保范数的 Expected-RoPE
   等距补全。它与密度 BC 核不是同一算子，保范数也不能推出几何含义必然正确。
3. 已实现的三个新增 compact controls 是 `world_isometric_plane`、
   `world_isometric_isovolume`、`world_center_half`。前两个在原始观测所确定的
   共同单位中改变 covariance，后一个保留与 moment/normal RoPE 一致的
   24 个中心通道和 24 个内容通道，控制是有效且必要的。
4. `world_density_softmax` / `world_density_linear` 使用逐 pair 的精确 BC。
   这可以作为已知无限维密度特征的隐式计算，但当前实现没有生成一个小维度
   per-token 密度向量，不能拿 zero-shot exact-BC 的成功替 compact PE 完成验证。
5. 区域 soft-label 任务条件于有标签的有效候选集合，排除了 dustbin。
   当前结果只能说明这一条件分布任务，不能评价无对应识别、遮挡拒绝或 dustbin 校准。

## 实际实现的算子

### Gaussian moment RoPE

对每个 pair，先计算所有有效 q/k support 的共同长度单位

\[
u=\sqrt{\operatorname{median}_{q,k}\operatorname{tr}\Sigma},\quad
\bar\mu=\mu/u,\quad\bar\Sigma=\Sigma/u^2.
\]

定义保 Frobenius 内积的上三角向量 `svec`，以及

\[
d(\bar\Sigma)=
[\operatorname{svec}(\bar\Sigma/\operatorname{tr}\bar\Sigma),
 \log\operatorname{tr}\bar\Sigma].
\]

这是 7 个坐标，含有一个 trace 约束，仍只有 covariance 的 6 个自由度；它是
完整 SPD covariance 的可逆描述子，不是一个新增统计量。模型将 48 维内容分成
24+24 维，分别用 12 个三维中心相位和 12 个七维描述子相位旋转：

\[
P_G q=
[R_{\Omega_c\bar\mu}q_{:24},\ R_{\Omega_d d(\bar\Sigma)}q_{24:}].
\]

优点是结构简单、无 eigenvector 导数、球形 SPD 处可微、轴符号/轴排列
gauge 安全。有限固定方向的相位坐标依赖世界坐标轴，不是精确 SO(3) 不变的。
描述子 RoPE 产生周期相位差相似性，并不等于高斯密度重叠或区域覆盖概率。

它相对 `world_rope` 同时改变了中心编码的通道数，不能仅靠这一个对照把收益
归因于 covariance。`world_center_half` 精确控制了通道分配，必须报告。
`world_normal_rope` 使用不同维度、不同尺度分布的描述子和另一组固定方向；
moment 优于 normal 时，仍需承认这不是仅移除一种几何自由度的严格消融。

### Expected-RoPE 等距补全

每个二维内容块 v 对应一个频率 ω，使用

\[
t_G=\omega^T\bar\Sigma\omega,\quad
a_G=e^{-t_G/2},\quad b_G=\sqrt{1-e^{-t_G}},
\]

\[
P_Gv=[a_G R(\omega^T\bar\mu)v,\ b_Gv].
\]

独立展开得到

\[
\langle P_iq,P_jk\rangle
=a_i a_j\langle R(\theta_i)q,R(\theta_j)k\rangle
+b_i b_j\langle q,k\rangle.
\]

第一项是 Expected RoPE，第二项保留丢失的内容能量；两项平方权重和为 1。
这不是 BC 核。对固定单位，极大 covariance 会令相位部分衰减、内容部分占优；
因此此算子未必能持续区分大 support 的形状。

`isometric_rotary` helper 在固定单位下的严格零 covariance 前向值退回 RoPE，
但补空间的平方根导数在零点奇异，实测 covariance backward 出现非有限值。
严格正 SPD 的球形输入在新增测试中梯度有限。当前固定观测训练不对 covariance
反传，不能据参数训练稳定外推端到端零厚度预测稳定。

还应把 helper 与 matcher 区分：若所有 support 一起趋于零但均值固定，matcher
使用的 u 也趋于零，使中心相位无界；整套 pair-normalized pipeline 不具备固定
世界频率下的连续 point limit。共同尺度变换
`μ→aμ, Σ→a²Σ` 的不变性是另一个性质，已经被检验。

## 单位与 covariance controls

`world_unit` 从同一对视图全部有效 token 的 trace 得到尺度，没有使用 labels、
tracks 或目标统计；无效 token 不定义单位，全部无效时回退 1。
同一个有效 token 的编码仍会随该 pair 其他 support 的集合改变，因此它不是完全
独立于其他 token、全局可缓存的一次性编码。若要跨所有 view pairs 复用 K，需先
固定场景单位或另行明确坐标约定。

所有方法（包括 `world_rope`、`world_center_half`）均使用这个 covariance 导出的
共同尺度。因此这里的中心对照是“不使用逐 token 形状相位、共用观测尺度的
中心 RoPE”，不是完全不读取 Σ 的 baseline。

| Control | 保留 | 移除 | 与 full 共享的量 |
|---|---|---|---|
| `world_isometric_plane` | 中心、最小特征值及其法向、两切向特征值之积 | 切向椭圆率与切向主轴方向 | 原始共同单位、content、频率、96 维输出 |
| `world_isometric_isovolume` | 中心、determinant | 法向与各向异性 | 原始共同单位、content、频率、96 维输出 |
| `world_center_half` | 12 个中心频率、后 24 维原始内容 | 后 24 维形状相位 | moment/normal 方法的中心频率与通道数 |

三个 controls 的实际 score 已通过直接公式重建核对。`plane_isotropic` 的 normal
需要最小特征值的 eigengap；若球形或重根，normal 不唯一，不能宣称这条 eigenvector
支路端到端稳定。完整 moment 与完整 isometric 不需要求 eigenvectors。

## Soft loss 与 baseline 公平性

`soft_loss` 只对实际候选 logits 做 log-softmax，以 `label_valid` 选择 query 行，
按有效 query 行平均交叉熵。`real_dataset` 还要求有效 q geometry，且至少保留
95% 的原目标质量后才条件化到有效 key；这与 exact-BC probe 的 `retained > 0`
协议不同，不能直接横向比较数值。

全部 compact 方法共用 108→96→48 content encoder、相同温度参数、初始化种子、
训练 batches、optimizer、验证 CE checkpoint 规则。非 absolute 模型总参数 15,171，
区域任务活跃参数 15,121；其余 50 个 dustbin/head 参数无训练梯度。
`world_absolute` 参数 18,051（活跃 18,001），输入同样的中心与完整 moment 信息，
是更强容量的 same-information control。它没有 relative PE 的共同平移不变性，
因此其成败同时反映绝对坐标参数化的学习难度，不能把失败直接解释成 PE 数学更优。

`world_isometric` 与 ordinary/moment RoPE 学习参数量一致，但输出维度 96 对 48；
实际 attention 计算/缓存成本应单独报告。密度 linear 方法另外使用
positive `softplus` 内容特征和 `log(dot)+log(BC)`；这更接近正核 row normalization，
与 signed qk 乘 BC 再 softmax 的 density-softmax 方法不是只差一个实现细节。

无效 key 的数值 mask 使用 -10,000 而非负无穷。对有界旋转分数足够；对于无界
负 logBC 不能形式上保证无效候选永远零概率。开发 train/validation 检查发现：
teatime 的最小有效 logBC 是约 -7,493.5，waldo_kitchen 约 -6,404.3，没有有效 query
的全部候选都低于 -10,000；这未在当前开发集触发主导错误，但不是一般数值保证。
若扩展到极薄/极远 covariance，应采用严格 mask 并避免 `0*(-inf)` 的 loss NaN。

## 新颖性与 exact-BC 的边界

将均值与 covariance 参数做正弦/余弦编码已有直接先例。Mip-NeRF 补充材料
G 节讨论了均值 PE 与 covariance 上三角元素的有符号平方根 PE 的拼接。
当前 trace-normalized covariance + logtrace + block RoPE 的具体参数化不与该式逐项
相同，但“把高斯参数描述子变成周期编码”不能作为已有先例不存在的独立机制。
这不否定它作为简洁工程方案的价值；价值需要可靠任务实证和强对照。
[Mip-NeRF 官方补充材料](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Barron_Mip-NeRF_A_Multiscale_ICCV_2021_supplemental.pdf)

平方根概率密度的内积是经典 Bhattacharyya probability product kernel，属于已知
Hilbert 特征构造；当前 exact pairwise 实现不是新的核，也没有因此得到小维度显式映射。
有限密度向量近似若要作为最终 compact PE，需另外验证近似质量、内存/延迟、同预算
对照以及外部数据表现。[Jebara 等，Probability Product Kernels](https://www.jmlr.org/papers/volume5/jebara04a/jebara04a.pdf)

现有独立 exact-BC 报告确实显示 full 3D covariance 优于 plane-isotropic 控制的
稀疏 SfM transfer CE，但更强 projected-center RBF 的 CE 仍更低：原两场景
primary pooled 为 world BC 1.1280 vs projected RBF 1.0955；外部两场景为
1.3315 vs 1.2078。它们支持“此 proxy 上 covariance 的部分几何信息有用”，
不支持“紧凑 PE 已经整体优于强位置 baseline”。这两个报告也使用不同于主 runner
的 retained-mass protocol。

## 新增独立测试与执行结果

新增 `tests/test_world_pe_audit.py`，覆盖：

- q/k token 数量不同且各自 permutation 后的 equivariance，排除 token 序号捷径。
- 使用独立显式二维旋转公式重建 moment-RoPE 分数，以及 covariance 描述子的可逆性。
- 由两项核公式重建等距补全的 pair score。
- 球形正 SPD 的 moment/isometric 几何梯度有限。
- 区域 loss 对 dustbin/head 无梯度，并对 encoder 与温度产生梯度。
- 精确 BC 的共同仿射不变性，以及固定频率 moment-RoPE 缺乏精确共同旋转不变性。

运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest \
  tests.test_gaussian_world_pe tests.test_world_pe_audit tests.test_region_pe -v
```

结果：25 tests 全部通过。测试证明所列数学/监督 contract，不证明效果、新颖性或真实
稠密对应。审查未修改 model 或 runner；发现 runner 的 `startswith('world_')`
dispatch 会遗漏 `density_linear_plane` / `density_linear_isovolume`，已单独通知主线程修复。
