# 合法源值的信息接口核对

2026-10-31。**只读源码准备；未选实验、未改 head、未执行新计算。** R20 已冻结
运行，本稿没有读取其新效果、目标标签或 fresh 数据，也不修改任何冻结源。
理论边界另见 [条件值信息草稿](conditional_value_information_diagnostic_draft.md)。

结论是：**R10 的合法密集输入足够重建每个 key 像素在 query 相机中的三维点，
再构造局部联合 `(u_query,v_query,z_query)` 矩；R7/R8 的导出模型输入本身不够。**
R7 输入已经压成 token 特征、二维支撑与一个 key 深度值，未保留密集深度、
RGB 和相机。需要引用原预测 cache 或 R10 的合法字段，不能从二维 C 猜 Czr。

## 已有字段与来源

下表是源码规定的 schema，未打开任何数据数组核验。R10 每 pair 来源模板为
`results/round10_partition_data/train/<scene>/<pair>/inputs.npz`；其 manifest
绑定 input hash、原 RGB/cache sidecar、相机与 selection 来源。

| 字段 | 单 pair 形状、dtype、单位 | 来源/用途 |
|---|---|---|
| `k_predicted_depth` | `[144,192]`，原 cache float32，预测 camera-z 米 | 原 RGB 预测完整图；几何构造先 double，原规则 nan→10、+inf→20、-inf→.1，再 clip [.1,20] |
| `k_K`,`q_K` | `[3,3]` float64，192×144 零基像素相机 | 已缩放 intrinsics；不能拿 native-resolution K 配 resized depth |
| `k_c2w`,`q_c2w` | `[4,4]` float64，三维刚体相机位姿；平移米 | 已知合法元数据；源深度来自 RGB 不等于整个系统无需相机 |
| `q_to_k_transform` | `[4,4]` float64 | `inv(k_c2w) @ q_c2w`；读 key 值须用反向 `inv(q_c2w) @ k_c2w` |
| `k_rgb`,`q_rgb` | `[144,192,3]` float32，sRGB/255 | PIL RGB→BOX resize；可提供源颜色，但不是深度或可见性真值 |
| `k_features`,`q_features` | `[432,128]` float32，原 L2 特征，无量纲 | frozen ResNet layer2，插值对齐 base8 中心；不等于逐像素 RGB |
| 原 native `bounds/owner` | `[K,4]` 整数半开像素格 / `[27648]` 像素归属 | 可复用固定 key 分区；每个源像素等面积权重，不能改成 query 投影面积权重而不声明 |

依据：[`round10_partition_data.py:169`](/home/liwt/GRoPE/experiments/round10_partition_data.py:169)
的 `dense_observations` 与 export；[`round5_cache.py:85`](/home/liwt/GRoPE/experiments/round5_cache.py:85)
的 RGB-only cache producer；[`round6_region_data.py:300`](/home/liwt/GRoPE/experiments/round6_region_data.py:300)
的 cache/hash/camera 核验。R10 已存 `q_points3/q_pixel_cov3`，**未存对应 key 的
`k_points3/k_pixel_cov3`**；反向几何必须从 key 的合法密集深度重建。

R7 的 `inputs.npz` 只有 `q,k,qmean,qcov,kmean,kcov,q_to_k,q_to_kcov,
q_depth,q_value_depth,k_value_depth`，场景文件另有 pair batch 维。
其中 `q,k=[B,432,128]`、各深度 `[B,432]`，均为 float32。
R8 继续使用这些原输入，另存 sensor oracle 与诊断字段；`oracle_*`、
`k_oracle_value_depth`、`k_sensor_fraction` 均不是可部署合法值。
特别是 `direct_corresponding_pred_depth` 虽取 RGB 预测深度，**采样位置来自
GT oracle correspondence**，不能直接转为新合法 bilinear 基线。

## 能重建的内容与仍需声明的语义

令 key 原像素 `s=(u_k,v_k)`，`z_k` 为按原规则处理的密集预测深度，`T_kq`
为 key→query 刚体变换。用已有 NumPy 相机约定可重建

```text
X_k(s) = z_k(s) * inv(K_k) * [u_k,v_k,1]
X_q(s) = R_kq X_k(s) + t_kq
r(s)   = project(K_q, X_q(s))
z(s)   = X_q(s)[2]
w(s)   = [r_x(s),r_y(s),z(s)]
```

原始 `X_q` 为 `[27648,3]` 米；每 key 的 `(m_r,d)` 为 `[K,3]`，联合矩为
`[K,3,3]`，含 `Crr[2,2]`（px²）、`Czr[1,2]`（m·px）、`Czz`（m²）。
若 r 改为 reference-8，均值除 8、Crr 除 64、Czr 除 8；z 保持米。
同批源点的 r 必须真正在 query 图像中投影，不能把原 key 像素坐标与
query-camera z 拼接后称为同一 query-ray 的联合统计。条件坐标是原 query
像素中心，不需要借 R19/R20 的 a 来替代它。

均值和经验 scatter 可以直接按原 key `owner` 等源像素面积求和。现有
`round6_region_data.fit_predicted_support` 给出三维 scatter+within-pixel
footprint；若要把 footprint 也纳入联合矩，可从
`z_k² inv(K_k)[:,:2] inv(K_k)[:,:2]^T/12` 旋转后，使用
`J_joint=[J_projection; (0,0,1)]` 做局部传播。这是另一项明确的一阶近似，
不是“投影三维均值/协方差”和“逐点透视后求矩”的精确互换。

`project_points_and_jacobian` 对 behind/invalid 行输出 NaN 与零 J；旧
`project_finite_support` 则有 safe-z、xy 和二维 covariance 截断。新联合矩
对原始 z、safe-z、范围、退化、是否包含 within-pixel 项的选择都尚未冻结；
不能把裁过的二维缓存与未裁的 z 硬拼，也不能删掉难点再声称全人口结果。
这些是接口未决项，不是本稿选择的实验参数。

## 原 V 的限制与最小可复用接口

[`round7_depth_data.py:21`](/home/liwt/GRoPE/experiments/round7_depth_data.py:21)
先将 key 每个 8×8 patch 的 **64 个预测三维点求均值**，再变换到 query
相机，取 z、clip [.1,20]、转 float32 作为 `k_value_depth`。因此这是每个
key 一个深度标量，不是每个 query 在该 patch 中命中位置的深度。clip 在
patch mean 之后；把每点 query-z 先 clip 再求均值会改变原值。

[`round7_depth_attention.py:43`](/home/liwt/GRoPE/models/round7_depth_attention.py:43)
实际值为 `V_k=GELU(Linear([feature_k,log(clamp(depth_k))]))`，形状
`[B,K,64]`。它对同一 key 的全部 query 相同，但含 **128 维 RGB 特征与
深度的 learned 表示**，不能说整个 V 只有一个深度常数。query 的路由权重
和最终 head 仍依 query 变化。R12 与 R17 actual-V 接口保留该值语义。

若以后另选信息诊断，最少可复用的是四种纯接口，而非立即改网络：

| 接口 | 可复用内容 | 不可视为等价替换的地方 |
|---|---|---|
| `key_pixels_to_query_points(depth,Kk,Kq,Tkq)` | 原 sanitizer、pixel grid、相机校验/刚体变换、投影 Jacobian | 原 `q_to_k_xy/C` 是反方向且丢失深度交叉矩；R7 导出输入不能唯一恢复密集点 |
| `joint_moments(points,projection,owner)` | 原 key bounds/owner、全部源像素与等面积权重 | 投影后求矩与投影矩不同；裁剪/footprint/无效行语义必须另冻 |
| `linear_value(P,query_xy,m_r,d,Crr,Czr)` | 固定已承诺 P 可用于标量仿射值的独立读取检查 | `d+Czr(Crr+h²I)^−1(r_i-m_r)` 本身是岭深度平面；它不是独立于 plane 的另一种信息 |
| `dense_bilinear(legal_key_xy,k_depth)` | R8 `bilinear_depth` 的纯四邻插值与边界语义可复用 | 不复用 oracle 位置/valid/fallback；插值 key-z 再沿连续 ray 变换，与先变换离散 z 再插值一般不同 |

同信息 3D plane/ray 相交与 inverse-depth plane 可以使用相同原始预测三维点，
但退化/平行/外推规则尚未选择。dense bilinear 还需要预先定义合法采样位置，
其路由通常不同，不能称为同 P 下仅换 V 的消融。

仿射标量值可用每 key 的 `(d−beta·m_r,beta_x,beta_y)` 三个系数先 `P@V`
再在 query 坐标求值；**log、GELU、逐 key clip/fallback 和最终 head 一般
不能与这些步骤交换**。因此三通道恒等式不是原 41,345 参数 head 的等价
加速，也不是任意 RGB/learned V 的通用表示。新值信息若值得测试，仍需另冻
其语义、same-information 对照与原深度任务人口；此处不启动该测试。
