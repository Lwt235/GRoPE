# SUN3D 数据接口与训练场景验收

2026-10-30。主线采用官方 SUN3D 稀疏帧，实际获取由 `experiments/round5_acquire.py` 完成。本文件补充最初的有界本地盘点；不将旧 GS 渲染深度当作新的传感器深度。

## 官方格式依据

- [SUN3Dsfm README 的深度格式](https://github.com/PrincetonVision/SUN3Dsfm#our-depth-file-format)：RGB/depth 预先对齐，深度为毫米整数的 16-bit PNG，存储时做 3-bit 循环移位。[depthRead.m](https://github.com/PrincetonVision/SUN3Dsfm/blob/master/depthRead.m) 的解码等价于 uint16 右循环移三位后除 1000。不能直接对原 PNG 除 1000。
- [depth2XYZcamera.m](https://github.com/PrincetonVision/SUN3Dsfm/blob/master/depth2XYZcamera.m) 使用 `1:640, 1:480` 网格，因此原生 K 的主点转换到零基像素中心时减一。
- [loadSUN3D.m](https://github.com/PrincetonVision/SUN3Dsfm/blob/master/loadSUN3D.m) 按完整 RGB 文件列表取帧，以文件名的时间戳差最小者关联深度；不是按 RGB/depth frame ID 相等配对。
- [outputCameraExtrinsics.m](https://github.com/PrincetonVision/SUN3Dsfm/blob/master/outputCameraExtrinsics.m) 和加载函数说明：每三行四列为一帧 C2W，按完整 RGB 列表顺序排列。稀疏下载的第 n 个文件不能直接索引第 n 个 pose。
- 提供的 pose 来自 RGB-D 重建估计。使用其作为所有方法共享相机输入时，应称“RGB 估计局部几何 + 提供的重建相机”，而非整个系统严格 RGB-only。监督对应也包含传感器和 pose 误差。

## 已实现的接口

`experiments/round5_rgbd_data.py` 提供：

```python
frames, selection = load_sun3d_scene(data_root, scene)
frame = frames[original_rgb_index]       # 原始完整列表的零基序号
observation = frame.observation()         # 仅 RGB、K、C2W、完整网格
labels = pair_supervision(
    query_depth_native, key_depth_native, # 米；不 resize/interpolate 深度
    K_query_native, K_key_native,         # 原生尺寸；零基主点
    c2w_query, c2w_key,
)
```

读取 scene 时验证 K/pose 哈希、完整 RGB/depth 列表计数、原始帧序号、最近时间戳关联、pair 完整性和相机矩阵基本数值性质。九个选定 scene 的这类元数据验证均通过；这不等价于已经检查其全部 RGB/depth 测量质量。

`FrameObservation` 的字段只有 `frame_id, rgb, K, c2w, pixels`。调用它不读取 depth，即使删除或破坏 depth 文件，返回值仍完全相同。此便利接口的 RGB resize 为 PIL BILINEAR；主线冻结 backbone 预处理独立采用 BOX，按主线缓存元数据记录，两者不要混称同一预处理路径。

## 坐标和标签合同

默认 RGB 分辨率 192×144，stride 8，24×18 个 cell，中心为 `(i+.5)*8-.5`：首个中心 `(3.5,3.5)`，最后一个 `(187.5,139.5)`。resize 使用 `x'=sx*(x+.5)-.5`，K 也乘同一仿射矩阵。

查询中心映射到原生深度坐标，最近邻采样传感器深度，不跨边缘插值。最近像素为 `floor(x+.5+1e-9)`：其中 `1e-9` 原生像素只保证精确半像素 tie 在 double 相机往返后稳定，方向仍是向上取邻居。这个数值保护由同帧回归发现，不根据匹配器结果调整。

查询点经 `inv(C2W_key) @ C2W_query` 变换，投影到 key 原生坐标后采样 key 传感器深度，要求：

```text
abs(projected_z - sensor_z) <= .05 m + .05 * sensor_z
```

有效图像支撑为 `[-.5,W-.5) × [-.5,H-.5)`。投影坐标转回 192×144，再产生最多四个 key token 的双线性权重；图像边缘半个 cell 的连续坐标 clamp 到最边缘 token，不因缺少四个邻居剔除可见对应。

返回 `gt_valid, pixcontinuous, key_indices, key_weights, status, metadata`。每个 query 始终有一行；invalid 行的 sparse indices 为 -1、weights 为零。status 区分 query depth missing、behind camera、outside image、key depth missing、depth inconsistent。深度缺测为 unknown，深度不一致也不能一律解释为遮挡。损失/评价只条件于 GT-visible 对应，没有拒绝类别；任何 GT mask 都不得进入 forward 候选筛选。未使用预测深度质量筛行。

`project_points_and_jacobian` 和 `project_covariance` 返回像素单位的 Jacobian 与 `J C J^T`，不添加 floor，不除 stride；调用者如转为 token 单位，应对均值除 stride、covariance 除 stride²，不能重复添加模型自身 resolution floor。

## 验收与复现

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_round5_rgbd_data.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.round5_rgbd_audit
```

16 项测试通过，覆盖深度编码/单位、K 一基转换和 resize、原始 pose 顺序、nearest timestamp、孔洞不插值、解析 Jacobian/投影 covariance 单位、平面/非平面同帧闭环、已知位移、双线性标签、边缘、越界/背后/缺测/不一致、元数据哈希与 GT 输入隔离。

真实数据检查只读取训练场景 `harvard_c3/hv_c3_1`，固定 selection.json 的前 10 对（5 个 query）。结果保存在 `results/round5_inputs/training_geometry_audit.json`：

- 1928 个 pose 均非 identity，rotation determinant 约 0.999999968；最大正交 Frobenius 误差 6.38×10⁻⁸。
- 10 对的位移为 0.033–0.612m，旋转为 1.46–12.92°；15 帧有效深度范围约 0.981–8.626m。
- GT-visible 比例 0.412–0.718；5 个同帧闭环与 query 有效传感器样本完全一致，双相机原生像素往返最大误差 7.96×10⁻¹³px。
- 在同一 GT-visible 群体上，RGB L1 诊断为重投影 0.05177、未 warp 0.13406，10/10 对改善。只作注册 sanity，不用于选择/排除帧或模型。没有亮度补偿，残差仍包含异步、边缘、传感器及 pose 误差。

数值保护前，第二个 query 的同帧测试有 2/325 个点误选边界另一侧的深度。原始审计保存在 `training_geometry_audit_before_tie_guard.json`，修复后全部通过；这避免只报告成功输出。

本检查没有读取 heldout test RGB 或 sensor depth，也没有手工检查 contact sheet。单个训练房间的成功不能证明全部 heldout 场景的相机/深度质量；后续应按预登记规则统计所有 pair 的缺测/不一致原因，保留难例，不按模型表现换场景。
