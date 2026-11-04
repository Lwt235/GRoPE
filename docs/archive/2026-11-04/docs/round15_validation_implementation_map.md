# Round15 旧验证：只读接口实施图

2026-10-30。本文只审阅已有源码，不读取当前 Stage B 区域结果，不授权运行。
执行设计以 [validation blueprint](round15_validation_blueprint.md) 为准；必须先有
完整 Stage B 独立审计与已绑定的最终 selection，再冻结新的验证/计时/审计实现。
所有下述既有文件保持冻结。新 isotropic 角色不加入当前 Stage B 八族校准。

## 最小新增模块

| 建议新文件 | 单一责任 |
|---|---|
| `ROUND15_VALIDATION_PROTOCOL.md` | 冻结 32 pair、18 alias→U 算子、68条件、全部 gate、新 q16 保护、数值覆盖、计时、存储和审计合同 |
| `models/round15_validation_pe.py` | 无标签的单角色 preparation→geometry→content→probability；matched isotropic 与角色所需统计；生产和计时共用此入口 |
| `experiments/round15_validation.py` | 元数据锁、forward/seal/replay、固定标签适配、metrics/gates、独立配对 CPU timing、最终完成链；不同 stage 分开 CLI 执行 |
| `experiments/round15_validation_audit.py` | 独立检查 alias/人口/来源/时序，重算保存样本、原始主候选、指标/门槛和固定数值样本；不调用生产 forward 或生产 gates |
| `tests/test_round15_validation_pe.py`、`tests/test_round15_validation.py`、`tests/test_round15_validation_audit.py` | 合成数值与角色等价、严格阶段/人口/边界、独立审计故障注入；不使用新效果调参 |

没有必要重新导出 RGB/depth 或另建中心标签 producer。runner 中 labels adapter
只消费已经审计的旧标签，并显式等待全局 forward seal；audit 保留独立投影核验。
若文件过大可拆 analysis/timing，但不能因此复制或改变科学合同。

下文简记：`A=experiments.round15_feasibility`，`R=experiments.round15_regional`，
`S=experiments.round12_stress_probe`，`M=models.round12_stress_pe`，
`D=experiments.round10_partition_data`。

## 可直接复用的低层接口，以及不能调用的整套入口

| 目的 | 可复用函数/对象 | 必须注意 |
|---|---|---|
| 哈希与运行约束 | `A.sha/read/write/array_sha`、`A.configure_runtime('cpu')`、`R.runtime()` | CPU4线程、确定性、TF32关闭；锁定版本和 torch build；新写入还要拒绝覆盖 |
| 旧数据身份 | `S.A.load_data(data)` | 校验80份 metadata/forward inputs，返回 entries；不解码 label。随后严格筛原32 val、每场景16，不照搬 Stage B 的 TRAIN_SCENES |
| 原生分区 | `M.partitions()`、`M.partition_tensors(parts,'cpu')` | g8/g16/mixed/g24 为432/108/243/48；g24 parent=-1；不可聚合到 common16 |
| 几何扰动 | `M.STATES`、`M.perturbed_observations(inputs,state)` | 见下一节；只消费合法 forward 字段 |
| float32 observations | `S.A.observation_tensors(raw,'cpu')` | 只把 xy/pixel_cov/q_features/k_features 转 float32 |
| 单高斯/point统计 | `A.prepare_query(obs,qpart,full=True/False)` | point/fixed 必须 False；它会显式放入零 C、保留同样 mean/content |
| four/quad统计 | `M.prepare_query(obs,qpart,'full4'/'quad4'/'quad16')` | four有局部scatter，quad只有局部均值与平均pixel covariance；不混成同一控制 |
| key/content | `M.prepare_key`、`models.round10_partition_pe.pool_content` | key为原生矩形；不要传旧 overlap 模型的 Gaussian key covariance |
| 八族几何 | `R.geometry(query,key,family,h)` | 只接受既有八族；新 iso 在新 adapter 中变换 C 后以 quartic_full 积分语义调用，不能改冻结 FAMILIES |
| 概率验证与 hash replay | `R.validate_probability`、`A.array_sha` | float64、非负、归一化误差≤1e−12；replay exact bytes，无近似 hash |
| 监督计数与区域指标 | `S.target_counts`、`S.common_probability`、`S.kl_js`、`R.measure` | `R.measure` 的 scenario 只接受 partition/scale；state 与 metric_kind 分开 |
| 中心指标 | `A.center_metric(prob,key_centers,point,valid)` | native argmax 首索引打破平局，squared distance≤64；返回整数 correct/valid |

不要直接运行 `S.forward/evaluate/finalize/gates`、`R.forward/evaluate/summarize_rows`
或 `A.summarize_rows`：这些函数分别写死旧8角色/旧门槛、48train全网格或训练
可行性选择。新的验证没有网格搜索，也没有 validation J winner。
`R.prepare_pair` 一次计算所有 family，不能用作单角色公平计时。

## zero 与六扰动的前向入口

新的严格 input allowlist 为十字段：

    q_to_k_xy, q_to_k_pixel_cov, q_features, k_features,
    q_points3, q_pixel_cov3, q_to_k_transform, k_K,
    q_predicted_depth, k_predicted_depth

`A.load_forward_inputs` 只读前四项，因此不能直接用于 stress loader。
不需要 q_rgb/k_rgb、sensor depth 或标签；不调用会把整个 npz 所有字段载入的
便捷加载器。每次 `M.perturbed_observations` 产出新的四字段 raw 和 diagnostics，
再进入完全相同的单角色算子。

- `zero`：重新投影原 points/cov/transform；原实现与缓存 xy/cov 做 allclose。
- `depth09/depth11`：query 3D点×.9/1.1、3D协方差×factor²；两边已clip预测depth
  同比例记录、不二次clip，key depth仍不参与当前位置算子。
- `ry_plus1/ry_minus1`：目标坐标系 y 轴旋转左乘 transform 的前三行，包含平移。
- `tx_plus01/tx_minus01`：仅目标坐标 x 平移项±.01m。

所有原投影有限化/裁剪继续 `D.project_finite_support` 的冻结规则，不能丢掉
behind-camera、tiny-FOV 或 floor-dominated rows。scale 五条件只用 zero 几何；
六非零状态仅原九条件。记录 projected xy/cov hash、transform、front count、
effective depth hash；各状态监督人口保持 zero 不变。

zero 重投影的 float64 allclose 不自动等于 Stage B 缓存路径 bitwise 相等。
冻结前应无标签核验二者进入算子时的 float32 arrays；若不同，必须明确冻结
采用哪条 zero 路径及数值一致性证据，不能在重放或复用旧概率时悄悄切换。

quartic 的 key bounds 必须是 `(native_bounds-.5)/8`，显式 GL24、query chunk16、
key chunk32，再由 `models.round15_quartic_tiled.quartic_cell_mass` 切≤1单位子格。
Gaussian 继续 `M.split_cell_bounds` / `M.tiled_cell_mass` 的24px→4个12px、GL32。
不要给 quartic 传 Gaussian child rectangles。最终统一：

    softmax(log(native_mass + 1e-30 * native_key_area/64) + T * content_dot)

## 独立中心标签与哈希链

q8、q16（及 mixed）中心来自 R10 数据 `labels.npz` 的
`{q}_point_xy/{q}_point_visible`；`D.export` 对每个 native cell 几何中心单独调用
`round6_region_data._project_gt_pixels`。它们不是区域投影均值。中心坐标为
`(lower+upper-1)/2`，原 sensor nearest-depth、半像素映射、.05+.05*z 可见性。

q24 来自 R12 `S.independent_24_center_labels(meta,parts)`；已保存位置为：

    results/round12_stress/evaluation/{scene_slug}/labels/{pair_dir}.npz
    results/round12_stress/evaluation/{scene_slug}/label_provenance.json

这个扩展 npz 已含旧 R10 所有字段加 q24 三字段，最少实现可统一读取它。只
解码 `gt_xy/gt_visible` 和四分区的 point_xy/point_visible；status仅独立审计需要。
不能借用 `results/round13_target_conflict` 的 TRAIN 标签当作 VAL 中心来源。

冻结前仅验证 metadata 和记录预期 hash；不得提前 hash/decode sensor/label 字节：

1. R10 `completion.json→manifest sha→manifest` 绑定 scene/pair/split、input hash、
   原 label hash、camera/source-index/selection 和 `sensor_sources` 的既有 hash。
2. R12 lock/partitions 与 completion 的文件表绑定每 scene evaluation completion、
   `labels/{pair_dir}.npz`、provenance；核 scene/pair 和原 label identity。
3. `results/round12_stress_independent_audit/verification.json` 必须 `passed=true`，
   并核 `stress_lock_sha256/stress_summary_sha256/stress_completion_sha256`；它的
   `scientific_gate_passed=false` 必须保留。这允许复用已核验标签，不是继承成功。
4. 本轮全部32pair×68条件×U算子前向完成/封存后，才核 label 文件真实 hash、
   解码，并按原 R10字段逐项相等、q24来源相符检查。先前 completion 的全文件
   `verify_files` 会连带 hash 标签，因此不能无差别放在 pre-forward freeze 中。

独立投影可复用 `experiments.round12_stress_result_audit.center_reference`、
`decode_sensor` 和 `native_sample`，以 native bounds 生成 q8/q16/q24 centers。
其 `check_centers` 固定只检查 q24，不能声称一次调用就覆盖三个尺度。
原始 sensor 重读仅在审计获准的 label 阶段；相机/深度不随 forward stress 改动。

## 建议记录与 gate 接线

一个唯一算子的记录 identity 为
`(scene,pair,operator_id,scenario,condition)`；scenario 取 zero/六stress/scale，
condition 为 `q_k`，metric_kind 在 scale 时为 direct，其余为 common16_to_8。
保存 family/h/T、shape/dtype、input/projected/array hash、sample path/field、
raw-FOV diagnostics。18个语义 alias 在 lock 中映射到 U个 operator_id，不重复前向。
总记录 **2176U**；全量 primary 2176数组、每额外算子固定两首pair样本136数组。

每行监督字段保持 `kl_sum,js_sum,direct_kl_sum,direct_js_sum,mass,n_query,n_key`，
另加 `pck_count,n_point,argmax_sha256` 和 label/replayed_array hash。使用
`R.measure` 时传临时 `scenario='partition'`（zero/六stress）或 `'scale'`，保存的
真实 scenario 不变。key center 使用 `parts[k]['mean']*8`，不能用 reference-unit
坐标直接与像素标签比较。所有角色、状态的对应 mass/n_point 必须精确相同。

新 summary 推荐 `metrics[operator_id][scenario]`，每项包含
`kl/by_scene/by_condition/by_scene_condition` 及
`pck_by_condition[condition]={macro,by_scene,integer_counts}`。PCK使用整数计数
经 Fraction 构造两场景等权比率；KL先 scene/condition 内质量加权，再条件/场景
等权。不要合并32pair分母后冒充等场景平均。gate 只查已冻结 alias 映射：

| gate | 输入位置 |
|---|---|
| 原九条件≥2% point收益、双场景同向 | zero `.kl/.by_scene`，分别 new strong point、legacy point |
| 原九条件准确率/成本二选一 | zero `.kl` + timing45格完整时长聚合，分别 new strong quad、legacy quad16 |
| 原九逐条件≤2% quad恶化 | zero `.by_condition`，同两quad |
| 原g8与新增g16中心保护 | zero `.pck_by_condition[g8_g8/g16_g8]`，三localizer分别精确−1/200 |
| 五尺度收益与逐条件保护 | scale `.kl/.by_scene/.by_condition`，相同 point/quad aliases |
| q24中心保护 | scale `.pck_by_condition[g24_g8]`，三localizer |
| 六stress与zero | 各state `.kl` ≤两个quad各自1.02倍；`.pck_by_condition[g8_g8]` 对三localizer；全部finite/normalized |

独立 audit 可复用 `round15_regional_audit_metrics` 的
`literal_parts/reconstruct_targets/common_prediction/divergence_sums/metric_reference`，
但不能复用其写死48train×8×12的 population/selector。`literal_parts` 有私有
lookup/area 字段，不能假设全部键均在保存 partitions.npz 中。
数值 reference 可复用 `round15_regional_audit_numeric.reference_banks`、
`gaussian_reference_mass/probabilities/probability_check` 与
`round15_train_numerical24.independent_mass`；其 `numeric_pair/planned_configurations`
固定 Stage B 网格，不能原封不动用作新 alias/stress/isotropic 覆盖。
扰动参考用 R12 audit 的 `perturb_reference` 独立重算；isotropic 在独立 NumPy
moments 上取 trace。具体首pair/全部68条件/固定query覆盖须在新协议预声明，
沿用 reference收敛≤1e−8、生产概率误差≤1e−4，不以无argmax变化豁免失败。

## 单角色公平计时的边界

生产与 benchmark 共用 `predict_role`，从同一公共 float32 projected tensors
开始；以五scene首pair×原九条件、全部U算子、CPU4线程、3warmup/15repeat计时。
每次包含实际query统计、key几何/content pooling、积分/floor/content/softmax、
common aggregation；独立进程配对循环，无并发争核。完整45格各自median再等权，
保存每次 raw time、scene/pair/condition/operator/config/source hash 和运行环境。

禁止使用 `R.prepare_pair` 的全部缓存；禁止从另一个角色借 full C、stratum bank、
几何 logits、key content；禁止把不同 h/T 或不同CPU/GPU旧timing拼接。固定原生
分区 bounds/owner 可作为公共元数据，但角色专属 native/child/tile几何构造应在
同一已声明边界内；所有roles采用一致规则，不能只预构建某一族的积分子格。

point/fixed 使用 `A.prepare_query(...False)`，不能走默认生成全 C 的
`M.prepare_query(...,'point'/'fixed')`。isotropic 的完整 C→trace 或直接累计
trace-only 二阶矩二选一应先冻结、校验，并在同轮生产/计时保持一致；不可
为使归因对照显得贵而强算未用统计，也不可免费借 primary 的 C。
T0必须测实际冻结路径，不能只在benchmark里跳过内容。Stage B 几何跨T共享
是校准加速，不是单配置部署成本。

公共RGB/depth预测、CPU dense projection、transfer另行记录，非增量门槛。
计时只应读取合法forward inputs；不能从label有效query筛快路径，不能把
common target counts放入common prediction。训练/验证/计时的scene/pair来源
分别绑定，五scene计时样本不改变32pair验证分母。

## 实施前仍需明确的阻断项

Stage B 独立审计及最终配置尚未在本文读取，因此没有实际 alias去重或验证授权。
新matched isotropic代码、角色专属公平统计、扰动数值覆盖、zero路径一致性、
标签元数据-only前置锁、完整streaming replay与独立audit必须先实现测试再冻结。
不得把 R12科学失败当作通行证，也不得把本轮region回归通过当作真实V、实际
optimized GS/NVS或未触碰场景确认通过；这些缺项仍按蓝图保留。
