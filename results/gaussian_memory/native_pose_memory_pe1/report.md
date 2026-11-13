# native_pose_memory_pe1：真实原生相机头6DoF开发比较

12次固定2000步拟合、4224个细化输出、352个共同Native初值、396份公共learned K/V状态已完整交付。所有地点/query和三个种子保留；独立NumPy float64从原始XYZW pose及来源变换复算相机指标，非旧区域中心RMSE。

**主终点G−P成功率：+2.778 pp，95%区间[-2.083, +8.333]，预定实用筛=False。** 筛仅用于这个已用数据固定配置，不代表通用门禁、非劣性或总体成功。最终科学取舍交回原Astra；本实现不选下一轮。

## 固定问题、历史与范围

[唯一科学合同](/home/liwt/GRoPE/results/gaussian_memory/autonomous_research_loop/round_009/science_plan.md)（SHA256 `8ad01108a9f96068b749d818772108e023cc596f35be1b595e5b68e6d4ef96ee`）固定Native/L/P/S/G；本次实质变化为真实初值下直接监督与评价官方camera head的6DoF。历史[原生PE](../native_gaussian_pe1/report.md)深度阴性、[camera-only](../camera_only_feature_pe1/report.md)路线关闭及round008未导出新合同不改；[区域对齐](../../region_pose_utility/attempt1/report.md)有限正例和[HRoPE确认](../../region_hrope_confirmation/attempt1/report.md)强对照结果并存。不是首次预测相机或首次streaming。

来源[0,3,6,9]单独确定首相机旋转/原点与六条有效来源基线比中位尺度；44场对齐均有效、0深度回退。query真值仅loss/评分。完整query RGB；预测内参固定，外参经冻结官方相机头细化。Native诚实为同来源dense-prefix初始化；静态四源可含早query之后的来源，不称因果SLAM、纯RGB建图、GS替代dense-prefix或单文件自足。

O2/O3公共GS、O4读取、O5真实render与O6服务原生camera head的reader位置组织分列。C1来源固定；C2–C6共享KV/相机残差/强基底PE/实际初值/相机监督按合同改变。sensor及给定来源相机、静态地图、冻结backbone、旧数据是临时范围；无verified fresh reserve。

## 完整主表

|组|成功率 %|旋转中位 / p90 °|中心中位 / p90 m|
|---|---:|---:|---:|
|Native|29.167|5.087 / 78.798|0.3293 / 3.6918|
|L|27.083|8.519 / 78.634|0.3921 / 3.2268|
|P|24.306|6.803 / 96.346|0.3799 / 2.6463|
|S|25.694|6.259 / 86.990|0.3704 / 2.8724|
|G|27.083|7.124 / 89.755|0.3703 / 2.7851|

成功须旋转严格<5°且地图相机中心严格<0.2m。地点/种子等权，5000次配对地点×种子bootstrap，seed20261113。Native重复三次仅供配对，并非三个独立模型。

|比较|均值 pp|95%区间 pp|三种子 pp|实用筛|
|---|---:|---|---|---|
|G−P|+2.778|[-2.083, +8.333]|+2.083 / +6.250 / +0.000|False|
|S−P|+1.389|[-2.778, +6.250]|+2.083 / +4.167 / -2.083|False|
|G−S|+1.389|[-2.083, +4.861]|+0.000 / +2.083 / +2.083|False|
|L−Native|-2.083|[-9.028, +5.556]|-4.167 / -4.167 / +2.083|False|
|P−L|-2.778|[-10.417, +4.167]|+0.000 / -4.167 / -4.167|False|
|P−Native|-4.861|[-11.806, +1.389]|-4.167 / -8.333 / -2.083|False|
|S−L|-1.389|[-10.417, +7.639]|+2.083 / +0.000 / -6.250|False|
|S−Native|-3.472|[-9.722, +0.694]|-2.083 / -4.167 / -4.167|False|
|G−L|+0.000|[-8.333, +10.417]|+2.083 / +2.083 / -4.167|False|
|G−Native|-2.083|[-8.333, +2.778]|-2.083 / -2.083 / -2.083|False|

只按成功率规则的待评价候选资格列表：[]；各向异性G/S筛=False。这里只记录合同flags，尾部、成本与最终取舍仍由Astra判断。

训练拟合query上G/Native成功率为93.490%/58.594%，训练其他query为79.948%/70.312%；这是明确的训练范围适配线索。开发其他query则为28.472%/31.250%，未转为开发净成功率收益。开发主query的G旋转均值25.014°、中位7.124°，Native为20.639°、5.087°；G中心均值1.077m低于Native1.231m，但中位0.370m高于0.329m。全部尾部和不确定性保留，不用某个有利连续均值替换成功率主终点。

逐预测完整误差见[per_prediction_metrics.csv](per_prediction_metrics.csv)，[summary.json](summary.json)保留训练原12/新增20、拟合/其他query、开发其他query、覆盖/初值/零支持分层和连续误差配对区间；[paired_comparisons.csv](paired_comparisons.csv)和[aggregate_metrics.csv](aggregate_metrics.csv)提供全表。跨零不称等效；共同reader收益不冒充PE。

352个query中全零内容覆盖8个，全部保留。最大G/P成功损失反例为 `fidelity_09` query11，其完整排名/旋转与中心伤害见[counterexamples.json](counterexamples.json)。无按初值或覆盖过滤，全部无效值计失败。

## 实现与必要验证

单个192宽4头reader，Q为冻结末层frame-attention patch分量、自身预测点集矩、真实GS render及有效内容覆盖；原始GS的1024D绑定内容和完整Σ普通字段、B与density共享。L从自身预测patch点集矩构造同容量内容/强P字段，GS render置零。P/S/G在共同25位置维后分别放density/trace球/fullΣ的4维，29→64→192位置MLP；不是nested加信息。共同HRoPE仍为八个3×3块及x/y旋转，Q转置、K/V逆、输出正变换，频率/矩阵边界沿用原控制。

输出camera为地图c2w外参，内参始终引用初始预测K；原始pose字段保留官方head全部9维，其候选FoV不作为发布内参，详见[output_camera_contract.json](output_camera_contract.json)。位置末层及camera-token残差末层零初始化；每次把残差加到原末层2048D camera token，然后用query之前的同一camera prefix运行官方冻结head。保留官方内部四次细化，无额外外层迭代或独立pose头。训练损失仅来源unit归一的相机中心平方＋旋转弧度geodesic平方；AdamW3e-4/wd.01/clip5，batch2，每100步完整恢复，最终2000步，不选模。

坐标/XYZW与WXYZ约定、闭式H逆与结构化transport检查见[coordinate_transport_checks.json](coordinate_transport_checks.json)；四组零残差及有限非零梯度见[preflight.json](preflight.json)。批量head与单例初值约1.7e−8差不构成科学收益或新增门禁。[independent_recomputation.json](independent_recomputation.json)为全部新相机输出独立复算；[budget.json](budget.json)核对最终model/Adam/RNG/sampler与resume相等及配对样本流。普通预训练前批量matvec修正记录在[engineering_notes.md](engineering_notes.md)，无范围/监督/端点变更、无GPU失败。

## 完整成本与存储

实际GPU累计 **0.436499 worker小时 / 8**；12×2000正式更新、48,000训练reader及48,000冻结head样本。全部失败/预检/加载/准备/评价/成本均由实际worker账计入；计数详见[budget.json](budget.json)。正式准备/训练最多两空闲卡并发；成本准入后改用单张约47GiB空余的低占用共享卡，allocator限制约9.5GiB，未操作他人作业；峰值allocated=5.839 GiB。新增DATA最终6.377 GiB，含两worker原子暂存的保守峰值上界6.418 GiB /24；项目精确字节见预算，低于2GiB。

每场dense aggregator prefix为404,619,264 bytes、camera prefix1,048,576 bytes；前者按场构建后释放、恢复需四源重放。后者永久保存供所有query及第二次相机头使用。每场原GS约4.32MB只读引用；新增learnedKV、普通内容和恢复依赖不省略。[DATA_manifest.json](DATA_manifest.json)包含全部实际路径/bytes/SHA；大官方权重仅继承既有权威身份，不重hash或复制。

成本在共享卡实际测得，不作为独占硬件速度承诺。成本实测重新执行真实query aggregator、初始camera head和depth、预测几何、gsplat、reader及第二camera head；读取缓存不冒充encoder。两场×8query，四组reader就在这些实际完整query内分段计时，另计prefix与KV构建、冷/暖加载和N=1/4/8/32摊销。Native可只取pose-only aggregator+初始head；实验共同初值还计算depth，两个成本均提供。L无需GS render，其geometry时间保守包含共同准备而非优化部署下限。

|组|完整query中位 ms|reader中位 ms|第二camera head中位 ms|
|---|---:|---:|---:|
|L|168.26|4.54|8.84|
|P|175.71|5.07|8.82|
|S|177.41|4.73|9.15|
|G|176.05|4.74|8.82|

全部成本分段/摊销见[actual_cost.json](actual_cost.json)。历史原24场两次sensor prepared-source→GS构建为3.230/3.155s；新增20场来源预测10.554 worker秒、fit36.986秒、绑定编码与渲染cache47.976秒，来自[cost_inherited.json](cost_inherited.json)。这些历史范围不包含全部sensor采集/原图预处理和预测器冷加载，不是端到端RGB构建免费。当前未重建GS或下载/拟合来源；保留原依赖及成本未知边界。

## 图、交接和配置边界

两张最终图为[主成功率与完整误差CDF](figures/pose_summary.png)和[固定例及最大反例](figures/pose_examples.png)，实际view_image审阅记录见[figure_review.json](figure_review.json)。

实现CLI UUID `01a1d350-d1ec-70f2-9e37-8960b2dfe7a8`。派发明确请求gpt-6.1-sol/high/service_tier=fast；这是请求与主client观察，不是本角色独立后台attestation。后台有效model/effort/tier未向本角色暴露，均记unknown，未静默切换或改全局设置/认证。

完整结果交回原Astra决定；本轮不宣布总体目标达成、fresh确认或选择后继。未操作主goal/state/events，全部自有GPU进程已实际退出，详见[process_exit.json](process_exit.json)：成本进程捕获OS exit=0，其余分离worker保留正常完成标记及进程消失证据，未额外声称捕获其OS退出码。最终handoff后结束本CLI turn并释放本轮写租约。
