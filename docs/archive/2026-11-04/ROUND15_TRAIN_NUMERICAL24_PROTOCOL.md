# Round15 固定 GL24 数值修复：完整旧 TRAIN 前向预检

2026-10-30。父线程已授权固定 tiled GL24 修复。此前 GL16 真实输入预检 FAILED 永久保留于 `results/round15_train_numerical`，锁 SHA 为 `03fbb72ed8f4d5bbdc7cb71ae37bb1ac277fabd1a02ea939582381490c4093e9`。三个失败行来自同一 corner-tail：raw FOV 约3.42e-28、总面积floor4.32e-28，GL16低估raw约1.25%，使归一化概率偏差达到0.00309。argmax未变不能豁免该失败。

本轮唯一算法变化是每个既有≤1 reference-unit 数值 tile 从 Cartesian GL16 提高为固定 GL24。调用同一冻结模型源码，显式传 `quadrature_order=24`；不增加 attention token，不修改分布族、矩、floor、内容、候选shape、h/T或样本。旧 GL16/GL48 源、协议、测试和所有结果不改写。

完整继承冻结 `ROUND15_TRAIN_NUMERICAL_PROTOCOL.md` 的范围和阈值：c3/c5/c6各firstTRAIN pair[30,45]，q8/q16/q24各8个linspace indices对全部432g8keys；quartic_full/point × h=.25/.5/1/2 × T0/5/10。不是只复查3失败行：必须重算576geometry rows、1728probability rows/216矩阵及9组支持。严格四字段allowlist，不读取标签、sensor、val/test。

独立参考仍是 NumPy 字面完整矩/内容与 Cartesian GL64、GL96，收敛≤1e-8；actual与hybrid相对ref96概率≤1e-4，支持误差与内容误差阈值分别仍为1e-4（relative-to-1+reference）与1e-5。全部tiny-FOV/floor份额和argmax差异都保存；双方winner概率及reference top-gap照旧，不新增过滤。

先完成同一120矩形/240split/12FOV及4组梯度的独立synthetic GL24验证。通过报告 `results/round15_quartic_tiled_independent_order24_v1` 已由新数值锁 `results/round15_quartic_tiled24_numerical_audit_frozen` 绑定，锁SHA `60f9d8d16a8eb1530e6fa6dcd7af6cfd49851baeb083bded1c9a15487814b7f1`。本轮来源还必须绑定旧真实输入FAILED的完整lock/summary/completion链、输入身份、旧源和全部结果哈希，确保没有改历史或更换样本。

新runner `experiments/round15_train_numerical24.py`、新测试同名。为便于逐行审阅，runner显式保留上一版算法，仅更换数值阶数、锁/输出命名并增加历史失败链核验。默认输出 `results/round15_train_numerical24`，单独freeze后CPU run，不覆盖、不续跑。保留全部actual/hybrid/ref64/ref96原生概率、raw mass和indices，completion绑定所有文件。无论通过或失败，GL16失败继续保留。此预检仍只证明有限输入的数值有效性，不构成任何科学效果门槛通过。
