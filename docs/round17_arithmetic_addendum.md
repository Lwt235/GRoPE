# Round17：真实预检已保存数组的同质量算术审计补充

2026-10-30。本补充在原 R17 core/preflight 已冻结、正式 synthetic 正在运行
而真实 TRAIN numerical 尚未开始时提出。补充自身必须先冻结，且 freeze
拒绝任何已存在的原 `train/` 目录；之后才允许原数值预检读取训练输入。
不改任何原冻结文件，不修改原 1e−4 概率误差或 1e−8 GL64/96 收敛门槛。

原 synthetic 已精确核对 tau32、bias32，并对相同 production raw mass 核对
概率至1e−12。本补充把这一强算术核验延伸至预先选定的全部真实预检样本，
不是新增效果实验、修改旧结果或绕过数值失败。

明确的运算合同为：将 C/h/T 转为 float64；
`s=max(h²,C00,C11); r=h²/s; c=C00/s+C11/s; tau=T*(r/(r+c))`；
随后 tau 转为 float32，与原 content32 在 float32 相乘，再转 double
加入既有 geometry64。其数学表达式始终为 T h²/(h²+tr C)，生产和独立
源码以及两份设计文档已锁定同一顺序；此处不改变公式或算子。

输入仅为已完成的原 numerical TRAIN 工件：42份 support archives、168份
geometry/probability archives、504个概率矩阵、4,032个原定 query行。
不打开原 observation inputs、sensor、中心/区域标签、VAL 或新场景。
先验证原数值锁、源码快照、完整 train 工件哈希、合成与 train 完成状态及
时间顺序，再使用保存的 actual covariance/content/raw mass/probability。

逐矩阵要求：独立 NumPy tau32 与重新执行固定 production trace32 精确相等；
独立 NumPy bias32 与 production bias32 精确相等；仅以相同 actual raw mass
独立重算的 probability 与保存的 production probability 最大差≤1e−12。
面积 floor 仍按 native reference8 area 加一次，完整保留 tiny/zero-FOV。
所有结果均保存；任一算术条件失败则补充审计失败。这里重算的 production
tau/bias是锁定函数，不是原运行曾另存下来的中间数组，报告必须如实区分。

每个矩阵保存差异、tau/bias数值及形状/数据hash，所有原概率仍由原train
archive hash绑定；不重复保存大概率文件。新 completion 绑定全部新工件，
verification绑定原 numeric lock、train summary/completion和新审计lock。
数学审计通过不能充当 PCK、区域质量、效率或一般性验证通过。
