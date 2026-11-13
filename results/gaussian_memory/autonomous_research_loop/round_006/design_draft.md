> 已由2026-11-13最终analysis.md、plan.json及docs/camera_only_feature_pe_execution_plan.md取代。以下为未锁定形成记录；其中原拟gsplat80上限及位置字段描述不是最终执行条件。最终采用共同实际render anchor、共同中心B投影＋额外P density，gsplat≤400。

# round006 可恢复设计草稿（未锁定；最终以plan.json为准）

2026-11-13。四项事实完整评价正在形成；不继续联网/综述，不复制GRF大系统。GPSToken/SplatFormer/JSRE不是独立参数PE正证据；GRF的真实footprint key分支提供条件线索，但普通K注入变化本身不构成新实验理由。关闭四项直接移植候选与原固定场＋depth联合适配的自动PE修补，不宣告总体停止、成功或外部阻塞。

选定一个实质问题：**直接监督公共GS状态＋相机的视图特征读取，在没有当前RGB读取旁路时，能否使参数位置组织相对同信息强中心得到原生VGGT用途？** depth、sensor固定场和监督方式不是核心要求；此处主动改变学习任务/接口，当前选择仍用native depth评价实际用途。不是由“无RGB”推定形状会有收益。

拟定`camera_only_feature_pe1`：同44份source-only实际GS输入，单个128维四头camera-query cross-attention＋共享小输出MLP预测1024维DINO patch特征。训练只有既有完整查询patch缓存监督，原VGGT与源几何冻结，无深度/接收器梯度。源内容及完整Σ、中心邻域结构为P/S/G共同普通输入；中心多尺度位置共同，指定四维位置分支为中心邻域投影形状P／球形S／原Σ G，名义容量、实际活跃输入、初值与训练范围匹配。公共K/V状态一次构建、永久保存，同一状态供8query；读取函数不接RGB/教师/深度/查询编号。

原生检验：在已知missing区域，用预测特征替换原遮挡query的对应encoder patches，其余保持；VGGT官方原生层/DPT冻结，无新增融合MLP。Native不填补、RasterFill用原GS合成内容归一化直接填补作为固定用途参照；G/P为同学习预算主PE对照。full输入不替换，输出复用共同Native-full，不能称非劣性。完整特征预测误差/偏差/空间项与native指标分层；特征阳性不能替代下游或fresh。

预期固定P/S/G×3seed×2000步，batch2；主12开发原query，训练及其他query完整报告。正式9×352=3168特征与missing-native预测；Native/RasterFill/full参照1056，预定G内容绑定打乱144，正式native4368；独立source-state导出396，另36打乱。预算拟6workerh、native≤4650、非训练reader≤3500、source编码≤600、实际raster≤80、DATA≤18GiB/项目≤1GiB。只保存小reader权重、FP32预测及永久K/V，不复制原生大模型。资源实时准入。

尚待定稿的是精确位置字段与有限成本计数、缓存身份/注入点说明及结果分支。以上未启动任何GPU、训练或实现。常规张量/缓存定位在同一有限实施计划预检内处理，不单列合同核查轮次。科学解释与方案仍由原Astra负责；主独占goal/state/events，Sol待命。
