# 当前保存内容的数据流事实（主任务只读追踪）

这是源码事实交接，不是新科学合同，不包含实测内容性质或 PE 效益结论。没有加载模型／场张量、训练、推理或渲染。

- `models/gaussian_memory_coarse_pe.py:130` 的 `patch_tokens` 对单幅图归一化后直接调用 aggregator.patch_embed，取 x_norm_patchtokens。没有经过 VGGT 的24对 frame/global 块，且四源是逐幅调用。因此不能将所存特征描述为已经多视图混合的原生24层输出。
- `experiments/gaussian_memory_native_run.py:19` 创建默认 StreamVGGT；`third_party/streamvggt/src/streamvggt/models/streamvggt.py:20` 构造默认 Aggregator；`third_party/streamvggt/src/streamvggt/models/aggregator.py:63,147` 默认 patch_embed 是 dinov2_vitl14_reg，映射 vit_large。故按本地源码构造链它本身是单图 Transformer 的上下文特征，不是原始局部像素平均。此为源码构造证据，不是新的运行时模型配置证明。
- `experiments/native_feature_field_source.py` 的 prepare_targets 用四幅独立 sourceRGB 编码得到4×32×32×1024目标。已有目标继承 source_geometry_prepare.py 的同一 patch_tokens 路径；新源另保存。fit_one 在 J 行优化每槽常量1024维 f、rotation/logscale；中心与opacity固定。300步轮流拟合四幅实际 gsplat 归一化 rendered feature 的全图MSE。此处不是24层Q/K，也没有query监督参与形成。
- `models/gaussian_scene_memory.py:23,53` 实际 gsplat 可生成贡献 W，初始内容 W^T F/mass；J 行之后为上述重构共同形成。`native_feature_field_data.py:34` 定义源目标缓存，fields/geometry/views/anchors 均已有路径接口，原 query full/missing patches 由 native_coverage_data.py:34 分派到原数据缓存。
- `experiments/native_feature_field_run.py:14` 查询准备读取冻结形成场，按query相机生成view metadata与真实归一化render anchor；camera_only_feature_run.py 的 stage_training 使用原 full patch_tokens 作为完整特征目标。reader 不接当前RGB；native() 仅替换 missing patch 中央16×16，再执行原生接收器和深度输出。
- `models/camera_only_feature_reader.py` source 普通输入为 layernorm(f1024)+共享source20（含完整logΣ六维、中心邻域等），source→K/V各128。query 为 normalized render anchor+camera20+logalpha；K 另加29维位置MLP，同common25＋P/S/G各4维。P是中心density4，S/G是球形／完整Σ投影footprint。P同样共享真实GS内容、普通完整Σ、render anchor和中心邻域，故PE比较不是有无形状信息比较。

尚未测量：相同场景点/相同局部支持处的这些单图DINO内容，随视图／采样变化是否稳定，还是主要受图像上下文或遮挡影响；物理Σ所描述支持与实际内容所需支持的联系。不能从上述源码直接下结论。

可复用缓存接口是事实；文件可用性／张量细节和小预算工程安排交Sol核查，不在此加载大型权重或复算旧评价。任何测量仍仅为development机制诊断，不能替代强中心PE及原生端点、独立确认。
