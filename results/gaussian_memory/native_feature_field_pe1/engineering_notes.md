# 实施具体化

唯一科学合同为round011/science_plan.md，未生成第二份方案。独立新模块native_feature_field_*通过内存中的数据适配复用原camera-only reader/训练/真实原生与cost路径，不修改历史源码。

- 24份完整来源目标已有作者模型实际输出缓存，20份无保存目标的新补80次来源编码；不从query补写。
- C/J仅更新has_content槽f；J额外更新这些槽的rotation/log-scale。无内容槽的固定opacity仍遮挡。所有固定字段逐tensor核对；C协方差原样保存。来源最终场FP16 f / FP32几何，最终来源损失在实际保存精度下复算。
- 每行六个discarded reader pilot后正式重置；18个正式拟合，三个seed样本排列及初始化跨行/PE一致。原§4的唯一G绑定诊断两行均执行，不追加扫描。
- 原Native-missing/full基线只读继承；每行352个RasterFill重新走相同冻结native。全图full仍共同Native-full，不称学习保真。
- 原§6旧资源/调用上限不套用于新88场拟合与18reader：本轮仅按唯一合同8worker小时/36GiB/1GiB/最多2worker；逐样本真实账完整保留。GPU监控优先空卡，最多1外部共享卡，10GiB进程cap/16GiB准入/6GiB余量作为工程保护。
- 形成后冻结，查询输入只用于相机条件读取、训练监督或评价；任何query梯度均不回传场。源拟合与query缓存准备函数分离。

- 动态资源事件：GPU7准入时空闲，后有他人新进程进入。仅让自有C worker在精确当前更新持久化退出，J继续；随后仅恢复C未完成来源更新/场及reader，不重跑已完成拟合。此次完整运行时间和退出75均计入8小时。source恢复同时显式恢复torch/CUDA RNG，循环仍固定step%4，无新增随机采样。

- 必要源评价计数：拟合预记录初始raw/覆盖，最终完整raw/spatial/bias；最终成本阶段对两行共同初始场一次176个source-only渲染补齐初始spatial/bias的NumPy参考，不加更新/对照/query。总真实raster预计28596，全部入账。源几何/ordinary/KV成本各行6次独立完整构建（12次source编码），无额外query-reader成本扫描。

- CPU交付脚本一次字符串补注产生多余右括号，py_compile发现后即修复；未启动该脚本或GPU/model工作，科学结果不受影响。
