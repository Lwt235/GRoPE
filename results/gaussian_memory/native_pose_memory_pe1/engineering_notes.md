# 工程实现记录

唯一科学合同为 round_009/science_plan.md；无第二份方案。

- v1：独立新模块 native_pose_memory_*，宽192、每头48使原HRoPE八个3维块＋12对二维旋转可直接复用。29→64→192位置MLP适配当前宽度；Q=冻结末层frame分量1024＋自身预测几何20＋render1024＋有效覆盖1。所有组同容量。
- 在训练启动前修正批量地图变换的矩阵广播，显式逐样本matvec；原准备只用非batch相机，两式在该路径相同，无重跑准备。CPU独立坐标和HRoPE逆检查见coordinate_transport_checks.json。未发生GPU失败。
- query前四层camera KV保存；每次官方head复制列表，保留官方内部四次迭代，外部无新增迭代。完整aggregator prefix每场构建后释放，恢复由四个固定来源重建；其每场404,619,264 bytes及相机prefix1,048,576 bytes计入成本，不称GS替代prefix。
- 训练仅访问固定冻结缓存；48,000 reader/head样本是真实训练前向计数，未将其记为48,000 backbone/source编码。Adam/RNG/sampler每100更新永久保存，保留最终2000步和最后恢复点。
- 原GS不可变，完整Σ普通内容与B/density共享。公开learned K/V独立保存，query相关投影和HRoPE不装入永久状态。原GS、源RGB/相机/深度与官方权重是只读恢复依赖，不称单文件自足。
- 全部query目标仅在loss/评分中读取；构造/相机初值/scale/renderer/reader均不用query真值。未对开发分数选模，无科学条件更改。

- HRoPE query二维位置采用冻结预测patch点集中心的投影（同patch内depth加权代表像素，归一化减0.5），与L自身中心位置一致，所有组共同使用；没有真实相机/对应输入或位置扫描。该具体张量约定记入configuration，不把它称固定几何grid中心。

- 最终成本准入首次因原卡已有新进程而只读退出，未启动CUDA成本工作。复查八卡均有其他进程；选择低占用、47GiB余量的GPU0共享，cost单worker allocator cap=0.20设备容量（约9.5GiB），远低于实测余量，保护他人6GiB以上。正式训练源码默认cap不变。
- 评分v1.1将覆盖/初值分层的辅助均值明确为地点/种子等权，并同时导出query加权值；主终点每地点四query原已等权，数值与5000次固定bootstrap未改变。仅CPU评分与图表更新，无模型重跑。
