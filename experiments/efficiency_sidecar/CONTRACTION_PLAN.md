# 两头位置项专用计算

假设：已有profile中较昂贵的窄矩阵乘法（head=2）可用直接归约/外积融合改善带宽利用，
从而同时改善前向与反向。竞争解释为自定义归约效率差、query梯度或其他开销抵消收益。
只替换 projected_query 与 hidden 的 contraction，公式/参数/FP32网络/FP64几何不变。
前向同时计算两个头，hidden梯度融合两头贡献；query梯度仍用原PyTorch einsum。
不使用TF32、混合精度、近似GELU或改变PE。

固定对照：geometry_factored；候选仅增加contraction。通过后再检查与已验证梯度融合的组合，
不能把组合收益全部归给新算子。relative5与Distance公平使用。
CPU FP64 gradcheck及整模型检查、GPU B4/N512全参数梯度/一步更新检查；
通过后先做B4/N512推理和完整训练步骤8块平衡A/A、B/B试验。
若有明确正向信号，再做B16/N512及组合测量；若失败/变慢则保留负结果，不盲目扩展。
固定旧seed11checkpoint、旧validation16对象池，相同样本/增强/随机数。
新GPU算子仅支持FP32/两头/一阶梯度，非CUDA保留PyTorch参考。

只与已登记项目任务共享GPU，单CPU线程、nice15、12GiB allocator上限；
初始空闲24GiB，运行保留8GiB，每阶段10分钟，出现未知进程只停自身。
不改主科学训练，不读新确认指标，不写科学checkpoint。
共享计时只作工程筛选，不能替代独占速度和完整训练终点验证。
