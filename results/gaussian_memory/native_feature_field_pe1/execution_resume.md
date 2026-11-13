# 唯一执行／恢复位置

科学合同只有 `../autonomous_research_loop/round_011/science_plan.md`。实现CLI固定 `01a1d350-d1ec-70f2-9e37-8960b2dfe7a8`，不另建角色或后继。

DATA：`/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/gaussian_memory/native_feature_field_pe_v1`。C/J各自：

- `source_checkpoints/{id}.pt`：精确来源step/f/rotation/log-scale/Adam/RNG、来源循环step%4和计数。`formation/{id}.json`完成即不重拟合。C资源让出保留confirm_03第293步，恢复仅跑最后7步及其余未完成场；已完成15场继承。
- `fields/{id}.pt`：最终FP16 f / FP32几何；后续查询绝不修改。`geometry/views/anchors`为可重建的冻结测量缓存。
- `resumes/{arm}_{seed}.pt`：每100 reader步完整恢复；`finals/{arm}_{seed}.pt`固定2000步model/Adam/RNG/order/budget。仅完整final验证后退休重复resume。
- `completed/*.json`：每个feature/depth原子发布身份；恢复跳过完成query，不重复推理。`states`为每场每模型一次构建并重载、供8query复用的K/V，原场保持依赖。

普通局部队列 `native_feature_field_resume.py` 等J actual exit0后仅启动C恢复；`native_feature_field_post.py`按各行完整完成标记作CPU独立评分，不派发模型角色或碰主协调状态。实际PID/exit/累计wall见本结果根runtime。任何异常恢复先检查旧PID实际结束、重新核验GPU/空间和剩余8worker小时，不重复队列或扩预算。

最终实际结束此CLI turn释放本轮科学写租约，科学取舍交原Astra，经主派发。
