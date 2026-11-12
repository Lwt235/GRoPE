# source_geometry_control1续接

用户已明确设目标并执行；目标active，尚未完成。固定方案docs/source_geometry_control_plan_20261112.md。

48份场景/绑定/查询缓存、8条件预检、配置锁及输入/原权重身份检查全部完成。12拟合×2000步，4组×3种子，各96评价；6个G模型各288干预，共2880份新预测。所有旧数据，来源传感器/共同有效性为诊断特权，不称RGB方法成绩。

后台：source_geometry_orchestrate队列包含12拟合+diagnostics+cost；postprocess等待固定队列后依次score、training_summary、final_checks、figures、report_tables。检查queue_status.json及logs、postprocess_state.json。父轮gaussian_scene_memory1文件不要修改。新训练复用原trainer，进程内只重定向OUT/CACHE/geometry_path，场景按mode隔离，原query/patch不变。

当前4个首种子在GPU0/1共享及GPU4/5空闲卡，cap12GiB、最多4worker、最多2外国任务共享卡，压力退让后从100步检查点恢复；保护其他用户任务。控制器PID见orchestrator.pid，后处理PID见postprocess_state.json。

常规修复：parent_source_paths_v1只修父轮源码哈希键相对固定仓库的路径，保存错误源与说明，不改任何科学设置。

剩余：监控完成全12fit/2880输出及预定质量/成本；处理实际运行错误并留版本记录；确认完整独立复算和身份；实际view_image逐一查看figures的3图后写review记录，不能只看manifest。阅读report_tables、quality_summary与scoring交互，判断来源质量操纵是否生效及PE增量；写report.md与method_decision.json，更新研究current/outline/core/evidence/continuation/plan文档，保留旧负结果。有限完成审计后update_goal complete，无自动后继。不要因为只完成预检/首种子/训练队列而标记目标完成。


运行更新：首个sensor/G_20261110已完成2000步、96评价及288干预；384新预测独立复算最大差2.50e−15。empty_source_diagnostic_v1已修复来源view6共同有效像素为0的undefined coverage，用null保留分母，不改科学设置。48诊断已产出。CPU控制器换为27200并接管三个原GPUwrapper（128024/128025/129217），无训练中断。后处理先因旧失败快照竞争退出，已在新队列clear且原PID退出后再次启动；新PID见execution_status.json及失败目录postprocess_resume.json。不要依据旧PID或旧失败文件重复启动。


首个完整配对种子已齐：4次训练、384适配、576干预、960新预测独立复算，最大差2.50e−15。first_seed_four_arm_description.json显示G对P在pred条件−6.25%、sensor条件+1.17%，但sensorG对L仅+0.11%，单种子不晋升方法。第二种子继续。成本任务是否完成以cost/complete.json及实际PID为准，不看过期queue行。


成本与实际预览已完成：cost_description.json汇总预处理RGB/深度→场景3–5秒、完整暖查询约198–218ms；不含AnySplat生成、解码及I/O，非端到端加速。preview/fixed_scene.png已实际view_image并写actual_review.json：sourceRGB相近却预测深度近乎平坦，sensor恢复部分层次，但query2大片无覆盖、query11仍模糊。最终三张图仍待全12fit后实际审阅，不能用preview替代。


两完整配对种子已完成：8fit/2000步，768适配+1152干预=1920新预测独立复算。two_seed_four_arm_description.json显示G-sensor对P的逐种子降误差+1.17%/−5.22%，合并−2.03%；主交互逐种子符号翻转，未确立可靠PE收益。G-sensor对L合并仅+0.28%。第三种子4个worker已经在运行，PID见queue_status，切勿补种子、改公式或把目前当最终。


## 2026-11-12有限目标完成

全部12次2000步、1152适配+1728干预=2880新预测，独立复算、48场景质量、成本、配对身份、三张最终图实际审阅、报告/方法取舍与七份研究文档同步完成。completion_audit.json逐项核对全部有限范围。P-sensor对P-pred主误差低4.27%且配对区间支持改善，G对P在pred/sensor条件主误差均值高6.62%/5.02%，交互不确定，完整ROI空间代价有支持。结束本固定诊断，不晋升G，也不把额外传感器成绩称RGB方法；没有自动后继或本任务后台进程。旧进度与失败记录均保留。
