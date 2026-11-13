# 当前双模型循环恢复入口

整体目标active、未达、未暂停；以[state.json](state.json)为唯一调度状态。round015 Astra/max实际完成并锁定[唯一科学合同](round_015/science_plan.md) native_task_writer_pe1：共享source-only物理几何/内容writer接受原生VGGT任务梯度，C/J×P/S/G，18×2000，GPU≤18worker小时、新DATA峰值≤80GiB/项目≤1GiB。原44场来源初始化，学习状态部署时只由来源输入生成，多query共用；PE比较为整条训练链效应，不宣称固定同场直接因果或准确物理形状。

**现唯一活跃实现任务：`/root/research_implementation_sol_recovery`，native collaboration，实际spawn参数gpt-6.1-sol/high/fork_turns=none已接受，持有实现科学写租约。** 用collaboration.list_agents/wait_agent检查同一handle；不要轮询旧CLI句柄或重建任务。主独占goal/state/events，收到完整交付或实质科学问题后才复用原 `/root/research_analysis_astra`，原创建gpt-6-astra/max。

接口恢复已公开说明：旧CLI UUID01a1d350-d1ec-70f2-9e37-8960b2dfe7a8的round015尝试008/009在模型回合前exit1，实际写锁由同时持有主任务的共享app-server拥有，可逆归档/恢复未释放；app010消息MCP传输失败，实际最后仍旧completed回合，没有新训练。记录在round015/implementation_startup_recovery与writer_owner、各attempt/completion。**旧实现接口已停用，不杀共享服务、不删锁、不重发旧任务。** 原12轮及所有旧训练完成结果不重跑。

恢复子智能体接口无Fast参数，本轮未显式请求Fast、不能声称开启；用户偏好保留，模型和high实际请求不变，后台effective未知。全部身份/请求/故障已记录，非静默模型替换。Sol按当前资源/CUDA/CPU-RAM-存储准入，用实时安全空卡动态并行，不沿用默认两卡，保留他人任务和完整恢复/永久final。完成独立评分、全部对照/辅助/反例/成本、实际图审阅和短implementation_handoff后由主交Astra。所有旧数据非fresh。现行[协议](protocol.md)及持续授权保持。

## 当前工程恢复状态

RAM迁移完成：C/P107、C/S200、C/G153共460有效更新，完整model/两个Adam/RNG/sampler字节身份匹配，旧子进程实际exit75且无丢弃。当前唯一manager以实验queue_status.json为准，已五路独立正式作业（4卡C/P+J/P，6卡C/S+J/S，7卡C/G）；只用自有空卡，具体并发由实时资源与自然更新吞吐/worker小时预测决定。周期完整恢复250步，SIGTERM当前更新后立即保存。/dev/shm本轮恢复目录计入RAM+DATA合并80GiB峰值，不能当重启持久备份；永久final仍DATA。不要把旧manager失败列表中的迁移exit75当科学失败或从头重跑，勿触碰旧已退出PID。
