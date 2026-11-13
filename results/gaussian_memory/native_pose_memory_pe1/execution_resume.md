# 本轮唯一恢复位置

科学合同：`../autonomous_research_loop/round_009/science_plan.md`，身份见configuration.json；不得另建科学计划或角色。本CLI UUID固定为`01a1d350-d1ec-70f2-9e37-8960b2dfe7a8`。

- 永久DATA：`/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/gaussian_memory/native_pose_memory_pe_v1`。
- 准备按preparation/{id}.json完成标记及cases/{id}/{query}.pt原子产物去重；source prefix可由原四张RGB重新构建。输入/官方大权重/GS只读引用见input_manifest.json。
- 拟合唯一恢复点：DATA/fits/{L,P,S,G}_{20261110,20261111,20261112}/resume.pt。每100步保存model/Adam/torch+CUDA RNG/sampler/预算。final.pt只在固定2000步完成后形成；既有final不重训。
- slot0顺序L/S，slot1顺序P/G，三个seed依次；新版本运行命令为现有兼容Python `-m experiments.native_pose_memory_run run --slot {0,1}`，环境及实际GPU UUID见admission.json、native_pose_memory_launch.py。任何恢复前必须确认旧pid已退出并重新核对实时GPU/磁盘/剩余worker预算，不能重复队列。
- 每个预测原子保存，evaluation标记形成前仅跳过已完成预测；不得删旧/失败产物或为挽救结论加训练。
- 当前round009只有本轮固定范围，完成后实际结束CLI turn并释放写租约；后继科学判断仅交原Astra，经主派发。
