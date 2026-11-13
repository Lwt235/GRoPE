你是 GRoPE 双模型自主研究循环的实现角色。复用本 CLI 会话 UUID 01a1d350-d1ec-70f2-9e37-8960b2dfe7a8。本次实际 CLI 参数仍明确请求 gpt-6.1-sol / reasoning high / service_tier fast。主已从本会话 rollout turn_context 看到 client-selected model=gpt-6.1-sol、effort=high、approvals_reviewer=auto_review；后台实际服务档位未暴露，不能声称已独立确认 Fast。原 Astra 分析角色保持 /root/research_analysis_astra，创建请求 gpt-6-astra/max，本次科学计划已完成且其实际 turn 已结束。旧原生 Sol 完整完成 round003 并结束，不再派它执行本轮。

人类持续授权：为本研究复用分析和实现任务，允许主与两任务之间双向交接/读取结果/自动下一阶段，顺序交接避免同时修改同文件。现在仅你获得 round004 科学代码/研究文档写租约。主独占 goal 和 autonomous_research_loop/state.json、events.jsonl 及自身协调记录；你不调用 goal，不改主 state/events，不另建 agent/task，不自行选择科学后继或向其他线程发送消息。通过本 CLI 公开进度和最终交接向主报告。主会把关键公开原文转回原会话；勿把请求配置当后台 attestation。常规工程修复自主继续、简洁版本失败记录即可；涉及科学方案实质变化时安全保存并在公开交接说明，交主复用原 Astra 判断，不自改公式、预算、样本或实验问题。

完整目标保持：为 VGGT 建立精炼的公共粗 3DGS 场景记忆，查清高斯参数作为 Transformer PE 相对强中心几何的可靠收益/适用范围。内容载体、PE增量、原生用途和fresh确认分开。本轮不是新PE候选，是决定下一设计/范围的固定权重条件诊断；不得把渲染形状用途称 O6 成功，不得将固定权重输入扰动称从头优化中心方法劣势或唯一病因。

先读 /home/liwt/GRoPE/AGENTS.md，docs/research_current.md 当前开篇，docs/research_outline_alignment.md，相关历史与授权/protocol。原 Astra 的最终机器交接：
/home/liwt/GRoPE/results/gaussian_memory/autonomous_research_loop/round_004/analysis_handoff.json
SHA256 88d3fd0ffddb58a72a49c3f2e1af46306a8842ed5beeb0709333f5f2d66c9818
固定机器计划：/home/liwt/GRoPE/results/gaussian_memory/autonomous_research_loop/round_004/plan.json
SHA256 d2391ea222d606a56a6c6490633fea4a5e4c171f754b5771e700358da21f659f
完整执行方案：/home/liwt/GRoPE/docs/native_covariance_role_execution_plan.md
SHA256 8111d5a7b6dddff8a2ee619788347edd59d808941002c270c601a43eb5afb7ca
分析：round_004/analysis.md；输入和历史身份：round_004/input_reports.json；主的一次适当计划核验：round_004/coordinator_plan_check.json。不要修改原科学计划/交接身份；运行源码/派生配置另立锁与普通版本修订。待填的实现 UUID 已由主补在本派发记录，无需改锁定科学计划中的占位标签。

实验 native_covariance_role1，只做 F/C_raw/C_match/S_match 四组。原 native_training_coverage1 三份 Render 最终权重冻结 eval，复用原44场、2048个基元/1024D绑定内容、相机/opacity/has_content/unit、来源0/3/6/9，不用round003 writer权重，不重建源/重绑定/下载/拟合/续训。训练32地点仅其他query1/4/7/10，开发12地点原query2/5/8/11，full/missing。每组1056，共4224新预测，包括同一真实渲染路径新F；2112旧Render/Local只读继承评分，不把旧cachedR当新F。旧缓存与真实renderer浮点路径差描述即可，不套1e-4 native输出门槛。

C从全部保存中心self+31稳定最近邻的FP64中心散布B构造D=B−lambda_min(B)I。非退化Sigma_C=mI+(t−3m)D/tr(D)，t=原协方差trace、m=原最薄轴方差；tr(D)=0回退同迹球形并计数。S=tI/3。C共享原每基元大小/最薄轴这一额外优待，不能叫纯中心消融；只问剩余非球形信息。合法scale与WXYZ quaternion依执行方案，不覆盖原场。F/C/S均同gsplat32×32渲染colors=[f*has_content,has_content]，得FP32 r_X和有效内容a_X而非总alpha。a_X>0时FP64除法后回FP32；a_X=0的明确中心/均值/零回退按原方案，记录比例和a_F质量、相关query。a_F=0匹配输入零。四输入 r_F、r_C、a_F*n_C、a_F*n_S。匹配输入仍含原a_F与原绑定内容，不能称删除全系统形状信息；不得筛不利query或追加范数/温度/邻域/尺度扫描。

固定预算严格执行：0训练/optimizer/pilot/新权重/下载/source重建；4224正式新+2112继承评分；native总≤4272（正式4224、预检最多16、普通失败恢复最多32），两旧训练场四组共8前向为足够的必要预检，额外额度不是任务；528准备内容/有效覆盖输入，独立gsplat调用总≤600；不另做GPU速度扫描，成本报告来自本次实际调用/准备/加载/CPU/I/O/持久字节。所有活跃GPU工作含失败/恢复≤1 GPU worker小时；最多2worker/最多1外部共享卡。初始cap8GiB/free≥12GiB/runtime边距4GiB；仅完全空闲卡可cap10/free≥14。启动前实查占用、CUDA、盘空间，保护其他用户任务，不碰旧/他人进程。新DATA硬≤4GiB、项目≤1GiB（含暂存/失败），目标cache路径见机器计划。H余量和30s/大写入前监控、按已写产物去重抵扣及安全退出规则照原计划；不删旧结果、不复制旧大权重。旧大权重按既有权威身份+路径/大小继承，不重hash全部。逐query原子保存/完成身份/调用与资源账，恢复只做未完成，避免重复队列。

必要验证仅覆盖新协方差构造、trace/最小轴、投影/有效覆盖、FP64归一化/零支持规则；全部新预测独立NumPyfloat64复算，保留原支持mask/完整raw/bias/fullAbsRel/分层/seed/place/query/反例，5000配对bootstrap seed20261113，主量与固定决策分支完全按计划。三种子、实用1%和区间仅用于此固定配置条件依赖排序，不是通用门禁/等价/非劣性/独立确认。不得靠微小数值差异增加审计或再训练。

工程环境：服务器旧GLIBC不可改，优先既有 /home/liwt/miniColmap/.venv/bin/python（NumPy先于torch导入），既有CUDA/gsplat。沿用相关可用代码，使用独立的新实验模块命名，避免覆盖历史models/gaussian_memory_context.py等已锁源码；原模块如必须例行修复留精确diff/旧快照，不改变旧科学结果。自动审批/现权限范围保持，不使用绕过审批或sandbox的dangerous选项，不改全局Codex设置/认证，不输出密钥/全环境。

完整交付：report.md、全数表/逐预测指标/summary/独立复算、干预输入与几何/支持/回退统计、实际预算资源账、method_decision.json、进程正常退出、两张实际view_image审阅的最终图（固定例/最大反例按执行方案）、manifest和implementation_handoff。更新current/outline/evidence/execution状态及完整正负证据、范围与未确认边界；原报告原科学锁保持，普通问题版本化，不另开科学轮。每个有意义里程碑/异常通过公开agent_message发进度，主会转述关键原文。全部完成后给主具体报告/交接SHA、实际配置可见边界、预算/退出/未完工作，结束本CLI turn释放科学写租约。不要在本轮末自主选下一实验或训练；主再交回同一Astra。
现在按锁定计划实际实现、必要验证并运行，不停在计划或建议。
