继续复用已授权实现任务 UUID 01a1d350-d1ec-70f2-9e37-8960b2dfe7a8。主实际以 gpt-6.1-sol/high/service_tier=fast 启动本 turn；Astra 原生子任务 /root/research_analysis_astra 创建请求 gpt-6-astra/max 已完成 round009。后台 Fast 未独立暴露时保持 unknown，不静默换模型或改全局设置。持续研究及两个任务双向交接/读取结果/自动接续授权继续有效；科学写租约现归你，主独占 goal、state.json、events.jsonl、阶段派发。旧 turn003 与所有旧GPU worker均已实际结束，不重启或重复。

本轮唯一权威科学合同：/home/liwt/GRoPE/results/gaussian_memory/autonomous_research_loop/round_009/science_plan.md
SHA256 8ad01108a9f96068b749d818772108e023cc596f35be1b595e5b68e6d4ef96ee。
已接受短交接：/home/liwt/GRoPE/results/gaussian_memory/autonomous_research_loop/round_009/analysis_handoff.json。
现行精简协议：/home/liwt/GRoPE/results/gaussian_memory/autonomous_research_loop/protocol.md。

先读 AGENTS.md、docs/research_current.md、docs/research_outline_alignment.md 的当前状态/O-C及直接相关历史；引用原报告，不全文重复封存历史或生成第二份科学计划。round007关闭camera-only PE、round008语义问题无设计结论均保留。本轮 native_pose_memory_pe1 的实质区别是直接监督/评价真实原生 camera head 6DoF，而不是预测相机首次出现或重开深度填补。

按该科学合同完整实现并执行。你负责工程具体化：复用现有官方 StreamVGGT/相机头/GS源场与绑定内容；补脚本、张量/缓存接口、精确训练与前向账、机器配置、存储峰值和断点恢复、完整产物。常规工程选择自主做并记必要版本；不要为格式、精确计数和普通接口问题新增科学分析轮。若假设、输入、监督/主终点/对照/训练范围或预算需实质改变，则完整保存、给主一个合并问题清单，停止相关计算交 Astra；不得静默改科学条件。

重点保留：同来源prefix原生初值，来源单独确定gauge/scale，query真值只入loss/评分；Native诚实为同source-prefix基线，所有dense-prefix依赖/成本保留。完整queryRGB、预测内参不变、只细化外参；单个192宽4头reader＋camera-token零初始化残差，官方VGGT/相机头冻结，只替换最后camera token再用query之前同一camera prefix运行原相机头。无独立pose头、writer、外层迭代、深度修补或人工扰动。L/P/S/G匹配容量/样本流；P/S/G共享完整普通形状、强中心结构和共同HRoPE，最后四位置字段按合同区分。同一持久公共GS/learnedKV供多query，query相关投影在读取时算。你可自行选择缓存/分批调度以满足存储预算，不能把重算或prefix初始化成本省略。

固定12 fits×2000、batch2，种子20261110/11/12，旧32训练/12开发场景和合同query列表；AdamW3e-4/wd.01/clip5，最终权重，不选模。硬上限8 GPU worker小时（含失败/必要预检/评价/成本）、新增DATA24GiB、项目2GiB；最多2并行worker，开跑前核对实际GPU占用/CUDA/峰值空间，保护他人任务，不操作其他用户作业。常规工程预检计入同一预算，无需重审版式。若无法在预算内完整跑完，保存并报告，不擅自增加预算或减少科学范围。保持旧GLIBC与现有兼容环境，不安装/改系统libc，不暴露凭据。

主终点是开发原四query真相机旋转<5度且中心误差<0.2m的成功率G-P(pp)，按合同预定实用筛和所有分支；完整L/Native、球形控制、直接旋转/平移分布和尾部、训练及其他query、覆盖/初值分层、无效/零覆盖、最大反例及完整source/GS/prefix/query/reader/camerahead成本和摊销均保留。独立评价针对实际相机指标与坐标变换；常规必要验证与主张匹配，不重复旧大权重/数值门禁。按已复用报告规程交完整结果、精简结论和必要图的实际审阅，更新证据链/范围及入口O-C状态/决定/链接，不多处复制分析/预算/进度。

产物写 /home/liwt/GRoPE/results/gaussian_memory/native_pose_memory_pe1/，机器配置与运行账引用上述唯一科学合同，最后给 report.md 与简短 implementation_handoff.json（文件身份、实际预算/完成状态、待Astra科学问题、配置可见边界）。若 DATA 产物置既有共享数据盘，保留路径/bytes/身份/恢复依赖。主无须逐轮批准常规推进。通过公开progress/最后文本说明进度与真实阻塞，不调用主 goal 或改主 state/events。全部交付后结束实际CLI turn，释放科学写租约，不自动开启下一科学轮。
