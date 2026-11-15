# stac_route_review1：固定作者入口的读取、融合与信息边界

2026-11-15，round047实施事实交付。[唯一合同](../autonomous_research_loop/round_046/science_plan.md)；固定 Rainzor/STAC commit **fd7e718597cf9963de85c8fffae32a698e8619f5**。复用round031作者身份、总体机制及norm/RoPE后append事实，不重审旧实验结果。本报告不作方法采用或后继科学选择。

**已定位的入口并非仅按体素保存均值。** 它同时保留首帧／近期／历史注意力heavy hitters热缓存、按最近预测几何邻域检索的体素代表、代表计数／分数／溢出和检索副本；默认四帧chunk内共同可见，默认相机另对全部历史pose token重算。局部pivot/buffer cap和固定检索输出长度不建立完整历史总byte上界。作者CUDA merger的内部融合、精确检索排序与count-bias实现尚未取得正文验证；下文严格区分已读调用事实、Python参考实现和unknown。没有加载权重、模型执行或精度／性能结论。

## 1. 配置和证据口径

分析入口为作者公开示例 `main.py --scene_dir … --base_model streamvggt --mode stac`，不是默认`stream3r`权重，也不是eval脚本的所有可能配置。[main.parse_args/main，L27–121](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/main.py#L27)；[README示例，L271–287](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/README.md#L271)。

| 已读直接配置事实 | 此入口的激活值／范围 |
| --- | --- |
| 模型／内部模式 | `CausalVGGT(base_model='streamvggt')`；`stac`展开为`window_chunk_merge`且强制streaming |
| 缓存与输入 | recent4、chunk4、HH2、pinned `[0]`，保留热token预算7帧等价、append预留11帧等价；transfer16帧 |
| 检索 | retrieval_size2帧等价，return_buf=True；并非每个query分别选择2个代表 |
| 几何／融合 | voxel_size .05、init_voxels4096、buffer cap8／pivot cap4（每layer-head-voxel）；sim和replace阈值均.8，score阈值.2；all global layers；segment allocator、seg_size1 |
| 注意力／精度 | attn_backend=`cuda`，不可用时Python调用处转Triton；subsample_ratio1；temperature .9只用于历史分数累积。dtype auto：CUDA capability≥8时BF16，否则FP16；CPU分支FP32。此为源码分支，未探测本机CUDA |
| 相机／采样 | cam_cache_update=False；camera迭代4、trunk4；size518；**kf_every10**，主入口先整段RGB载入device再做stride10 |

`STAC_DEFAULTS`仅对等于argparse默认值的参数替换，因此显式传入同默认值也会被preset替换；例如经`--mode stac`传chunk1不能凭该值绕开chunk4。直接`window_chunk_merge`或程序接口可激活其他配置，但不在本报告混称默认STAC。[model_wrapper.STAC_DEFAULTS/run_model，L86–129](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/model_wrapper.py#L86)；[session.pipeline，L381–435](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stream_session.py#L381)；[register_kv_mgr，L567–615](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stream_session.py#L567)。dtype保留在pipeline的kwargs并传manager，不能把register函数的FP16后备默认误认成无条件FP16。

## 2. 当前query实际读取什么

**直接调用链。** `main→run_model→StreamSession.pipeline→CausalVGGT._forward_streaming→CausalAggregator.inference→SparseAttention.forward→STACVoxelKV.append_kv/decode_sparse_attn`。24个global块均登记manager；frame块仍为每帧自己的普通注意力。[aggregator.register_kv_mgr，L159–173](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/causalvggt/models/aggregator.py#L159)；[inference/_process_global_attention，L488–648](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/causalvggt/models/aggregator.py#L488)。

1. **写入后立刻读。** Q/K先各自norm，再用原来源2D patch位置RoPE；作者将RoPE结果`.type_as(v)`。当前chunk全部K/V先append热缓存，再以当前全部Q读热缓存和已有retrieval副本。manager分支不向decode传attn_mask；其他非manager分支的mask不是本路径的保护。[attention.SparseAttention.forward，L113–160](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/causalvggt/layers/attention.py#L113)。
2. **完整分母的集合是hot∪retrieved，而非全部原始旧token。** decode按head拼接hot和retrieval K/V；hot列bias0，retrieval列用返回的bias，包括无效列的−inf；query不另做逐行top-k。CUDA wrapper默认scale `D^−.5`；已读fwd kernel遍历整个`seqlen_k`块，施加列bias与越界mask并做online softmax。由此静态推断每行是对这个拼接集合做共同softmax，不能把各集合分别归一化后相加。[stac_voxel.decode/_build_bias，L209–282](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/stac_voxel.py#L209)；[CUDA wrapper，L6–45](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/attn-cuda/attn_cuda/__init__.py#L6)；[fwd kernel，L103–109、L291–422](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/attn-cuda/csrc/kernels/fwd_kernel.h#L291)。被删原token不会逐个出现在分母；即使代表bias按log(count)实现，也不是恢复所有不同原K的精确softmax。
3. **热选择是跨query历史内容分数，不是当前每个query的独立地址。** 注意力列和汇总到每layer/head的hot scores；新列赋本次分数，旧列`.9*old+new`，分数缩放`T_live/Tq`。prune保留首帧整组token、最近4帧整组，再从其余前缀按每layer/head分数top-k保留2帧等价token。HH可包含patch和特殊token，并非额外完整2帧。[h2o._update_scores，L177–217](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/h2o.py#L177)；[_parallel_prune_kv，L245–355](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/h2o.py#L245)。CUDA列和具体kernel正文未取，本轮不声称逐位验证其计算。
4. **几何检索共享选择集合，且时间滞后。** 当前chunk输出pointmap后append_positions，用最近4帧等价有效voxel IDs（最后slot/head0读取，体素ID初写时全层/head相同）寻找邻域；不足recent预算时可取所有已有IDs。`neighbors(radius=.1)`换算Chebyshev网格半径ceil(radius/size)，返回该格立方邻域的现有体素，不应简称严格欧氏.1米球截断。该共同voxel集合送全部layer/head store，再形成各自代表输出；不输入当前每条Q作检索。检索条件为`frame_idx>max(11,16)`，chunk4时第一次在20开始的chunk输出后调用，**供下一chunk24的attention使用**；此前retrieval为空。[append_positions/_search_recent_neighbors，L294–366](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/stac_voxel.py#L294)；[voxel.neighbors，L495–524](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/voxel.py#L495)；[session实际顺序，L456–494](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stream_session.py#L456)。
5. **retrieval发生在本次prune之前。** 文首流程注释不能代替L478–494实际顺序；新被prune的对象本次尚未进入刚生成的retrieval。store返回K/V/mask/bias后按每layer/head最多`2*t`截断，覆盖旧retrieval。返回的invalid槽由bias屏蔽；`return_buf=True`先取pivot，只有输出长度余量才追加buffer；若pivot返回已填满目标shape，即使含padding，本Python分支也不再取buffer。**CUDA实际填充／优先排序／bias是否严格logC未知**；Python参考retrieve按W降序、有效pivot bias logC、buffer bias0，不能冒称CUDA已验证同式。[stac_voxel.retrieve_kv_parallel，L665–764](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/stac_voxel.py#L665)；[merger._merger_cuda_retrieve，L1261–1375](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/merger.py#L1261)；[Python参考retrieve，L1378–1543](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/merger.py#L1378)。

特殊camera/register共5token的源2D位置都是0，patch网格位置+1；首帧用第0套可学习特殊token，其余用第1套。manager对特殊和patch共同热缓存与HH打分；**特殊token的几何valid=False**，掉出hot后不能通过有效voxel进入远端库。独立camera head的pose-token历史另述，不能与global块第0token的热保留混为一谈。[aggregator.inference，L520–540及slice_expand，L652–680](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/causalvggt/models/aggregator.py#L520)；[session.get_pointmap，L238–256](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stream_session.py#L238)。

## 3. 融合、永久遗忘与完整backing

**激活CUDA路径的直接事实。** STAC在hot压缩前提取所有drop K/V、分数、原tokenID和voxelID；按layer×head展平送`VoxelKVMerger.insert_and_merge`。该函数先过滤无效voxel，再按head-voxel排序；原tokenID没有传进CUDA merge调用。segment分支将本次drop和旧overflow拼接，把`.8/.8/.2`阈值送`insert_and_merge_with_rows_seg`，保存其返回的overflow K/V/S/rows；随后hot及voxel索引同步压缩。[stac_voxel._pool_drops_vectorized_parallel，L421–512](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/stac_voxel.py#L421)；[merger.insert_and_merge，L1158–1247](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/merger.py#L1158)；[CUDA-seg wrapper，L1051–1138](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/merger.py#L1051)。

**明确分开的Python参考事实。** `backend='python'`的fusion不是当前默认CUDA实现：按有效K的余弦相似度匹配；sim≥.8合并，默认权重`exp(sim)`（不是score权重）；W加权平均K/V，S累加score、C累加数量。低sim且score高于`.2*row_average`进入buffer，buffer满后最高score种子聚成新pivot；pivot满时最小W旧pivot先加权并入最相似其他pivot，再覆盖其槽。K/V已含来源RoPE，参考代码不解RoPE、不按合并中心重新RoPE、不保存逐来源2D位置／精确成员表。低分过滤及均值融合分别造成token级丢失／不可逆压缩；一个对象掉出hot不等于立刻永久消失，因为仍可能在buffer、pivot、overflow或旧retrieval中。[merger.default_weight_fn，L43–46](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/merger.py#L43)；[参考聚合，L312–406](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/merger.py#L312)；[参考替换，L462–585](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/merger.py#L462)；[参考在线融合／筛选，L587–760及L766–915](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/merger.py#L587)。

**CUDA缺口与可确定遗忘。** 未取`merger-cuda`实现正文，不能由上述参考分支确认默认CUDA同样的tie-break、score gate、重并权重、计数更新、pivot替换或精确淘汰条目。固定tree仅确认缺失实现路径包括`merger-cuda/merger_cuda/__init__.py`、`merger-cuda/csrc/merger_wrapper.cu`和`merger-cuda/csrc/kernels/merger_pipeline.cu`／`merger_kernels.cu`；这里只提供精确缺口，不自动续取正文。能够直接确认的是：无有效voxel的drop在Python调用端被过滤，已压缩hot不再持有它；本次CUDA接口没有逐来源ID／位置恢复表。retrieval副本被下次覆盖，旧hot位置／分数尾部清空，session clear/reset重建manager和体素表。代码未显示在本入口从RGB重跑backbone以恢复这些token；但整段RGB依然可读，故**token状态中不可取回不等于完整系统已销毁全部原始图像信息**。不能把体素索引误当原token暗库，也不能忽略仍在的RGB及相机历史。[h2o.compact，L357–438](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/h2o.py#L357)；[stac reset/clear retrieval，L149–179、L368–375](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/stac_voxel.py#L149)；[session.clear，L87–101](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stream_session.py#L87)。

| 直接可见状态／必要副本 | byte依赖与界限（静态，非实测峰值） |
| --- | --- |
| hot全24层 | 设L=24,H=16,D=64,t=patch数+5,T=11t,b=KV每元素字节；KV为`2LHTDb`，tokenID int64、voxelID int32、两份FP32 score/count另`20LHT`；活跃预算7t不等于只分配7t |
| 远端库 | head数展平为LH；每体素最多4pivot/8buffer只是局部cap。voxel keys/centers、pack/sorted keys、zones和邻域索引随到访体素增长；`ensure_capacity`按需要或1.5倍增长，没有入口全局voxel淘汰阈值 |
| Python与CUDA pools | constructor先创建Python segment pivot/buffer字段及row indirection，再初始化CUDA merger；没有在该调用点释放Python pools。Python initial capacity4096 pivot／10000 buffer是初始分配参数，不能读成总量硬限。CUDA实际容量／workspace／共享storage和峰值unknown |
| retrieval／overflow／暂存 | 每layer-head最多2t输出复制、FP32 bias/score、mask；hot+retrieval cat/contiguous、drop提取、overflow拼接、排序/gather及池扩容会另占临时内存；overflow长度随输入／实现动态变化，不能免费排除 |
| camera | 默认不是有限camera KV窗口，而是全部2048维pose tokens list、cat副本，4轮×4trunk的全历史计算和本次临时激活／预测；没有在该入口按hot prune裁掉pose list |
| RGB／输出／CPU／磁盘 | main先全RGB到device再stride10；session transfer16只是视图／传输组织，不抹掉整段原输入。CPU累积world_points/conf/depth/conf/images及最终cat副本；最新全pose输出在session中为device tensor，get_all_predictions时另转CPU。热CPU缓冲只有reserved>gpu_limit250才分配非零，本配置11→0；alloc CPU offload=False。已读激活链未见磁盘KV backing或恢复checkpoint；外部调用者／扩展内部分配未知，不能由未读路径证明不存在 |

证据：[KVManager.__init__/reset，L72–89、L163–175](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/kv_manager.py#L72)；[stac reset，L149–179](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/stac_voxel.py#L149)；[merger.__init__/ensure_capacity，L103–198、L218–239](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/merger.py#L103)；[segment allocator，L2060–2189](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/allocator.py#L2060)；[voxel append-only表，L196–325](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stac/voxel.py#L196)；[session输出归档，L117–149](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stream_session.py#L117)。因此README“bounded”声明不提供我方B=2,726,555,844bytes定义下的证明；本轮既未运行也未测同byte容量。

## 4. 几何、额外pass与chunk权限

当前chunk只做一次backbone，**同一次forward的point_head输出world_points/conf**提供几何，双线性缩到patch网格后以conf>2过滤；不采用GT、外部相机、深度反投影或另一个几何模型。depth_head也在同forward执行，所以相对只调用depth的本地实现，point head成本不能省略。默认不激活`get_pointmap`的depth分支，本轮不修复或评价未激活分支。[CausalVGGT._forward_streaming，L190–249](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/causalvggt/models/vggt.py#L190)；[get_pointmap，L213–256](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stream_session.py#L213)。

pipeline先做**一次首帧warmup完整模型调用**，再登记manager；默认camera在正式循环被关闭。每chunk取global最后输出的camera token，detach后append全部pose_tokens_list、cat全历史，再独立camera_head.inference(mode='full',kv_cache_list=None)，重算全部历史pose；camera inference将mask设None，4轮×4trunk无KV列表分支。故不是每帧4轮有限camera状态，也不是仅补算本chunk相机。对于N个已采样帧，静态调用数为1次warmup+ceil(N/4)次chunk backbone，另ceil(N/4)次全历史camera head；未执行，不能将这个公式称实际调用／耗时。[pipeline warmup，L258–276](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stream_session.py#L258)；[camera_head_inference，L188–211](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/stream_session.py#L188)；[camera.inference/trunk_fn，L121–214](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/causalvggt/heads/camera_head.py#L121)。

**静态信息边界：** 当前chunk的4帧在global attention中互见，最早帧最多看到同chunk后3帧；几何及热分数也由该共同输出形成，供以后chunk使用。全历史camera重算又允许旧pose利用后来chunk的pose token。因此不能原样称我方旧12×96逐帧单到达、旧预测不回写任务的相同输入条件。transfer16和整段RGB预载本身没有在所读global调用中提前计算后续chunk backbone；不能把“可读全RGB”直接夸大成所有未来RGB已进入每次Q。改chunk1、限制camera前缀或改输出提交时点会改变激活条件／数值，科学可比性须明确列出，不能暗称纯接口适配。

## 5. 本地官方StreamVGGT：可见相同、不同与未知

本地文本仅对照[StreamVGGT构造，L18–26](/home/liwt/GRoPE/third_party/streamvggt/src/streamvggt/models/streamvggt.py:18)、[Aggregator构造，L50–133](/home/liwt/GRoPE/third_party/streamvggt/src/streamvggt/models/aggregator.py:50)、[Attention.forward，L44–87](/home/liwt/GRoPE/third_party/streamvggt/src/streamvggt/layers/attention.py:44)、[CameraHead，L71–150](/home/liwt/GRoPE/third_party/streamvggt/src/streamvggt/heads/camera_head.py:71)；4份小源码身份在[local_text_identities.json](local_text_identities.json)，没有读／哈希权重。

| 类别 | 固定文本支持的边界 |
| --- | --- |
| 直接相同的可见结构 | 两侧img518/patch14/embed1024；aggregator24个frame和24个global、16head、4register+camera、qk_norm=True、rope_freq100；camera dim2048/trunk4/4迭代；point DPT output4 inv_log/expp1、depth output2 exp/expp1。顶层aggregator/camera_head/point_head/depth_head名称对应 |
| 作者权重入口事实 | README指向`lch01/StreamVGGT`；wrapper选择`ckpt/streamvggt`或用户路径，支持safetensors/pt/pth；CausalVGGT构造中的base_model参数未用于切换结构。`strict=False`后仍拒绝missing keys、允许unexpected keys。作者默认不构造track_head，本地构造track_head；多余track权重可被该策略忽略。没有读checkpoint内部key或运行加载 |
| 缓存／位置直接差别 | 本地官方缓存raw K/V（norm/RoPE之前），读取拼历史后重新norm及重复源patch网格位置；作者缓存有效norm/sourceRoPE K并cast至V dtype。两侧特殊位置0、patch+1及首帧／后续特殊token选择意图对应；author不是按新体素中心重设PE。有效K与raw K的融合不可互换，也不能把FP32/BF16差异静默当逐位相等 |
| 相机／调用直接差别 | 本地官方inference逐帧，camera四轮复用其每trunk cache；作者此默认入口逐chunk加全部历史pose-token重算，不共享相同camera状态演化。两侧顶层输入／返回值不同，不能直接换类名运行旧runner |
| 静态可适配范围 | 张量转置、返回字段和manager调用接口、raw→effective读取位置有明确源码对应，是可能的工程接点；这仅为结构线索，不承诺完整模块命名/shape或输出相等。重做几何为depth+camera、用别的权重、变chunk/相机权限、删backing或改融合均超出无条件“语义保持适配”的声明 |
| unknown | 完整参数名／shape覆盖（作者Block、DPT、RoPE实现正文未全取）、checkpoint格式包装与实际权重身份／加载成功、扩展依赖能否在旧GLIBC运行、CUDA merger内部规则／bias与排序、实际峰byte及速度／精度。未读取路径不作不可行证明 |

作者证据：[CausalVGGT.__init__，L16–36](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/causalvggt/models/vggt.py#L16)；[aggregator构造，L57–148](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/causalvggt/models/aggregator.py#L57)；[load_model/load策略，L17–84](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/model_wrapper.py#L17)；[README checkpoint，L174–194](https://github.com/Rainzor/STAC/blob/fd7e718597cf9963de85c8fffae32a698e8619f5/README.md#L174)。

## 6. 交付范围与实际账

本轮服务O2空间组织、O3完整公共状态／成本、O4强基线信息边界；O5/O6没有新实验，C1–C6不作科学变更。入口的“原样可比／需改变条件／证据不足”竞争解释现有上述可核查事实，取舍返回原Astra；不选择后继。原C/I/G、整帧保存、PE等关闭与历史正负保持；[主控制目标](../autonomous_research_loop/goal_revision_after_round042/revised_goal.md)不变，全部旧数据used，无fresh或总体成功。

取证首操作 **2026-11-14 22:54:29 UTC**，45分钟截止23:39:29；实际完成时间／墙钟及新增字节见[execution.json](execution.json)。**17/24次显式请求、16/16份作者正文、467,805正文bytes**；其中1次完整tree元信息、16次文件正文，无重试／失败；连接器内部HTTP数不可见，明确unknown。新增证据总量另含tree、报告、身份和脚本，见终态账，远低16MiB。正文全部按固定tree Git blob SHA1核对，并记SHA256；序列化多出的末尾换行仅为恢复原blob而移除，未改作者源码。[来源身份](source_identities.json) · [取回账](retrieval_ledger.json) · [固定tree](repository_tree.json)。

**0 GPU调用或探测、0模型调用／预测／训练／数据权重下载、0作者模块导入、0依赖安装／CUDA构建。** 仅同步GitHub取文本、文件读写及stdlib AST／小文本身份核对；没有manager、后台CPU/GPU作业或需要恢复的owner句柄。源码中出现的CUDA调用只被当文本读取。未增加图、效果测试或旧评分审计。所有工具均同步返回，最终收据及唯一短交接见[terminal.json](terminal.json)与[implementation_handoff.json](implementation_handoff.json)。
