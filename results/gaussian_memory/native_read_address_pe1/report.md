# native_read_address_pe1：固定公共GS的地址偏置

9个配对1000步末步拟合、1056份实际读取缓存及6336份full/missing预测已完成。主A/P降误差 **-0.05170%**，95%[-0.3494116253747398, 0.24870905912100527]，三seed[-0.1194984003586752, -0.18103235118878125, 0.14098055239490925]，预定筛=False。A/B、Native／Local净用途和A/S形状范围分别见下表；结果交原Astra科学取舍，无fresh确认或总体成功。

[唯一科学合同](/home/liwt/GRoPE/results/gaussian_memory/autonomous_research_loop/round_019/science_plan.md) SHA256 `92e0880cbf45c0a8d566c605fb979a36145e4603d6449a8d6cc21761653dec3e`。[固定实测flags](method_decision.json) · [全数配对表](report_tables.md) · [逐预测与支持CSV](scoring/per_prediction_metrics.csv) · [全部summary](scoring/summary.json)。

## 问题与实际接口

检验共享槽贡献关系是否超过同完整读行的多尺度中心关系，竞争解释为中心足够、共同少量适配、训练片段记忆或内容偏置损害空间预测。原round018未执行的读行Gram就是本算子；正交槽标识的等价别名仍成立且不另拟合，不称Gaussian独有数学贡献。round016逐token原生任务成场自动变体保持关闭，全部历史正负保留。本轮不重训writer、场、fusion、旧PE或native。

每seed固定引用round015原C/P最终模型和44份一次形成保存场，共132份；A/P/S共享完全相同物理场、内容、原46维中心与完整普通logΣ6、原52→64→1024强中心PE、原24global Q/K和2D RoPE，全部旧参数冻结并保持eval。只训练24个FP32 gamma，每global block一个、同头共享、零初始化、可正负无新幅度界。旧writer只读依赖，不再执行；源场不接query、不优化或写回。

W用实际gsplat槽单位基颜色渲染，含有序alpha合成、opacity及有效内容mask。A=N(WWᵀ)，P=N(W Cμ Wᵀ)；Cμ为有效中心最近严格正距离中位数d决定的d/2,d,2d三等权RBF exp(−r²/2s²)，所有有效中心重合则全1；重复中心的0距离不作最近正距离。P同样保留完整W／槽维数，不截断为低维矩或随机特征。S只把PE影子scale改为sqrt(traceΣ/3)，位置／opacity／绑定与相机不变，真实内容／普通Σ仍用完整W。A/S包含覆盖、遮挡比例变化，不能称唯一方向性、位姿不确定性或Gaussian独占机制。

N除以sqrt(Huu Hvv)，0对角行列先置0，然后每patch-query行在所有patch-key上减均值；乘gamma后加原已缩放QK logit，相机／register行列增量全0。对所用核H零对角的query行无新偏置（A/P是实际W，S是影子WS），原实际零覆盖样本全部保留；0支持key列在行去均值后可能是负均值偏移，按同一明确定义实现，不额外删key或重新归零破坏行中心化。其他位置改变可经原注意力传播，不能承诺零覆盖query最终预测与B不变。

1056个(seed,place,view)各只构建一次共同content／旧PE量和三核，W／WS／Cμ瞬态丢弃；原字段SHA与query输入SHA绑定。训练与正式评价可读cache；成本另实算完整新view W／核，绝不称cache免费。三组实际内容与中心／普通Σ逐值共同，S的影子仅位置核，不用其支持改评价mask。

O2/O3复用公共粗GS，O4/O5真实读取，O6比较通用GS地址PE与同完整输入强中心；C1/C2/C3/C5保持，C4新增成对偏置、C6仅标量学习。给定相机／unit、sensor／joint-validity、静态四源、2048槽1024D、depth／人工缺失／原ROI和冻结接收器均为临时范围，不晋升用户核心要求。

## 固定训练与必要工程修复

原32训练／12已用开发，四源0/3/6/9；训练query2/5/8/11、其他1/4/7/10、原full/missing/valid/ROI保持。三个原配对seed20261110/11/12各A/P/S，9×1000、batch2、仅末步，无开发适配或早停；取原seed schedule前2000样本，逐样本、顺序和配对初值核验。Adam LR.01、wd0、clip5，原全有效图raw与centered logMSE等权，FP32gamma/Adam、nativeBF16。旧大权重只引用，不复制为九份final；永久final只含gamma、Adam、RNG、step及采样／B权重身份。

首个预检在反传遇到已安装fused偏置梯度的LSE strideH对齐错误：5次native、2次raster、20.243秒，0优化器更新，完整保留计账。global块autograd现固定PyTorch MATH SDPA，维持nativeBF16和同一定义的FP32偏置；frame块及no_grad评价沿原自动后端。成功重试7次native、1次丢弃pilot优化器更新。A/P/S零gamma评价对原C/P深度逐值maxdiff=[0.0, 0.0, 0.0]，共同读取量maxdiff={'extra': 0.0, 'position': 0.0, 'alpha': 0.0}。训练MATH后端零gamma与fused评价的metric-depth最大差0.02507401m（该预检旧训练case）；不称训练算术逐值回放。该共同实现没有按分数选择后端或调科学条件，完整浮点差异交回评估。

仅24gamma有梯度，旧参数requires_grad=False、version及grad在实际更新前后保持；gamma/Adam/RNG完整恢复预测maxdiff=0.0。真实A核与FP64独立式最大差3.576e-07；正最近距离、全重合、0行列、行去均值和special0的有限例验证。该预检不证明用途。[预检](preflight.json) · [工程修复与全部失败](engineering_revisions) · [旧源码/配置](runtime_versions)。

准备与训练使用实时真空闲卡，保护外部PID；自有GPU清退／新外部批次有独立观察记录。新或曾含外部任务的卡需60秒持续真空闲；自有连续观察安全且正常清退的卡可当前live确认后复用。初轮阶段切换出现保守等待，GPU4预检后安全复用标记未保留而再观察60秒；该瞬时API返回未存档，具体清退PID／驱动原因未知。后续GPU6正常交接实际仅约0.247秒。未把空闲时间伪装GPU工作。唯一旧manager实际exit75/allchildrenwaited后才顺序恢复；不复制manager、fit或有效更新。全部失败／加载／缓存I/O与回放时间按GPUworker完整存活墙钟计算。

## 预定主量与用途

主量为十二开发地点原四query的missing ROI centered log-depth MSE，query先地点内等权，5000次配对seed/place bootstrap（20261114）。实用筛为平均降误差≥1%、95%下界>0、三seed正。A/P、A/B及A/Native、A/Local必须分别成立；A/B包括新增成对计算及标量适配，不能代替强中心增量或原生净用途。A/S仅限定形状主张，不以其不显著自动否决通用地址；跨零不称等价／非劣性。

|比较|降误差%|95%区间|三seed%|预定筛|
|---|---:|---|---|---|
|A-P|-0.05170|[-0.3494116253747398, 0.24870905912100527]|[-0.1194984003586752, -0.18103235118878125, 0.14098055239490925]|False|
|A-S|+0.04094|[-0.6185544740080862, 0.72207023575735]|[0.714025332647672, -0.3493438928102277, -0.20010614091017231]|False|
|P-S|+0.09259|[-0.5756520099313783, 0.8226036385940746]|[0.8325288743190162, -0.16800739388611596, -0.34156823809394776]|False|
|A-B|-0.02237|[-0.47908226604927884, 0.41901070881301145]|[-0.06571017304952687, -0.11323432586212157, 0.10913188302605903]|False|
|A-Native|+14.23598|[-10.715784079409707, 27.694726500820636]|[18.382035638956552, 11.9112060665992, 12.414707643650157]|False|
|A-Local|+8.60276|[-17.873119110104287, 25.73447913856678]|[12.381324602154587, 4.474101347794579, 8.901737097473262]|False|
|A-Render|-0.17084|[-12.823326448247036, 7.516351416773274]|[0.2723464424130517, -4.093339533714202, 3.1002830451782315]|False|
|P-B|+0.02931|[-0.5047400333775642, 0.4612106828223507]|[0.05372402795513764, 0.0676755107583693, -0.031893633189095105]|False|
|P-Native|+14.28030|[-10.580434819168982, 27.931015868790507]|[18.479451390508515, 12.070387112201185, 12.291055088614344]|False|
|P-Local|+8.64999|[-17.824725995689008, 25.84293436653596]|[12.485902548697236, 4.646721629564166, 8.77312494508823]|False|
|P-Render|-0.11909|[-12.726140096582721, 7.61630067988389]|[0.39137715333412704, -3.9052374393694134, 2.9634804238549917]|False|
|S-B|-0.06334|[-0.9025360758961476, 0.5439409718312426]|[-0.7853430540513129, 0.23528760407274252, 0.3086204554527572]|False|
|S-Native|+14.20085|[-10.731644163115304, 27.575345036109702]|[17.795071625679025, 12.217867585168996, 12.589621179462895]|False|
|S-Local|+8.56533|[-17.807755791540046, 25.59569086119223]|[11.751205856211843, 4.806653490188273, 9.08366626436864]|False|
|S-Render|-0.21188|[-12.903367669466085, 7.492473689819902]|[-0.44485526955284627, -3.7309617538737516, 3.2937980938334555]|False|

[实际环境](runtime_environment.json) · [运行配置及原始身份](configuration.json)。以下三组固定首批比较只用已有B预测作CPU参考，不新增模型调用；共同MATH训练与fused评价的浮点差不宣称全局等价。

- seed20261110：原B首批损失0.00316386402；零gamma MATH训练{'A': 0.003154823905788362, 'P': 0.003154823905788362, 'S': 0.003154823905788362}；相对差%={'A': -0.28573021282216815, 'P': -0.28573021282216815, 'S': -0.28573021282216815}。
- seed20261111：原B首批损失0.00257250364；零gamma MATH训练{'A': 0.0025710888439789414, 'P': 0.0025710888439789414, 'S': 0.0025710888439789414}；相对差%={'A': -0.054996828167788794, 'P': -0.054996828167788794, 'S': -0.054996828167788794}。
- seed20261112：原B首批损失0.00279336446；零gamma MATH训练{'A': 0.0027793197659775615, 'P': 0.0027793197659775615, 'S': 0.0027793197659775615}；相对差%={'A': -0.5027877269167035, 'P': -0.5027877269167035, 'S': -0.5027877269167035}。

训练原12／新增20／全部32、开发其他query及八query pooled、full/missing、raw／signed bias／bias²、wholeAbsRel、原GT支持、原零／正覆盖及当前完整W支持均在全数表／summary／CSV。后者在同seed A/P/S/B共同，影子S不改内容支持。空支持区域的指标为null，仅在该分层聚合时略过，无样本从完整图或主量删除；逐预测像素数完整保留。关键辅助保留：

- train32/fit A-P: +0.0469% [-0.11036479678152022, 0.24251630420424602]。
- train32/fit A-S: +0.1902% [-0.08419828376783123, 0.5607855249240763]。
- train32/fit A-B: -0.3811% [-1.3287467275436715, 0.6244286302446832]。
- train32/fit A-Native: +63.6708% [56.70599085749351, 69.33172308022333]。
- train32/fit A-Local: +42.4440% [35.66400823695268, 49.158381882326545]。
- train32/fit A-Render: +5.6756% [2.4896040660737975, 9.092653881155805]。
- train32/other A-P: -0.0124% [-0.26890598423811085, 0.23508632136303376]。
- train32/other A-S: +0.2860% [-0.12123136618900299, 0.9917394513474298]。
- train32/other A-B: -0.1052% [-0.8759689460915582, 0.5972979840716015]。
- train32/other A-Native: +62.5237% [56.01267884582411, 67.87559470629262]。
- train32/other A-Local: +44.6685% [38.40863414013498, 50.08076453294153]。
- train32/other A-Render: +4.6903% [2.270040020402049, 6.90165375803639]。
- development12/other A-P: -0.0152% [-0.387896129891106, 0.4230626869174213]。
- development12/other A-S: +0.0194% [-1.3155601709054676, 1.4708384678674042]。
- development12/other A-B: +0.2524% [-0.5713694106624906, 1.1065114652613874]。
- development12/other A-Native: -9.9130% [-72.77241739333128, 30.32344944044957]。
- development12/other A-Local: -4.1524% [-35.62860185347671, 28.676055723871663]。
- development12/other A-Render: -8.0870% [-19.868718251868568, 6.548853059127475]。
- development12/fit missing/roi/log_mse A-P: -0.0399% [-0.789448035923036, 0.8881000683029441]。
- development12/fit missing/roi/log_mse A-S: +0.0957% [-1.3451450150528177, 2.0747768507115434]。
- development12/fit missing/roi/log_mse A-B: +1.8629% [-3.250288724388674, 9.401413911051419]。
- development12/fit missing/roi/log_mse A-Native: +92.1536% [84.27923628851033, 96.21455142847763]。
- development12/fit missing/roi/log_mse A-Local: -22.5617% [-126.84032804938342, 9.58692274008247]。
- development12/fit missing/roi/log_mse A-Render: -30.1446% [-111.62627547683705, -1.7079150481243819]。
- development12/fit missing/roi/bias_squared A-P: -0.0366% [-1.0876241288989508, 1.1536252722253064]。
- development12/fit missing/roi/bias_squared A-S: +0.1108% [-1.7054790234685462, 2.7532411121082507]。
- development12/fit missing/roi/bias_squared A-B: +2.3712% [-4.662794831858373, 12.496774894142742]。
- development12/fit missing/roi/bias_squared A-Native: +93.7276% [85.91774353280785, 97.5061000391133]。
- development12/fit missing/roi/bias_squared A-Local: -35.3046% [-255.19273969334807, 10.07223823139019]。
- development12/fit missing/roi/bias_squared A-Render: -41.8698% [-172.53280184267732, -3.1026370985210123]。
- development12/fit missing/whole/absrel A-P: -0.0917% [-0.6265965842162535, 0.549732470102463]。
- development12/fit missing/whole/absrel A-S: -0.0385% [-0.8274847227431092, 1.1083243429451057]。
- development12/fit missing/whole/absrel A-B: +1.9272% [-3.883743580730684, 9.531677501963205]。
- development12/fit missing/whole/absrel A-Native: +58.9540% [33.42567383813945, 73.83684303897054]。
- development12/fit missing/whole/absrel A-Local: -17.5814% [-63.15640667596773, 5.186219749730253]。
- development12/fit missing/whole/absrel A-Render: -15.7590% [-43.49990046880082, -1.3611976332025137]。
- development12/fit full/whole/absrel A-P: -0.0790% [-0.6990819641222795, 0.6168742164192973]。
- development12/fit full/whole/absrel A-S: -0.1107% [-0.96590609628462, 1.2837798202418735]。
- development12/fit full/whole/absrel A-B: +1.5664% [-5.226482508016369, 9.423416582200414]。
- development12/fit full/whole/absrel A-Native: +61.6980% [36.388816885766424, 75.38681627989148]。
- development12/fit full/whole/absrel A-Local: -21.0805% [-67.14822942463894, 1.8464826571490094]。
- development12/fit full/whole/absrel A-Render: -16.4358% [-45.85045673526851, -2.1442884912830436]。

逐seed／地点（原四query均值）最大反例，不能由总体均值掩去：

- missing/roi/centered_log_mse/A-P：最大绝对增误差seed20261110／fidelity_01／brown_bm_4，test=0.06191232、reference=0.061381、差=+0.0005313188；最大相对退步seed20261110／fidelity_11=-2.4602%降误差。
- missing/roi/centered_log_mse/A-Native：最大绝对增误差seed20261111／fidelity_05／home_at，test=0.01782894、reference=0.008101368、差=+0.009727573；最大相对退步seed20261111／fidelity_05=-120.0732%降误差。
- full/whole/absrel/A-Render：最大绝对增误差seed20261110／fidelity_03／brown_cs_3，test=0.3838739、reference=0.1164166、差=+0.2674573；最大相对退步seed20261110／fidelity_03=-229.7416%降误差。

相对B/Native/Local/Render的full/whole AbsRel增误差≥1%且区间支持的材料代价flag全部保留在method_decision.json；微小不确定负均值不作硬零容差，训练内／raw／bias正证据不替换主量。

全部原整query零覆盖共10，身份与支持mask未筛除；逐seed地点的主量／fullAbsRel最大绝对及相对反例见[完整反例](scoring/counterexamples.json)，A/P相对最佳及固定例见[选择](figures/case_selection.json)。两图已实际查看并记录[SHA及具体观察](figures/figure_review.json)。例图仅seed20261110，排名用三seed均值，target2–98百分位行色域会裁剪饱和；相对best不是绝对深度准确。

## 实际调用、资源与成本

独立Torch/NumPyfloat64复算全部6336新预测，最大指标差3.109e-15；B2112、Native704、Local/Render各2112按原相同输入输出与独立核验身份继承，不重复训练或旧大权重审计。[完成检查](completion_checks.json) · [输入依赖身份](input_manifest.json)。

实际GPUworker 2.436692/6小时；9000正式有效更新、18000样本，正式丢弃0；失败预检0优化器更新、成功预检1丢弃更新单列。1056准备view／2112实际raster，6336评价native；总24418 native／2176 raster（含两次预检共4raster）、134 source-Cμ及2个小toy Cμ公式调用；成本70完整query／70新encoder／60raster／2source-only中心核构建，0新writer构建。全部细计数、worker实际wait及失败在[budget_actual](budget_actual.json)。

新DATA持久17.796249GiB，含八路原子暂存及128MiB余量的保守峰值18.171249/24GiB；项目保守上界136.59/1024MiB。未使用自有RAM恢复文件，普通工作张量另按CPU/GPU内存准入；完整小gamma/Adam/RNG恢复与所有final/缓存/预测均永久DATA，原大权重／场保持原路径。

|方法|暖完整fresh query ms|新view W/普通量/核 ms|native/fusion/旧PE/新bias/DPT ms|source Cμ ms|峰值GiB|
|---|---:|---:|---:|---:|---:|
|Native|105.887|0.000|90.252|0.000|4.004|
|Local|106.186|0.000|90.565|0.000|4.012|
|Render|111.647|6.680|88.957|0.000|4.020|
|B|124.981|18.967|90.443|0.000|4.031|
|A|126.660|19.337|91.133|0.000|4.031|
|P|126.920|19.788|91.246|49.251|4.047|
|S|138.677|31.531|91.295|0.000|4.039|

[完整分段／source Cμ／view-cache写读与N摊销](cost.json)：七方法×两固定case×(一warmup＋四保留)=70完整query；固定seed20261110，第一训练q2missing、第一开发q1full。每次都从准备RGB/mask做真实queryencoder、新W／内容／核及完整native/DPT，S另做影子raster；无预缓存核冒充完整读取。图像/mask GPU准备在begin前单列；完整wall含最终resize和GPU→CPU结果传输。原保存预测只用于计时后的独立回放比较，不进入模型。同步及记录开销包含在分段wall，模型／encoder冷加载与大权重逻辑reload另列，不保证硬件独占或OS冷盘。

P的Cμ只由source中心构建，单独计一次source准备并按N摊销；A/P/S新增每view W／核仍在每个freshquery中，不因缓存或标量少就免费。单arm新view-cache的CPU传输／原子保存fsync／逻辑reload和bytes也实测单列。原132来源场不重建：沿round015同seed C/P的一次writer／持久化实测依赖及本轮源场逻辑reload，另显示历史sensor构建3.230/3.155秒参考的N=1/4/8/32摊销；该跨轮参考不含AnySplat预测生成、sensor采集、图像预处理及模型冷加载，不能称端到端RGB速度。

## 身份、历史及交接

round018槽标识等价与一阶内容换基结论不改，round016及更早特征形成／工作副本／参数字段PE家族关闭状态不改；本次只检验同一读行算子的通用地址用途。shared source-state utility、A/P编码增量、A/B一般适配、A/S形状范围、Native/Local净用途与fresh确认独立，任何正辅助不追认旧主终点成功。

所有地点/query均已用development，没有fresh候选或已核验reserve。本实现只交实测flags，不选择后继、解冻、增层、增seed/step或扫描；科学取舍交主复用原Astra。当前/outline/O-C/证据入口原位同步，完整结果及历史负例保留。

实际复用原恢复角色/root/research_implementation_sol_recovery，创建请求gpt-6.1-sol/high；followup无model/effort/Fast字段，未请求或核验Fast，后台effective独立unknown。双向顺序自主授权保持；未建agent/goal、未改main state/events或自派后继。全部GPU/CPU工作实际退出后，本turn结束释放科学写租约。
