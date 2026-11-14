# native_sampling_scope1：自然采样尺度条件范围诊断

**固定H条件在旧数据中只有一个开发地点，跨地点用途区间不可估。** 原四query的H/L/U为1/40/7；H是home_bksh的q2。H内G/P点降幅+2.50212%，三seed两正一负；G/Native点降幅+49.34306%，均不能替代缺失的跨地点支持。H−L直接交互仅同一地点，区间也不可估。阈值、模型和原ROI均未移动，H预定条件用途筛未获支持，科学取舍交原Astra。

[唯一科学合同](/home/liwt/GRoPE/results/gaussian_memory/autonomous_research_loop/round_021/science_plan.md) SHA256 `da48ea1a0d3d598145877de2b21139e1ddf2961ec4be5ffc2c9ac2f034f3f52b`。[完整标签](sampling_labels.json) · [事先计数](sampling_counts.json) · [标签完成提交](labels_complete.json) · [全分层统计](summary.json) · [全部配对CSV](all_comparisons.csv) · [实测flags](method_decision.json)。

## 问题、固定对象与顺序

检验来源描述细于查询采样时，原有形状PE是否更有原生用途；竞争解释为共同渲染或强中心已处理尺度、尺度伴随遮挡／难度、或上下文特征不能按局部滤波解释。Mip-Splatting与Neural Gaussian Scale-Space Fields仅提供定义采样条件的原始依据，渲染／经验滤波成绩不是Transformer PE证据；完整来源与推论见唯一合同，不继续检索或下载。

唯一旧对象为native_feature_field_pe1的J行，12个开发地点的公共场、全部三seed原末步P/S/G，以及同场RasterFill和原Native。按形成机制固定J，不按本轮分数选C/J或历代模型。所有PE组和seed共用每个来源场；μ/opacity固定，f/Σ来源拟合后冻结；普通完整Σ、中心邻域／多尺度输入、reader容量和旧训练预算匹配。这里PE在memory-reader，输出经真实VGGT，不称原生24层新PE。旧训练家族仍关闭。

逐有效内容中心，在来源相机正深度且投影在原448×448图像边界内时取sqrt(fx*fy)/z，四源0/3/6/9取最大。查询只选中心位于原[112,336)×[112,336)缺失窗口且有来源率的槽，取log2来源率／查询率的中位数。r≥1为H，r<1为L，无合格中心为U；不用Σ、opacity阈值、RGB、query真值／valid／误差或新可见性阈值。投影为给定c2w逆变换的相机坐标，原448图像intrinsic，全程FP64几何算术；没有遮挡测试或表面密度估计。

先由独立labels进程保存全部96标签及计数、fsync和SHA提交，再实际exit0；提交中score_reads=0，标签产物设为只读。随后aggregate进程才连接旧分数，提交时间严格早于连接时间。标签阶段仅消费12场的中心／有效槽／四来源相机及96个查询相机和原窗口尺寸；mmap到CPU，RGB／target／有效深度／features／Σ均不被读取运算。四个有限解析例覆盖H等号、L单位比、无窗口中心及负深度U；不按结果调阈值。

12开发×8query共96标签；32训练地点不参与新子组扫描。已有源场／原九份末步权重只引用身份，不加载模型、不复制场、不重哈希大权重。1056条missing旧逐query评分（P/S/G各288、Native／RasterFill各96）原值连接；96条Native_full只作共享构造引用。0新像素指标复算、模型预测、训练、渲染、CUDA或下载。

## 标签与样本支持

|查询组|H query／地点|L query／地点|U query／地点|总query|
|---|---:|---:|---:|---:|
|fit|1/1|40/12|7/4|48|
|other|1/1|40/12|7/4|48|

|地点|原四query H/L/U|其他四query H/L/U|
|---|---|---|
|fidelity_00 / brown_bm_1|0/3/1|0/3/1|
|fidelity_01 / brown_bm_4|0/4/0|0/4/0|
|fidelity_02 / brown_cogsci_1|0/2/2|0/2/2|
|fidelity_03 / brown_cs_3|0/1/3|0/1/3|
|fidelity_04 / harvard_robotics_lab|0/4/0|0/4/0|
|fidelity_05 / home_at|0/3/1|0/3/1|
|fidelity_06 / home_bksh|1/3/0|1/3/0|
|fidelity_07 / home_md|0/4/0|0/4/0|
|fidelity_08 / mit_32_d507|0/4/0|0/4/0|
|fidelity_09 / mit_32_g725|0/4/0|0/4/0|
|fidelity_10 / mit_46_6conf|0/4/0|0/4/0|
|fidelity_11 / mit_w59_conference|0/4/0|0/4/0|

H两条全部标签：

- fidelity_06 / home_bksh / q2 (fit)：r=1.7621924，合格中心401／有效槽2048。
- fidelity_06 / home_bksh / q1 (other)：r=1.7831693，合格中心418／有效槽2048。

来源视锥与固定query窗口只给几何采样代理，不说明遮挡、精确footprint频率、真实抗混叠机制或独立因果。U不是零覆盖别名，且不被筛除。

## 主量、对照与直接交互

旧missing-ROI centered log-depth MSE不在像素内重新选ROI或去均值。每地点内该条件query等权，再地点等权，三原seed配对；5000次seed／地点bootstrap，种子20261113沿原。少于两个地点时跨地点区间记不可估，绝不以三seed重采样冒充。S/P与G/S分别限定一般尺度／非球形解释，G/Native与同场RasterFill检验用途。

|查询组/条件|比较|地点|降误差%|95%跨地点区间%|三seed%|
|---|---|---:|---:|---|---|
|fit/H|G-P|1|+2.50212|不可估（少于2地点）|[0.9181595688637634, 6.528081269812969, -0.5368254450886845]|
|fit/H|S-P|1|+3.45658|不可估（少于2地点）|[1.181274477235661, 8.53105814048516, -0.06652219673934923]|
|fit/H|G-S|1|-0.98864|不可估（少于2地点）|[-0.26626017182471706, -2.1897890474654513, -0.4699906002775478]|
|fit/H|G-Native|1|+49.34306|不可估（少于2地点）|[48.27366257487677, 47.37930131716459, 52.37621675728599]|
|fit/H|G-RasterFill|1|+1.22146|不可估（少于2地点）|[-0.8638149956793129, -2.6077754794032293, 7.135963969210135]|
|fit/L|G-P|12|-0.02805|[-2.13643, +2.11509]|[2.0385579767894235, -1.6745301433713111, -0.5158984571648206]|
|fit/L|S-P|12|+0.35133|[-1.71613, +3.86820]|[2.5052430873025, -1.0234141510997885, -0.4984733850890377]|
|fit/L|G-S|12|-0.38072|[-2.45413, +1.61815]|[-0.47867713638278797, -0.6445198845662192, -0.01733864355233372]|
|fit/L|G-Native|12|-6.10517|[-44.02595, +14.40841]|[-6.161743439896572, -6.694317919407133, -5.4594601628848105]|
|fit/L|G-RasterFill|12|+9.70297|[-0.99748, +22.13377]|[9.654831773948047, 9.201603243733459, 10.252484927976102]|
|fit/U|G-P|4|+1.24615|[-7.29825, +12.18263]|[2.1295008589034548, -2.493606609303445, 4.441006927204382]|
|fit/U|S-P|4|+1.36344|[-8.29154, +12.22389]|[-2.674080719789207, 7.396506951085091, -1.02321204711473]|
|fit/U|G-S|4|-0.11891|[-10.01883, +7.39126]|[4.678475370821444, -10.680065335293975, 5.40887471660546]|
|fit/U|G-Native|4|-71.39927|[-357.88771, -31.00579]|[-70.62091778265271, -86.30833861525925, -57.26856603551018]|
|fit/U|G-RasterFill|4|+17.20467|[-2.77615, +27.84787]|[17.58066003611195, 10.002768136560626, 24.03058441975071]|
|fit/all|G-P|12|+0.08566|[-2.87034, +3.11668]|[1.3828242385991008, -2.4592668426757225, 1.3530895246013808]|
|fit/all|S-P|12|+0.58574|[-1.71430, +3.18104]|[0.2993080242593149, 1.8441494087882204, -0.4060740709731503]|
|fit/all|G-S|12|-0.50303|[-5.44126, +4.09981]|[1.086769001165444, -4.384268717089856, 1.752048978960219]|
|fit/all|G-Native|12|-20.13399|[-65.56358, +4.88281]|[-19.926277895840848, -23.840402784558346, -16.635287150989186]|
|fit/all|G-RasterFill|12|+10.45386|[-1.78716, +19.51090]|[10.60868222713115, 7.691149990919078, 13.061739260353777]|
|other/H|G-P|1|+1.33064|不可估（少于2地点）|[1.9128925059284763, 1.6754133918075897, 0.3630963052453251]|
|other/H|S-P|1|+4.55019|不可估（少于2地点）|[2.4843746497557495, 2.132384283575972, 9.294019771742123]|
|other/H|G-S|1|-3.37304|不可估（少于2地点）|[-0.5860416131001633, -0.4669275821457264, -9.846013949711473]|
|other/H|G-Native|1|-33.13072|不可估（少于2地点）|[-29.200380790855473, -40.482980617359644, -29.708802314412974]|
|other/H|G-RasterFill|1|-6.28330|不可估（少于2地点）|[-3.14555811987165, -12.15288494835658, -3.5514502812297843]|
|other/L|G-P|12|+2.80171|[-3.19763, +7.43828]|[6.199742093338201, 0.6571206859855128, 1.3102127142702091]|
|other/L|S-P|12|+2.88184|[-2.86560, +7.99805]|[6.543662064068179, -0.08759398238207261, 1.9438984553662997]|
|other/L|G-S|12|-0.08250|[-1.00440, +0.86167]|[-0.3680006924364543, 0.7440629140297417, -0.6462481488799909]|
|other/L|G-Native|12|-23.26871|[-78.13329, +28.89943]|[-24.508897682929167, -24.068632862043703, -21.228601816510384]|
|other/L|G-RasterFill|12|+9.22069|[-8.01191, +17.16606]|[8.307371268718544, 8.631597404489955, 10.723093812293294]|
|other/U|G-P|4|-3.98151|[-17.84775, +5.33317]|[3.791093485876307, -16.833528517752192, 0.5650393950717425]|
|other/U|S-P|4|+0.20818|[-6.52819, +13.38461]|[2.4045896092360852, -0.7415680382028667, -1.1984019972878457]|
|other/U|G-S|4|-4.19843|[-18.67114, +5.82903]|[1.4206650405882648, -15.973506063998322, 1.7425585360595353]|
|other/U|G-Native|4|-78.23975|[-471.61220, -36.76375]|[-72.69366867650375, -95.88758164567564, -66.13799581672188]|
|other/U|G-RasterFill|4|+0.09827|[-29.57915, +19.21334]|[3.2067974006380395, -9.79317610332675, 6.881191350080718]|
|other/all|G-P|12|+0.59159|[-8.10373, +6.04832]|[4.661178016519929, -4.284742793566432, 1.1989573494979422]|
|other/all|S-P|12|+1.83208|[-3.16554, +6.91513]|[4.161770390781417, -0.29372737610884325, 1.504595335864256]|
|other/all|G-S|12|-1.26363|[-7.55696, +2.18129]|[0.5210943772384402, -3.9793270445428464, -0.3103068487393168]|
|other/all|G-Native|12|-27.71540|[-83.04171, +14.40889]|[-27.17210227850291, -32.344254992718355, -23.629840701598127]|
|other/all|G-RasterFill|12|+5.47504|[-11.99919, +12.92535]|[5.877149588336506, 2.049126401613799, 8.498854746531014]|

预定实用开发筛仍是均值降幅≥1%、95%下界>0、三seed同向；H内G/P与G/Native必须分别成立。H区间不可估，且H的G/P第三seed为负，不能以两个正点或Native大点改善保留用途线索。G/S的单地点点值为负也不证明球形总体更优。

|直接交互（同有H/L地点）|地点数|H−L G/P降幅差（百分点）|95%区间|三seed差|
|---|---:|---:|---|---|
|fit|1|+2.76294|不可估（少于2地点）|[-1.21907642484812, 8.116274653549183, 0.5292136963975125]|
|other|1|+8.58882|不可估（少于2地点）|[10.869237942480314, 3.4385558871801836, 11.374560953463842]|

交互只在home_bksh直接估计配对H/L差；不是不同地点集合H与L总体点值相减，也不从显著性差异推交互。其区间不可估，不说明收益确实集中在缩小条件。辅助H/q1对Native点降幅−33.13072%，三seed均负，也不由原H/q2的正点遮蔽。条件观察不是采样率唯一因果。

## Raw、偏差、全图代价与零覆盖

所有旧区域（ROI／whole／visible、原几何支持／非支持及零覆盖）、六个原指标均完整保存在summary及CSV。log_bias保留有符号均值与绝对差，不计算有符号偏差的误差降幅百分比；空支持区域为null，仅该区域聚合略过。下表保留原四query中G/P和G/Native／RasterFill关键辅助，不替换主量。

|条件|区域/原指标|比较|降误差%|95%区间%|
|---|---|---|---:|---|
|H|roi/log_mse|G-P|+0.04846|不可估（少于2地点）|
|H|roi/log_mse|G-Native|-2.05361|不可估（少于2地点）|
|H|roi/log_mse|G-RasterFill|+0.61957|不可估（少于2地点）|
|H|roi/bias_squared|G-P|+0.04759|不可估（少于2地点）|
|H|roi/bias_squared|G-Native|-2.08941|不可估（少于2地点）|
|H|roi/bias_squared|G-RasterFill|+0.61936|不可估（少于2地点）|
|H|whole/absrel|G-P|+0.01358|不可估（少于2地点）|
|H|whole/absrel|G-Native|-1.22115|不可估（少于2地点）|
|H|whole/absrel|G-RasterFill|+0.19350|不可估（少于2地点）|
|L|roi/log_mse|G-P|-0.09707|[-0.78631, +0.39133]|
|L|roi/log_mse|G-Native|+0.91183|[-2.19235, +4.15462]|
|L|roi/log_mse|G-RasterFill|+0.64251|[-0.11073, +1.41450]|
|L|roi/bias_squared|G-P|-0.09844|[-0.83291, +0.43043]|
|L|roi/bias_squared|G-Native|+1.04119|[-2.30052, +4.68849]|
|L|roi/bias_squared|G-RasterFill|+0.44506|[-0.27905, +1.16007]|
|L|whole/absrel|G-P|-0.04240|[-0.16438, +0.06808]|
|L|whole/absrel|G-Native|-1.00059|[-2.59599, -0.09143]|
|L|whole/absrel|G-RasterFill|-0.16879|[-0.52289, +0.09419]|
|U|roi/log_mse|G-P|+0.23323|[-2.42853, +1.92715]|
|U|roi/log_mse|G-Native|+5.48647|[-2.46358, +17.62327]|
|U|roi/log_mse|G-RasterFill|+0.53812|[-10.77151, +17.35301]|
|U|roi/bias_squared|G-P|+0.16557|[-3.35581, +1.86097]|
|U|roi/bias_squared|G-Native|+8.20711|[-0.75565, +22.27356]|
|U|roi/bias_squared|G-RasterFill|-0.80259|[-14.54446, +17.43829]|
|U|whole/absrel|G-P|-0.33605|[-2.31383, +0.74128]|
|U|whole/absrel|G-Native|+2.05140|[-0.18633, +7.42276]|
|U|whole/absrel|G-RasterFill|-5.24888|[-11.40610, -1.39079]|
|all|roi/log_mse|G-P|+0.05527|[-0.30798, +0.45641]|
|all|roi/log_mse|G-Native|+1.15246|[-1.55479, +3.58644]|
|all|roi/log_mse|G-RasterFill|+0.34263|[-1.26019, +1.66630]|
|all|roi/bias_squared|G-P|+0.05452|[-0.35544, +0.47385]|
|all|roi/bias_squared|G-Native|+1.58245|[-1.30970, +4.30659]|
|all|roi/bias_squared|G-RasterFill|+0.06439|[-1.74655, +1.44963]|
|all|whole/absrel|G-P|-0.04584|[-0.29140, +0.12261]|
|all|whole/absrel|G-Native|-0.46333|[-1.23587, +0.01207]|
|all|whole/absrel|G-RasterFill|-0.70633|[-1.63824, -0.03207]|

|条件|P ROI有符号log_bias|G ROI有符号log_bias|Native ROI有符号log_bias|RasterFill ROI有符号log_bias|
|---|---:|---:|---:|---:|
|H|-1.45424|-1.45389|-1.43894|-1.45842|
|L|-0.95089|-0.95104|-0.95740|-0.95027|
|U|-0.81690|-0.83064|-0.90214|-0.77699|
|all|-0.98192|-0.98268|-0.99498|-0.97620|

相对RasterFill的主量或missing全图AbsRel增误差≥1%且区间支持才记材料伤害，小而不确定代价不加硬零容差。完整flags如下，False在H中含“不可估”，不能读作已无伤害。

```json
{
  "all": {
    "G_P_practical": false,
    "G_Native_practical": false,
    "G_S_practical": false,
    "conditional_development_line_only": false,
    "RasterFill_primary_material_harm": false,
    "RasterFill_missing_whole_AbsRel_material_harm": false
  },
  "H": {
    "G_P_practical": false,
    "G_Native_practical": false,
    "G_S_practical": false,
    "conditional_development_line_only": false,
    "RasterFill_primary_material_harm": false,
    "RasterFill_missing_whole_AbsRel_material_harm": false
  },
  "L": {
    "G_P_practical": false,
    "G_Native_practical": false,
    "G_S_practical": false,
    "conditional_development_line_only": false,
    "RasterFill_primary_material_harm": false,
    "RasterFill_missing_whole_AbsRel_material_harm": false
  },
  "U": {
    "G_P_practical": false,
    "G_Native_practical": false,
    "G_S_practical": false,
    "conditional_development_line_only": false,
    "RasterFill_primary_material_harm": false,
    "RasterFill_missing_whole_AbsRel_material_harm": true
  }
}
```

|查询组/条件|query数|原整query零覆盖|J整query零覆盖|
|---|---:|---:|---:|
|fit/H|1|0|0|
|fit/L|40|0|0|
|fit/U|7|4|4|
|other/H|1|0|0|
|other/L|40|0|0|
|other/U|7|5|4|

U的原四query共7条／4地点：其G/Native主空间降幅−71.39927%，95%[−357.88771,−31.00579]%，是保留的明显退步；U中对RasterFill的missing全图AbsRel材料伤害flag=True。不能由H单地点的正点掩去。其他query原零覆盖5条而J零覆盖4条，二者身份分别保存，不混称。原轮全部44地点的10个零覆盖query及历史反例仍在原报告，训练部分不新扫描。

[全部U、零覆盖及交叉计数](zero_coverage_and_U.json) · [逐query原值、seed、预测身份和支持像素数](joined_scores.csv)。full在原构造下共用Native_full，没有新增模型适配输出，不能作为学习后全图非劣性。

## 最大反例与原全体结论

下列为固定条件内三seed平均后的最大query绝对增误差，最大相对反例及全部逐query和每seed数值在counterexamples.json；H只有一个query，不作代表性选择。

|条件/指标/比较|最大绝对反例|G或前者减后者|该例降误差%|
|---|---|---:|---:|
|H/centered_log_mse/G-P|fidelity_06 / home_bksh / q2|-1.8750926e-05|+2.50212|
|H/centered_log_mse/G-Native|fidelity_06 / home_bksh / q2|-0.00071170096|+49.34306|
|H/absrel/G-RasterFill|fidelity_06 / home_bksh / q2|-0.0014399667|+0.19350|
|L/centered_log_mse/G-P|fidelity_00 / brown_bm_1 / q11|+0.0024517372|-3.47619|
|L/centered_log_mse/G-Native|fidelity_05 / home_at / q8|+0.046916532|-307.34431|
|L/absrel/G-RasterFill|fidelity_06 / home_bksh / q11|+0.026057066|-4.45293|
|U/centered_log_mse/G-P|fidelity_02 / brown_cogsci_1 / q8|+0.0065468571|-4.68160|
|U/centered_log_mse/G-Native|fidelity_02 / brown_cogsci_1 / q11|+0.12397881|-293.79629|
|U/absrel/G-RasterFill|fidelity_02 / brown_cogsci_1 / q11|+0.078656281|-12.37131|
|all/centered_log_mse/G-P|fidelity_02 / brown_cogsci_1 / q8|+0.0065468571|-4.68160|
|all/centered_log_mse/G-Native|fidelity_02 / brown_cogsci_1 / q11|+0.12397881|-293.79629|
|all/absrel/G-RasterFill|fidelity_02 / brown_cogsci_1 / q11|+0.078656281|-12.37131|

未分层旧开发主G/P +0.0856601% [−2.870340,+3.116677]%，G/Native −20.1339893% [−65.563584,+4.882806]%，原未过筛结论不改。九个主比较的新连接全体聚合与旧summary最大差3.331e-14，仅校验连接／聚合接口，没有重算任何像素指标。原训练拟合／来源重构／共同特征与绑定正证据、全图代价及round012关闭均保留；不因单条件点值追认旧全体主量成功。

## 旧成本、实际资源与身份

没有新GPU成本或benchmark。沿原J行完整查询成本（含当时scope及共享硬件边界），不是H特有计时：

|原方法|完整query毫秒|
|---|---:|
|Native|108.8479|
|RasterFill|117.6155|
|P|122.2371|
|S|122.1211|
|G|123.1458|

[原完整query／source-state成本及身份](cost_reference.json)：逐场300次来源拟合、来源构建、KV保存重载和原上游sensor／模型依赖仍存在，不能称本轮免费成场或端到端新性能。未生成新图像、可视化预测或新深度数组；表格直接来自冻结旧评分。

实际科学CPU阶段均已wait exit0；完整aggregate子进程（含import／I/O）11.383383s，标签内部正文0.261963s不含import，未造完整标签精确计时。包含工程、只读探测、等待和最终60s余量的保守全任务worker上界0.184835/1小时，非纯推理时间；峰RSS0.800503/16GiB；新增永久+单原子暂存及最终余量保守峰34.898/256MiB。实时准入记录CPU／RAM／空间，单计算线程，CUDA隐藏且未初始化。0GPU／训练／新预测／渲染／下载／旧像素复算。标签内部计时边界已明确修正，原raw记录保留；无科学运行失败或丢弃统计结果。

固定源字段身份与实际消费的中心／有效槽／相机元数据SHA在label_inputs.json；九份旧末步权重的发布SHA和status小文件身份在fixed_model_identities.json，未重新读大权重；原分数及独立像素核验authority在score_input_identity.json。标签进程0分数读取、score join前后SHA和小型有限例在labels_complete／definition_validation；CPU阶段实际wait记录在terminal_science_completion及workers。所有产物在manifest.json。

O2/O3公共粗GS、O4/O5已完成真实多视图读取沿用，O6只新增事前定义范围检验；仅C5评价分层改变，C1–C6模型／训练／监督不改。sensor／给定相机／静态四源／冻结接收器／depth仍是临时能力范围，不晋升用户核心要求。

全部旧数据已用development，0fresh或独立确认；没有采用方法、总体成功、重新训练、调阈值、挑行、增seed或自派后继。原Astra据完整结果作本范围诊断的科学取舍。原实现角色/root/research_implementation_sol_recovery创建请求gpt-6.1-sol/high，followup无model/effort/Fast字段；Fast未请求或核验，后台effective未知。主独占goal/state/events，持续双向顺序授权保持。实际最终CPU进程正常退出并结束本turn才释放写租约。
