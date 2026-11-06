# 后续原生确认：对象清单与任务接法核对

2026-11-06。此次只读取作者公开对象 ID、配置和 loader，以及本仓库已有的对象使用清单；没有下载或解码新的高斯/图像，没有读取新模型分数，也未改变正在训练的对称关系实验。

**ShapeNet 需要独立划分；ModelNet 的余集中有七类可以按同名类别对应现有 60 类输出，值得准备直接跨来源确认。** 这是数据准备结论，不是位置编码效果结论。

更正：上一版把“ModelNet 来源内的 30 类”误当成“整个 60 类分类头”，漏掉了 MACGS 来源的同名类别，因此错误地否定了所有余集与现有输出的对应。现已按完整 60 类重新计算；原分析和解释保存在 `analyze_v1.py`、`result_v1.json`、`notes_v1.md`，当前 `result.json` 为修正后的第二版。ShapeNet 交集与 ModelNet 已用对象排除计数不受此更正影响。

ShapeSplat 作者仓库的 ShapeNet 清单有 51,934 个训练路径、55 类；测试清单有 520 个路径、43 类，各类 1–88 个对象。按完整文件名和模型 ID 分别求交，**520 个测试对象全部出现在训练清单中**；另有 12 个训练类别未出现在该测试清单。独立 `sort`/`comm` 计数也为 520。作者相应配置是 Gaussian MAE 预训练，val/test 指向同一份 `test.txt`。所以这两份清单提供了对象目录，但不直接提供本研究需要的不重叠 55 类分类确认划分。这一判断限于核对的文件和用途，不替代对作者其他实验的评价。[训练清单](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE/blob/main/datasets/shapenet_split/train.txt) · [测试清单](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE/blob/main/datasets/shapenet_split/test.txt) · [预训练配置](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE/blob/main/cfgs/pretrain/pretrain_enc_full_group_xyz_1k.yaml)

ModelNet 的当前发布清单有 9,842 个 train 和 2,467 个 test 对象。合并早期 R48/R58 已用点云对象和当前 GSModel60 全部 TRAIN/TEST 的原生对象身份后，其中 11,562 个有已知使用记录，剩余 **747 个无已知已用 ID**：原 test 中 746 个、原 train 中 1 个。它们属于另外十个 ModelNet 来源类别，其中 airplane、bottle、bowl、car、chair、guitar、table 七类与现有 **MACGS 来源输出**同名，合计 621 对象。限于原 test，得到 **620 对象 / 七类**：bowl 20 个，另六类各 100 个，对应输出 ID 为 airplane=20、bottle=1、bowl=15、car=13、chair=12、guitar=11、table=4。这为保留现有分类头的直接跨来源评价提供候选，不需要先重训类别头。其余 cup=20、flower_pot=20、night_stand=86 没有严格同名输出，当前没有推定别名映射；单个原 train 余项另列，不混入测试候选。[作者 train 清单](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE/blob/main/datasets/modelnet_split/modelnet40_train.txt) · [作者 test 清单](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE/blob/main/datasets/modelnet_split/modelnet40_test.txt) · [本仓库历史对象核对](../../../native_confirmation_metadata/attempt1/historical_overlap1/result.json)

ShapeNet loader 另有一个接入时必须保留的语义区别：它先从 PLY 的 log scale 取指数、归一化 quaternion，再按坐标半径缩放；预训练配置还会对 scale 特征减均值并另作归一化。后者已是网络属性，不应直接当作物理高斯轴长。若后续建立本研究缓存，应从原始 PLY 的 scale/rotation 构造正定协方差，并随中心使用一致的空间缩放，沿用匹配控制的共同预处理。这是对源代码的接法判断，未改动本轮模型。[作者 loader](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE/blob/main/datasets/ShapeNet55Gaussian.py)

当前决定是先等待已固定的 15 次开发拟合和完整比较。若新候选值得推进，可考虑将 ModelNet 原 test 的同名七类固定为跨来源确认候选，保持完整 60 类输出，预测到其他类仍计为错误，不能在看分数后决定是否限制输出类别。是否采用、全部模型/种子/条件及统计规则须在读取新高斯和分数前确定；这不会覆盖全 60 类，也不能替代其他任务。ShapeNet 仍需按对象身份另行固定开发/选模/确认边界。OmniObject 的可靠对象对应及高斯拟合配置仍待核实。不同数据源的 ID 不同也不证明底层几何绝无重复；这里的 ModelNet 余集仅是已知 ID 层面的排除结果，不预先声称已获得独立效果确认。

后续已把这 620 个对象与现有输出的对应保存为候选元数据，并核对 ModelNet_Splats 发布版本 `6538575534e3a48a31d91beaef781e6c1db2b456`、七个类别 ZIP 的大小及 LFS 身份。Hugging Face 连接工具当前可返回数据说明，但未返回版本/文件清单，故通过 hf-mirror.com 匿名直连公开 API 补齐元数据。仓库属性为 auto-gated，尚未检查用户现有权限或提交申请；归档内部对象路径、可用性和原始属性尚未读取。此候选文件不等于已冻结科学协议。[620 对象候选元数据](../modelnet620_candidate_metadata.json) · [公开元数据获取记录](../modelnet_metadata_access.json)

来源均绑定 Git blob：ShapeNet train/test 为 `dea6edcdbf3c94826814ab6b26f0e7a214e6db4b` / `a9d10ce294b05145f296bf1987b711c48030e011`，ModelNet train/test 为 `1b0c58cf52163734345e5411e39baa7d50e33773` / `ddf959319649739eeba3b2ebb31dffc98a7e72c6`。七份文件均按 Git blob 哈希验证，ModelNet 两份本地副本仅补了末尾换行且单独记录。GitHub 工具无法读取较大的 ShapeNet train 文本，随后匿名直连公开 Git blob API，获取约 3.13 MiB 响应、解码为 2.27 MiB 的 ID 文本；未申请受限数据权限。[获取记录](train_metadata_fetch.json) · [完整计数、交集与来源身份](result.json) · [ModelNet 逐对象记录](modelnet_membership.json) · [分析程序](analyze.py)
