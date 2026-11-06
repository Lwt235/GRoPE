# 后续原生确认：对象清单与任务接法核对

2026-11-06。此次只读取作者公开对象 ID、配置和 loader，以及本仓库已有的对象使用清单；没有下载或解码新的高斯/图像，没有读取新模型分数，也未改变正在训练的对称关系实验。

**两个公开来源都需要另行确定任务和独立划分，不能直接作为当前 60 类模型的新确认集。** 这是数据准备结论，不是位置编码效果结论。

ShapeSplat 作者仓库的 ShapeNet 清单有 51,934 个训练路径、55 类；测试清单有 520 个路径、43 类，各类 1–88 个对象。按完整文件名和模型 ID 分别求交，**520 个测试对象全部出现在训练清单中**；另有 12 个训练类别未出现在该测试清单。独立 `sort`/`comm` 计数也为 520。作者相应配置是 Gaussian MAE 预训练，val/test 指向同一份 `test.txt`。所以这两份清单提供了对象目录，但不直接提供本研究需要的不重叠 55 类分类确认划分。这一判断限于核对的文件和用途，不替代对作者其他实验的评价。[训练清单](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE/blob/main/datasets/shapenet_split/train.txt) · [测试清单](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE/blob/main/datasets/shapenet_split/test.txt) · [预训练配置](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE/blob/main/cfgs/pretrain/pretrain_enc_full_group_xyz_1k.yaml)

ModelNet 的当前发布清单有 9,842 个 train 和 2,467 个 test 对象。合并早期 R48/R58 已用点云对象和当前 GSModel60 全部 TRAIN/TEST 的原生对象身份后，其中 11,562 个有已知使用记录，剩余 **747 个无已知已用 ID**：原 test 中 746 个、原 train 中 1 个。它们属于 airplane、bottle、bowl、car、chair、cup、flower_pot、guitar、night_stand、table 十类，**没有一个属于当前 GSModel60 的 30 个 ModelNet 类别**。因此要利用这些对象，应事前建立匹配的新分类任务和分类头；不能直接用现有 60 类输出进行有效分类确认。单个 train 余项不是独立大样本测试。[作者 train 清单](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE/blob/main/datasets/modelnet_split/modelnet40_train.txt) · [作者 test 清单](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE/blob/main/datasets/modelnet_split/modelnet40_test.txt) · [本仓库历史对象核对](../../../native_confirmation_metadata/attempt1/historical_overlap1/result.json)

ShapeNet loader 另有一个接入时必须保留的语义区别：它先从 PLY 的 log scale 取指数、归一化 quaternion，再按坐标半径缩放；预训练配置还会对 scale 特征减均值并另作归一化。后者已是网络属性，不应直接当作物理高斯轴长。若后续建立本研究缓存，应从原始 PLY 的 scale/rotation 构造正定协方差，并随中心使用一致的空间缩放，沿用匹配控制的共同预处理。这是对源代码的接法判断，未改动本轮模型。[作者 loader](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE/blob/main/datasets/ShapeNet55Gaussian.py)

当前决定是先等待已固定的 15 次开发拟合和完整比较。若新候选值得推进，ShapeNet 需按对象身份另行固定开发/选模/确认边界；ModelNet 剩余对象只能支持明确的新类别任务。OmniObject 的可靠标签、划分及归档内部对象对应仍待核实。不同数据源的 ID 不同也不证明底层几何绝无重复；这里的 ModelNet 余集仅是已知 ID 层面的排除结果，不预先声称获得了全新独立确认。

来源均绑定 Git blob：ShapeNet train/test 为 `dea6edcdbf3c94826814ab6b26f0e7a214e6db4b` / `a9d10ce294b05145f296bf1987b711c48030e011`，ModelNet train/test 为 `1b0c58cf52163734345e5411e39baa7d50e33773` / `ddf959319649739eeba3b2ebb31dffc98a7e72c6`。七份文件均按 Git blob 哈希验证，ModelNet 两份本地副本仅补了末尾换行且单独记录。GitHub 工具无法读取较大的 ShapeNet train 文本，随后匿名直连公开 Git blob API，获取约 3.13 MiB 响应、解码为 2.27 MiB 的 ID 文本；未申请受限数据权限。[获取记录](train_metadata_fetch.json) · [完整计数、交集与来源身份](result.json) · [ModelNet 逐对象记录](modelnet_membership.json) · [分析程序](analyze.py)
