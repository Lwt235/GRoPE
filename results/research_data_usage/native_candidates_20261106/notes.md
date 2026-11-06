# 后续原生高斯确认数据：元数据候选核对

2026-11-06。当前九次 matched256 训练继续按原方案执行。这里仅核对后续数据入口，不是新实验协议，也不是新的效果证据。

当前 GSModel60 全部官方 TRAIN/TEST 均已使用，不能在设计迭代后把原 TEST 再称为全新确认。作者数据卡还明确指出，GSModel60 的 30–59 类来自 ModelNet_Splats，因此换用该仓库不自动构成新对象确认；仍可通过对象身份求出真正未用的部分，不能反过来把整个来源一律判为已用。[GSModel60 数据说明](https://huggingface.co/datasets/zxAi/GSModel60)

## 已核实的入口与限制

| 候选 | 可核实事实 | 对当前工作的意义与尚缺信息 |
| --- | --- | --- |
| [ShapeSplats/omniobject_3dgs](https://huggingface.co/datasets/ShapeSplats/omniobject_3dgs/tree/612f9a55be5f80bb424f29ea1343b66e1d8adf55) | 公开元数据列出 218 文件，112,022,872,945 字节；访问类型 auto-gated；类别命名 ZIP 在目录中可见 | 值得优先核对真实扫描来源的原生基元用途。当前未核实 ZIP 内对象数、标签对应、拟合配置、划分及历史对象交集，不能宣称已经取得可用独立测试集 |
| [ShapeNet/ShapeSplatsV1](https://huggingface.co/datasets/ShapeNet/ShapeSplatsV1/tree/1dabf50223bf360307afa4e78539fbd1aa73d65e) | 公开元数据列出 61 文件，305,700,063,582 字节；manual-gated。作者代码提供 ShapeNet55GS 配置和训练/测试清单入口 | 可以作为另一个对象分布或部件任务的候选；55 类体系不能直接套当前 60 类分类头。需要事前确定新的匹配训练/评价流程，不能把换头训练称作现有模型直接泛化 |
| [ShapeSplats/Objaverse_Splats](https://huggingface.co/datasets/ShapeSplats/Objaverse_Splats/tree/0f606a35456a4d2ae316e6fd473d0435ea8b1865) | 157 文件，381,865,646,044 字节，auto-gated；数据卡介绍 141,703 个对象及 caption 元数据 | 原生基元规模足够，但当前未核实与 60 类的可靠标签映射。caption 不自动成为分类真值，因此先保留候选，不因数据大就立即启动分类实验 |
| [ShapeSplats/ModelNet_Splats](https://huggingface.co/datasets/ShapeSplats/ModelNet_Splats) | 作者 README 报告 12,309 个 ModelNet 对象、40 类；它也是当前 GSModel60 的已知上游来源 | 主要价值是与历史对象清单做精确求差，或检查拟合版本变化；相同网格的新拟合不能算全新物体 |
| [GaussianVerse](https://gaussianverse.stanford.edu/) | 项目页介绍 254,371 个拟合，每个 16,384 个高斯，并提供几何与外观属性 | 更偏生成数据入口；当前未核实本分类任务的标签与划分，不优先引入额外标注或任务转换 |

OmniObject3D 原项目介绍的是 6,000 个扫描对象、190 类；这些数字不能直接套在当前高斯仓库上。本次没有打开 ZIP，仓库内对象与原项目的对应仍未知。[OmniObject3D 原项目](https://omniobject3d.github.io/)

ShapeSplat 作者的新版 README 把 ShapeNet、ModelNet、Objaverse 分别列为 52,121 / 12,309 / 141,703 个对象；早期项目页的 65K 数字与新版包含 Objaverse 的约 206K 不是同一覆盖范围。作者 DATA.md 提供 `datasets/shapenet_split` 以及部件标注接法；这些入口说明后续可以留在 Transformer 的理解任务上，无须为了新数据重新转向渲染任务。[作者 README](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE/blob/main/README.md) · [数据准备说明](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE/blob/main/DATA.md)

另一个来源边界：MACGS 论文的原基准使用 MVImageNet 来源的 20 类、4,000 个对象；当前 GSModel60 数据卡标明的是 30 个 MACGS-derived 类。两者不能直接视作同一发布清单，未来排除历史对象必须依本仓库实际 ID 清单，不用论文总数代替。[MACGS 论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Mitigating_Ambiguities_in_3D_Classification_with_Gaussian_Splatting_CVPR_2025_paper.pdf)

## 本次访问范围与下一步

Hugging Face 连接工具返回 UNAVAILABLE，随后使用作者 GitHub 文件工具、公开网页及 hf-mirror.com 的匿名公开 API。三个 API 响应总计约 98 KiB，仅包含仓库版本、文件名、大小与发布元数据；直连、无代理、未发送凭据。没有申请权限，没有下载 ZIP、PLY、图像或读取归档内部内容。`gated` 是仓库属性；没有查询当前用户是否已有访问权限，也未据此把研究判为阻塞。[原始元数据与访问记录](metadata_access.json)

候选层面的判断是：优先核对 OmniObject 原生基元与 ShapeNet 两个入口的任务适配性；ModelNet 先处理已知来源重叠；Objaverse 和 GaussianVerse 保留为后备。此优先级依据任务、来源和元数据可用性，未查看候选上的模型得分。当前不启动批量下载或新训练。

等 matched256 固定比较完成，再结合其结果选择接下来的科学问题。若采用新来源，先按对象身份固定开发/确认边界、强控制、指标和预算，随后才开放相应数据内容。分类或区域匹配是验证 PE 的主任务；渲染不是进入新数据验证的前提。新来源的正结果也不能删除已有 RGB、直接迁移或官方基础确认的不确定性。

后续更新：matched256 已完成并审阅，对称关系的 15 次开发拟合已启动。等待期间完成了公开对象清单核对：ShapeNet 的 520 个 test 对象全部位于其 51,934 对象 train 清单中，而且只覆盖 43/55 类，需要另定独立分类划分。ModelNet 排除已知已用 ID 后有 747 对象，其中原 test 的 620 对象 / 七类可与当前 MACGS 来源输出同名对应，值得准备保留现有头的跨来源评价。上一版误将 ModelNet 来源的 30 类当作全 60 类而漏掉这些对应，已保留原记录并修正；没有读取新高斯或模型分数。[具体计算、更正与来源身份](split_check1/notes.md)

同样按当前完整 60 类词表检查 OmniObject 高斯仓库的 216 个类别名 ZIP，有 18 个严格同名条目，合计约 8.459 GiB。它们只是目录名称对应，尚未核实内部对象数、语义边界、拟合属性与官方原始发布的对象关系；不从原项目 190 类/6,000 对象推定这个发布版的覆盖。可作为后续真实扫描来源的调查入口，当前不据此下载或宣称完成跨域确认。[同名条目与来源身份](omni_literal_name_mapping.json) · [原始项目](https://omniobject3d.github.io/) · [原始作者仓库](https://github.com/omniobject3d/OmniObject3D)
