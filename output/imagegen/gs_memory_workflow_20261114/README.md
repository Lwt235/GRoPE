# 粗 3DGS 场景记忆：六环节示意图

由内置 image_gen 生图模型生成，按当前跟踪／来源分解实现绘制。用于解释与头脑风暴，非实际场景、预测结果或有效性证明。

## 01 构建公共粗场景

四张来源图像与来源几何辅助建立公共粗高斯场。当前实验并非纯 RGB 成场；椭球布局是概念示意，不是实际重建结果。

![构建公共粗场景](/home/liwt/GRoPE/output/imagegen/gs_memory_workflow_20261114/01_scene_construction.png)

## 02 将图像内容写入高斯

逐图冻结 DINO 提取上下文特征，再按实际渲染贡献绑定到高斯。图中的四项加权均值是来源聚合的简化表示；实际还累加各来源的像素贡献。渲染权重 w 与基元透明度 α 区分。

![将图像内容写入高斯](/home/liwt/GRoPE/output/imagegen/gs_memory_workflow_20261114/02_content_binding.png)

## 03 按当前视角定位记忆

已保存的来源高斯记忆与预测查询相机共同决定投影和可见贡献。当前视图不参与重建或写回该公共记忆。

![按当前视角定位记忆](/home/liwt/GRoPE/output/imagegen/gs_memory_workflow_20261114/03_query_addressing.png)

## 04 在已有注意力中读取

当前图像 Q 在第一个原生 global attention 中读取固定来源上下文及只读 GS K/V。图像自身 K/V 路径为简洁而省略；位置项仅加到 GS K，完整 Σ 普通输入为各 PE 对照共有。

![在已有注意力中读取](/home/liwt/GRoPE/output/imagegen/gs_memory_workflow_20261114/04_attention_reading.png)

## 05 输出对应位置并训练适配

来源锚图像和当前图像均进入原生网络；高斯记忆经图04的接口提供信息，再用官方 TrackHead 预测点对应。当前跟踪版本固定公共高斯场，只训练映射、PE 和早期 LoRA。预测点和虚线范围仅为示意，非实测精度或校准不确定性。

![输出对应位置并训练适配](/home/liwt/GRoPE/output/imagegen/gs_memory_workflow_20261114/05_task_and_learning.png)

## 06 当前实验：保留来源差异

Round28 的正在验证设计：F 保留逐来源分量，M 保留相同均值和来源贡献占比而去掉来源残差。每个高斯均为 4×(255+1)=1024 维；它以通道分辨率换取来源分解，仍需与旧完整均值比较。图不是效果结论。

![当前实验：保留来源差异](/home/liwt/GRoPE/output/imagegen/gs_memory_workflow_20261114/06_source_resolved_experiment.png)

## 生成记录

[完整提示词与来源记录](/home/liwt/GRoPE/output/imagegen/gs_memory_workflow_20261114/prompts.json)

图像已检查关键箭头和符号，并用生图模型作针对性修正；仍属于简化示意，准确的数据流以代码和实验方案为准。

