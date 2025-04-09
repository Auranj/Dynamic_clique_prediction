# 只使用EvolveGCN的动态节点分类实现

这个实现是原始DynamicCliqueNet的简化版本，只使用EvolveGCN模型进行动态节点分类，移除了SNN和模型融合部分。这个实现可以用来比较纯EvolveGCN与融合模型在动态节点分类任务上的性能差异。

## 模型特点

- 只使用EvolveGCN进行特征提取和时序建模
- 移除了SNN（Simplex Neural Network）部分
- 移除了模型融合层
- 保留了与原始模型相同的评估指标和训练流程
- 直接将EvolveGCN的输出连接到分类器，进行节点分类

## 目录结构

- `evolvegcn_classifier.py`: 只使用EvolveGCN的分类模型定义
- `config.py`: 配置文件，包含模型、训练和评估参数
- `train.py`: 训练脚本
- `eval.py`: 评估脚本
- `README.md`: 说明文档

## 使用方法

### 训练模型

```bash
cd /root/DynamicCliqueNet/only_evolvegcn
python train.py
```

训练结果将保存在`only_evolvegcn_results/train_results`目录下，包括:
- 训练和验证指标图表
- 训练历史数据
- AUC和AP指标变化趋势图

模型权重将保存在`only_evolvegcn_results/checkpoints`目录中。

### 评估模型

```bash
cd /root/DynamicCliqueNet/only_evolvegcn
python eval.py
```

评估结果将保存在`only_evolvegcn_results/test_results`目录下，包括:
- 测试指标摘要
- 每个时间步的指标
- 混淆矩阵
- 指标随时间变化的趋势图

## 与原始模型比较

这个只使用EvolveGCN的实现可以用来:
1. 评估EvolveGCN本身在动态节点分类任务上的性能
2. 对比SNN-EvolveGCN融合模型与纯EvolveGCN的性能差异
3. 分析模型融合是否带来了实质性的性能提升

通过比较两个模型在相同数据集上的表现，可以确定SNN的贡献和模型融合的必要性。 