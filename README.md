# DynamicCliqueNet

## 项目简介
这是一个结合SNN（单纯形神经网络）和EvolveGCN的动态团预测系统。该系统旨在解决动态图中的团结构预测问题，通过融合SNN对高阶拓扑结构的捕捉能力与EvolveGCN对时间演化建模的优势，实现对动态图中团结构变化的准确预测。

## 主要特点
- 融合SNN和EvolveGCN的优势
- 专注于动态图中的团结构预测
- 使用Bitcoin-OTC数据集进行验证
- 完整的训练和评估流程

## 项目结构
```
DynamicCliqueNet/
├── data/
│   ├── __init__.py
│   ├── data_loader.py     # 数据加载与预处理
│   └── clique_utils.py    # 团结构检测与标注
├── models/
│   ├── __init__.py
│   ├── snn.py             # 简化版SNN实现
│   ├── evolvegcn.py       # EvolveGCN核心组件
│   └── fusion_model.py    # 融合模型架构
├── utils/
│   ├── __init__.py
│   ├── evaluation.py      # 评估指标
│   └── visualization.py   # 可视化工具
├── config.py              # 配置参数
├── train.py               # 训练脚本
├── eval.py                # 评估脚本
└── README.md             # 项目说明
```

## 数据集
使用Bitcoin-OTC作为主要测试数据集，数据集特点：
- 5881个节点，35592条边
- 89%的正例比例
- 时间跨度约5年，可分为138个时间步

## 环境要求
- Python 3.7+
- PyTorch
- NetworkX
- NumPy
- Scikit-learn

## 使用说明
1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 数据预处理
```bash
python -m data.data_loader
```

3. 模型训练
```bash
python train.py
```

4. 模型评估
```bash
python eval.py
```

## 主要模块

### 数据处理模块
- 加载和处理Bitcoin-OTC数据集
- 构建动态图的时间序列表示
- 团结构检测与特征提取

### 模型模块
- SNN模块：处理高阶拓扑结构
- EvolveGCN模块：处理时间演化
- 融合模型：结合两个模块的优势

### 评估模块
- 团结构预测评估指标
- 可视化工具

## 许可证
MIT License# Dynamic_clique_prediction
