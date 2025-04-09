# 数据集配置
DATASET_CONFIG = {
    'data_path': '/root/DynamicCliqueNet/data/soc-sign-bitcoinotc.csv',
    'time_steps': 138,  # 时间步数（增加到原来的2倍）
    'train_ratio': 0.7,  # 训练集比例
    'val_ratio': 0.1,   # 验证集比例
    'test_ratio': 0.2,   # 测试集比例
    'min_window': 7,    # 最小时间窗口（天）
    'max_window': 30,   # 最大时间窗口（天）
    'window_step': 1,   # 时间窗口搜索步长（天）
    'adaptive_window': True  # 是否使用自适应时间窗口
}

# 模型配置
MODEL_CONFIG = {
    # SNN配置
    'snn': {
        'in_channels': 64,
        'hidden_channels': 128,
        'output_dim': 64,  # 输出维度，用于提取团结构特征
        'dropout': 0.3     # dropout概率以减少过拟合
    },
    
    # EvolveGCN配置
    'evolvegcn': {
        'in_channels': 64,
        'hidden_channels': 128,
        'num_layers': 2
    },
    
    # 融合模型配置
    'fusion': {
        'in_channels': 64,
        'hidden_channels': 128,
        'output_dim': 64,  # 输出维度应与SNN匹配
        'dropout': 0.3     # dropout概率以减少过拟合
    },
    
    # 团演化预测配置
    'evolution': {
        'attention_dim': 64,   # 注意力机制的嵌入维度
        'num_heads': 4,        # 多头注意力的头数
        'dropout': 0.3,        # 注意力机制的dropout概率
        'threshold': 0.5       # 团演化预测的概率阈值
    }
}

# 训练配置
TRAIN_CONFIG = {
    'num_epochs': 150,                # 训练轮数
    'batch_size': 32,
    'learning_rate': 0.0005,          # 学习率
    'weight_decay': 5e-5,             # 权重衰减以减少过拟合
    'early_stopping_patience': 15,    # 早停耐心值
    'save_dir': 'checkpoints',
    'loss_weights': {
        'node_loss_weight': 0.3,      # 节点级别损失权重
        'evolution_loss_weight': 0.7  # 团演化预测损失权重
    }
}

# 评估配置
EVAL_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc', 'ap', 
               'evolution_accuracy', 'evolution_precision', 'evolution_recall', 'evolution_f1'],  # 团演化预测指标
    'visualization': {
        'plot_confusion_matrix': True,
        'plot_metrics_over_time': True,
        'plot_evolution_metrics': True,  # 团演化预测指标可视化
        'plot_attention_weights': True   # 注意力权重可视化
    }
}