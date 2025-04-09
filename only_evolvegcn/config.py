# 数据集配置
DATASET_CONFIG = {
    'data_path': '/root/DynamicCliqueNet/data/soc-sign-bitcoinotc.csv',
    'time_steps': 138,  # 时间步数
    'train_ratio': 0.7,  # 训练集比例
    'val_ratio': 0.1,   # 验证集比例
    'test_ratio': 0.2,   # 测试集比例
    'min_window': 7,    # 最小时间窗口（天）
    'max_window': 30,   # 最大时间窗口（天）
    'window_step': 1,   # 时间窗口搜索步长（天）
    'adaptive_window': True,  # 是否使用自适应时间窗口
    'overlap_ratio': 0.5  # 时间窗口重叠比例，0.5表示50%重叠
}

# 模型配置
MODEL_CONFIG = {
    # EvolveGCN配置
    'evolvegcn': {
        'in_channels': 64,
        'hidden_channels': 128,
        'num_layers': 2,
        'dropout': 0.3     # dropout概率以减少过拟合
    }
}

# 结果目录配置 - 使用绝对路径
RESULTS_CONFIG = {
    'base_dir': '/root/DynamicCliqueNet/only_evolvegcn_results',
    'train_dir': '/root/DynamicCliqueNet/only_evolvegcn_results/train_results',
    'test_dir': '/root/DynamicCliqueNet/only_evolvegcn_results/test_results',
    'checkpoints_dir': '/root/DynamicCliqueNet/only_evolvegcn_results/checkpoints'
}

# 训练配置
TRAIN_CONFIG = {
    'num_epochs': 150,                # 训练轮数
    'batch_size': 32,
    'learning_rate': 0.0005,          # 学习率
    'weight_decay': 5e-5,             # 权重衰减以减少过拟合
    'early_stopping_patience': 15,    # 早停耐心值
    'save_dir': RESULTS_CONFIG['checkpoints_dir'],  # 更新为新的检查点保存路径
    'label_smoothing': True,          # 是否使用标签平滑
    'l2_lambda': 0.001,               # L2正则化强度
    'max_grad_norm': 1.0              # 梯度裁剪阈值
}

# 评估配置
EVAL_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc', 'ap'],  # 节点分类指标
    'visualization': {
        'plot_confusion_matrix': True,
        'plot_metrics_over_time': True
    }
} 