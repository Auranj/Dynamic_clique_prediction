"""EvolveGCN分类器模型，只使用EvolveGCN处理动态图的节点分类任务。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import sys
import os

# 添加项目根目录到路径
sys.path.append('/root/DynamicCliqueNet')
from models.evolvegcn import EvolveGCN

class EvolveGCNClassifier(nn.Module):
    """只使用EvolveGCN的动态图节点分类模型。"""
    
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 2, dropout: float = 0.2):
        """初始化EvolveGCN分类器模型。

        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            num_layers: GCN层数量
            dropout: Dropout概率
        """
        super().__init__()
        
        # EvolveGCN模块，用于处理时间演化
        self.evolvegcn = EvolveGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers
        )
        
        # 分类器层，用于节点分类
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)  # 二分类任务，输出维度为2
        )
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入特征，形状为[num_nodes, in_channels]
            adj: 邻接矩阵，形状为[num_nodes, num_nodes]

        Returns:
            output: 分类预测结果，形状为[num_nodes, 2]（二分类）
        """
        # 使用EvolveGCN提取特征
        evolvegcn_out = self.evolvegcn(x, adj)
        
        # 节点分类
        output = self.classifier(evolvegcn_out)
        
        return output 