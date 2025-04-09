"""简化版单纯形神经网络(SNN)实现，专注于团结构特征提取。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

class SimplexLayer(nn.Module):
    """单纯形卷积层，用于处理高阶拓扑结构。"""
    
    def __init__(self, in_channels: int, out_channels: int):
        """初始化单纯形卷积层。

        Args:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 定义可学习的权重矩阵
        self.weight_up = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_down = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # 初始化参数
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化层参数。"""
        nn.init.xavier_uniform_(self.weight_up)
        nn.init.xavier_uniform_(self.weight_down)
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor, up_adj: torch.Tensor, down_adj: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入特征，形状为[num_simplices, in_channels]
            up_adj: 上邻接矩阵，形状为[num_simplices, num_simplices]
            down_adj: 下邻接矩阵，形状为[num_simplices, num_simplices]

        Returns:
            输出特征，形状为[num_simplices, out_channels]
        """
        # 上行消息传递
        up_msg = torch.matmul(up_adj, torch.matmul(x, self.weight_up))
        
        # 下行消息传递
        down_msg = torch.matmul(down_adj, torch.matmul(x, self.weight_down))
        
        # 合并消息并添加偏置
        out = up_msg + down_msg + self.bias
        
        return out

class SimplexNeuralNetwork(nn.Module):
    """简化版单纯形神经网络，专注于团结构特征提取。"""
    
    def __init__(self, in_channels: int, hidden_channels: int, output_dim: int, dropout: float = 0.2):
        """初始化单纯形神经网络。

        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            output_dim: 输出特征维度
            dropout: Dropout比率
        """
        super().__init__()
        
        # 特征转换层，将原始特征维度转换为指定维度
        self.feature_transform = nn.Linear(64, in_channels)  # 64是data_loader生成的特征维度
        
        # 创建网络层
        self.input_layer = SimplexLayer(in_channels, hidden_channels)
        self.hidden_layer = SimplexLayer(hidden_channels, hidden_channels)
        self.output_layer = SimplexLayer(hidden_channels, output_dim)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
            
    def forward(self, x: torch.Tensor, up_adj: torch.Tensor, down_adj: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入特征，形状为[num_simplices, input_dim]
            up_adj: 上邻接矩阵，形状为[num_simplices, num_simplices]
            down_adj: 下邻接矩阵，形状为[num_simplices, num_simplices]

        Returns:
            输出特征，形状为[num_simplices, output_dim]
        """
        # 特征转换
        x = self.feature_transform(x)
        
        # 输入层
        x = self.input_layer(x, up_adj, down_adj)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 隐藏层
        x = self.hidden_layer(x, up_adj, down_adj)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 输出层
        x = self.output_layer(x, up_adj, down_adj)
        
        return x
    
    def extract_clique_features(self, features: torch.Tensor, cliques: List[List[int]]) -> torch.Tensor:
        """提取团的特征表示。
        
        Args:
            features: 节点特征，形状为[num_nodes, feature_dim]
            cliques: 团列表，每个团是一个节点索引列表
            
        Returns:
            clique_features: 团特征，形状为[num_cliques, feature_dim]
        """
        # 获取节点数量，用于边界检查
        num_nodes = features.size(0)
        feature_dim = features.size(1)
        
        # 提取每个团的特征
        clique_features = []
        for clique in cliques:
            # 过滤掉无效的节点索引
            valid_indices = [idx for idx in clique if 0 <= idx < num_nodes]
            
            if not valid_indices:
                # 如果没有有效索引，使用零向量
                clique_feature = torch.zeros(feature_dim, device=features.device)
            else:
                # 获取团中节点的特征
                clique_node_features = features[valid_indices]
                
                # 结合多种池化方法以获取更丰富的团表示
                max_pool = torch.max(clique_node_features, dim=0)[0]  # 最大池化
                mean_pool = torch.mean(clique_node_features, dim=0)   # 平均池化
                
                # 组合池化结果
                clique_feature = (max_pool + mean_pool) / 2
            
            clique_features.append(clique_feature)
        
        # 如果团列表为空，返回空张量
        if not clique_features:
            return torch.empty((0, feature_dim), device=features.device)
        
        # 将特征列表堆叠成张量
        return torch.stack(clique_features)