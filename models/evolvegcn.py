"""EvolveGCN核心组件实现，用于处理动态图的时间演化。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class EvolveGCNLayer(nn.Module):
    """EvolveGCN层，使用GRU更新GCN权重。"""
    
    def __init__(self, in_channels: int, out_channels: int):
        """初始化EvolveGCN层。

        Args:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # GCN权重
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # GRU单元，用于更新权重
        self.weight_gru = nn.GRUCell(in_channels, in_channels)
        
        # 初始化参数
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化层参数。"""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入特征，形状为[num_nodes, in_channels]
            adj: 邻接矩阵，形状为[num_nodes, num_nodes]

        Returns:
            输出特征，形状为[num_nodes, out_channels]
        """
        # 标准GCN传播规则
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support) + self.bias
        return output
    
    def evolve_weight(self, hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """使用GRU更新权重参数。

        Args:
            hidden_state: GRU的隐藏状态，形状为[in_channels]

        Returns:
            更新后的隐藏状态
        """
        # 将权重矩阵转换为GRU输入格式
        weight_input = self.weight.t()  # [out_channels, in_channels]
        
        if hidden_state is None:
            # 如果没有隐藏状态，使用零初始化
            hidden_state = torch.zeros(self.in_channels, device=self.weight.device)
        
        # 使用单个样本进行GRU更新，避免批次大小不匹配
        # 取第一行作为输入样本
        input_sample = weight_input[0].unsqueeze(0)  # [1, in_channels]
        hidden_state = self.weight_gru(input_sample, hidden_state.unsqueeze(0))[0]
        
        # 更新权重矩阵 - 修复维度不匹配问题
        # 由于hidden_state的形状是[in_channels]，我们需要将其调整为与weight兼容的形状
        # weight的形状是[in_channels, out_channels]
        new_weight = torch.zeros_like(self.weight)
        for i in range(self.out_channels):
            new_weight[:, i] = hidden_state
        self.weight.data.copy_(new_weight)
        
        return hidden_state

class EvolveGCN(nn.Module):
    """EvolveGCN模型，用于动态图表示学习。"""
    
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 2):
        """初始化EvolveGCN模型。

        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            num_layers: GCN层数量
        """
        super().__init__()
        self.num_layers = num_layers
        
        # 特征转换层，将原始低维特征转换为指定维度
        self.feature_transform = nn.Linear(64, in_channels)  # 64是data_loader生成的特征维度
        
        # 创建EvolveGCN层
        self.layers = nn.ModuleList()
        self.layers.append(EvolveGCNLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(EvolveGCNLayer(hidden_channels, hidden_channels))
            
        # 批归一化层
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
        # 隐藏状态
        self.hidden_states = [None] * num_layers
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入特征，形状为[num_nodes, in_channels]
            adj: 邻接矩阵，形状为[num_nodes, num_nodes]

        Returns:
            输出特征，形状为[num_nodes, hidden_channels]
        """
        # 特征转换
        x = self.feature_transform(x)
        
        # 确保邻接矩阵和特征矩阵的维度兼容
        num_nodes = x.size(0)
        
        # 检查adj是否是二维矩阵
        if len(adj.shape) != 2:
            # 如果不是二维矩阵，创建一个空的邻接矩阵
            adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        elif adj.size(0) != num_nodes or adj.size(1) != adj.size(0):
            # 如果邻接矩阵维度不匹配或不是方阵，创建一个与节点数匹配的新邻接矩阵
            new_adj = torch.zeros(num_nodes, num_nodes, device=x.device)
            # 只复制有效部分
            min_rows = min(adj.size(0), num_nodes)
            min_cols = min(adj.size(1), num_nodes)
            new_adj[:min_rows, :min_cols] = adj[:min_rows, :min_cols]
            adj = new_adj
        
        # 更新每一层
        for i in range(self.num_layers):
            # 更新权重
            self.hidden_states[i] = self.layers[i].evolve_weight(self.hidden_states[i])
            
            # GCN传播
            x = self.layers[i](x, adj)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            
        return x
    
    def reset_hidden_states(self):
        """重置所有层的隐藏状态。"""
        self.hidden_states = [None] * self.num_layers