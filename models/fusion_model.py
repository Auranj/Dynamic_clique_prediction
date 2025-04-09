"""融合模型实现，结合SNN和EvolveGCN的优势。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from .snn import SimplexNeuralNetwork
from .evolvegcn import EvolveGCN

class SNNEvolveGCN(nn.Module):
    """融合SNN和EvolveGCN的动态团预测模型。"""
    
    def __init__(self, in_channels: int, hidden_channels: int, output_dim: int, dropout: float = 0.2):
        """初始化融合模型。

        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            output_dim: 输出维度
            dropout: Dropout概率
        """
        super().__init__()
        
        # SNN模块，用于提取团结构特征
        self.snn = SimplexNeuralNetwork(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # EvolveGCN模块，用于处理时间演化
        self.evolvegcn = EvolveGCN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=2
        )
        
        # 融合层 - 修复维度不匹配问题
        # SNN输出维度为output_dim(32)，EvolveGCN输出维度为hidden_channels(128)
        # 因此连接后的特征维度为32+128=160
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 预测层
        self.predictor = nn.Linear(hidden_channels, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入特征，形状为[num_nodes, 2 * hidden_channels]，包含SNN和EvolveGCN的输出特征

        Returns:
            output: 预测结果，形状为[num_nodes, output_dim]
        """
        # 融合特征
        fused_features = self.fusion_layer(x)
        
        # 预测
        output = self.predictor(fused_features)
        
        return output
    
    def predict_evolution(self, curr_features: torch.Tensor, curr_cliques: List[List[int]], 
                         prev_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """预测团的演化。

        Args:
            curr_features: 当前时间步的特征
            curr_cliques: 当前时间步的团列表
            prev_features: 前一时间步的特征（可选）

        Returns:
            演化概率矩阵，形状为[num_curr_cliques, num_possible_states]
        """
        # 获取当前团的特征表示
        curr_clique_features = self.snn.extract_clique_features(curr_features, curr_cliques)
        
        if prev_features is not None:
            # 确保prev_features和curr_features的节点数量一致
            if prev_features.size(0) != curr_features.size(0):
                # 如果不一致，调整prev_features的大小
                num_nodes = curr_features.size(0)
                if prev_features.size(0) > num_nodes:
                    prev_features = prev_features[:num_nodes]
                else:
                    # 如果prev_features节点数量不足，使用零填充
                    padding = torch.zeros(num_nodes - prev_features.size(0), prev_features.size(1), device=prev_features.device)
                    prev_features = torch.cat([prev_features, padding], dim=0)
            
            # 如果有前一时间步的特征，使用EvolveGCN更新
            evolved_features = self.evolvegcn(curr_features, prev_features)
            
            # 获取节点数量，用于边界检查
            num_nodes = evolved_features.size(0)
            
            # 安全地获取团特征
            evolved_clique_features = []
            for clique in curr_cliques:
                # 过滤掉无效的节点索引
                valid_indices = [idx for idx in clique if 0 <= idx < num_nodes]
                if not valid_indices:
                    # 如果没有有效索引，使用零向量
                    clique_feature = torch.zeros(evolved_features.size(1), device=evolved_features.device)
                else:
                    # 获取团中节点的特征
                    clique_node_features = evolved_features[valid_indices]
                    # 使用最大池化聚合团特征
                    clique_feature = torch.max(clique_node_features, dim=0)[0]
                evolved_clique_features.append(clique_feature)
            
            evolved_clique_features = torch.stack(evolved_clique_features)
            
            # 融合特征
            # 确保特征维度匹配
            if curr_clique_features.size(1) + evolved_clique_features.size(1) != self.fusion_layer[0].in_features:
                # 如果维度不匹配，调整特征维度
                expected_dim = self.fusion_layer[0].in_features
                curr_dim = curr_clique_features.size(1)
                evolved_dim = evolved_clique_features.size(1)
                
                # 创建正确维度的特征
                combined_features = torch.zeros(curr_clique_features.size(0), expected_dim, device=curr_clique_features.device)
                # 复制有效部分
                valid_curr_dim = min(curr_dim, expected_dim // 2)
                valid_evolved_dim = min(evolved_dim, expected_dim - valid_curr_dim)
                
                combined_features[:, :valid_curr_dim] = curr_clique_features[:, :valid_curr_dim]
                combined_features[:, valid_curr_dim:valid_curr_dim+valid_evolved_dim] = evolved_clique_features[:, :valid_evolved_dim]
                
                fused_features = self.fusion_layer(combined_features)
            else:
                # 正常情况下直接连接
                fused_features = torch.cat([curr_clique_features, evolved_clique_features], dim=1)
                fused_features = self.fusion_layer(fused_features)
        else:
            # 如果是第一个时间步，只使用当前特征
            # 检查维度是否匹配
            if curr_clique_features.size(1) * 2 != self.fusion_layer[0].in_features:
                # 如果维度不匹配，调整特征维度
                expected_dim = self.fusion_layer[0].in_features
                curr_dim = curr_clique_features.size(1)
                
                # 创建正确维度的特征
                combined_features = torch.zeros(curr_clique_features.size(0), expected_dim, device=curr_clique_features.device)
                # 复制有效部分
                valid_dim = min(curr_dim, expected_dim // 2)
                
                combined_features[:, :valid_dim] = curr_clique_features[:, :valid_dim]
                combined_features[:, valid_dim:valid_dim*2] = curr_clique_features[:, :valid_dim]
                
                fused_features = self.fusion_layer(combined_features)
            else:
                # 正常情况下直接连接
                fused_features = self.fusion_layer(torch.cat([curr_clique_features, curr_clique_features], dim=1))
        
        # 预测演化概率
        evolution_prob = torch.sigmoid(self.predictor(fused_features))
        
        return evolution_prob