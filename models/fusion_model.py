"""融合模型实现，结合SNN和EvolveGCN的优势，专注于动态团预测。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from .snn import SimplexNeuralNetwork
from .evolvegcn import EvolveGCN

class SNNEvolveGCN(nn.Module):
    """融合SNN和EvolveGCN的动态团预测模型，专注于团演化预测任务。"""
    
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
        
        # 团演化预测模块
        self.evolution_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels, 
            num_heads=4, 
            dropout=dropout
        )
        
        # 团演化预测器
        self.evolution_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
        
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
    
    def predict_evolution(self, curr_features: torch.Tensor, curr_cliques: List[List[int]], 
                         prev_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """预测团的演化，专注于团演化预测任务。

        Args:
            curr_features: 当前时间步的特征
            curr_cliques: 当前时间步的团列表
            prev_features: 前一时间步的特征（可选）

        Returns:
            演化概率矩阵，形状为[num_curr_cliques, num_possible_states]
        """
        # 获取当前团的特征表示
        curr_clique_features = self.extract_clique_features(curr_features, curr_cliques)
        
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
            
            # 使用EvolveGCN从前一时间步更新当前时间步
            evolved_features = self.evolvegcn(curr_features, prev_features)
            
            # 提取前一时间步的团特征
            # 简化处理：假设前一时间步有一个团，这里我们使用全局特征来表示
            prev_clique_feature = torch.mean(prev_features, dim=0, keepdim=True)
            
            # 使用注意力机制计算当前团与前一时间步团之间的关联
            # 调整特征维度以适应多头注意力输入要求：[seq_len, batch, embed_dim]
            q = curr_clique_features.unsqueeze(1).permute(1, 0, 2)  # 查询：当前团
            k = prev_clique_feature.unsqueeze(1).permute(1, 0, 2)  # 键：前一时间步的团
            v = prev_clique_feature.unsqueeze(1).permute(1, 0, 2)  # 值：前一时间步的团
            
            # 应用注意力机制
            attn_output, _ = self.evolution_attention(q, k, v)
            
            # 调整维度 [batch, seq_len, embed_dim] -> [seq_len, embed_dim]
            attn_output = attn_output.permute(1, 0, 2).squeeze(1)
            
            # 连接当前团特征和带注意力的前一时间步团特征
            evolution_features = torch.cat([curr_clique_features, attn_output], dim=1)
            
            # 预测演化概率
            evolution_logits = self.evolution_predictor(evolution_features)
            evolution_prob = torch.sigmoid(evolution_logits)
            
            # 确保输出形状为 [num_curr_cliques, 1]
            if len(evolution_prob.shape) == 1:
                evolution_prob = evolution_prob.unsqueeze(1)
            
            return evolution_prob
        else:
            # 如果是第一个时间步，无法预测演化（没有前一时间步的信息）
            # 返回全零概率
            return torch.zeros((len(curr_cliques), 1), device=curr_features.device)