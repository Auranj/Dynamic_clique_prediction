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
        # SNN输出维度为output_dim(64)，EvolveGCN输出维度为hidden_channels(128)
        # 因此连接后的特征维度为64+128=192
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 预测层
        self.predictor = nn.Linear(hidden_channels, output_dim)
        
        # 团演化预测模块 - 确保多头注意力的embed_dim可以被num_heads整除
        self.num_heads = 4
        self.attention_dim = 64  # 明确指定为64，必须是num_heads的倍数
        
        # 特征投影层
        self.feature_projector = nn.Linear(output_dim, self.attention_dim)
        
        # 多头注意力层
        self.evolution_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.num_heads, 
            dropout=dropout,
            batch_first=True  # 使用batch_first=True，使输入形状为[batch, seq_len, embed_dim]
        )
        
        # 强化团特征表示能力
        self.clique_feature_enhancer = nn.Sequential(
            nn.Linear(output_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, output_dim)
        )
        
        # 团演化预测器 - 改进为映射到单个评分
        self.evolution_predictor = nn.Sequential(
            nn.Linear(self.attention_dim * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()  # 使用Sigmoid确保输出在[0,1]范围内
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
        """提取团的特征表示，增强版本。
        
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
                
                # 结合多种池化方法获取更丰富的表示
                max_pool = torch.max(clique_node_features, dim=0)[0]  # 最大池化
                mean_pool = torch.mean(clique_node_features, dim=0)   # 平均池化
                # 新增: 加权注意力池化
                attn_weights = F.softmax(torch.matmul(clique_node_features, mean_pool.unsqueeze(1)), dim=0)
                attn_pool = torch.sum(clique_node_features * attn_weights, dim=0)
                
                # 组合多种池化结果
                clique_feature = (max_pool + mean_pool + attn_pool) / 3
            
            # 增强团特征表示
            enhanced_feature = self.clique_feature_enhancer(clique_feature.unsqueeze(0)).squeeze(0)
            clique_features.append(enhanced_feature)
        
        # 如果团列表为空，返回空张量
        if not clique_features:
            return torch.empty((0, feature_dim), device=features.device)
            
        # 将特征列表堆叠成张量
        return torch.stack(clique_features)
    
    def predict_evolution(self, curr_features: torch.Tensor, curr_cliques: List[List[int]], 
                         prev_features: Optional[torch.Tensor] = None, 
                         prev_cliques: Optional[List[List[int]]] = None) -> torch.Tensor:
        """预测团的演化，增强版本，为每个当前团预测与前一时间步所有团的关联概率。

        Args:
            curr_features: 当前时间步的特征
            curr_cliques: 当前时间步的团列表
            prev_features: 前一时间步的特征（可选）
            prev_cliques: 前一时间步的团列表（可选）

        Returns:
            演化概率矩阵，形状为[num_curr_cliques, num_prev_cliques]（如果提供了prev_cliques）
            或者形状为[num_curr_cliques, 1]（如果未提供prev_cliques）
        """
        # 获取当前团的特征表示
        curr_clique_features = self.extract_clique_features(curr_features, curr_cliques)
        num_curr_cliques = curr_clique_features.size(0)
        
        # 如果没有当前团，返回空张量
        if num_curr_cliques == 0:
            return torch.empty((0, 1), device=curr_features.device)
            
        if prev_features is not None:
            # 投影当前团特征到注意力维度
            curr_clique_features_proj = self.feature_projector(curr_clique_features)
            
            # 如果提供了前一时间步的团列表
            if prev_cliques and len(prev_cliques) > 0:
                # 提取前一时间步的团特征
                prev_clique_features = self.extract_clique_features(prev_features, prev_cliques)
                prev_clique_features_proj = self.feature_projector(prev_clique_features)
                
                num_prev_cliques = prev_clique_features.size(0)
                
                # 如果没有前一时间步的团，返回全零概率
                if num_prev_cliques == 0:
                    return torch.zeros((num_curr_cliques, 1), device=curr_features.device)
                
                # 初始化演化概率矩阵
                evolution_prob_matrix = torch.zeros((num_curr_cliques, num_prev_cliques), device=curr_features.device)
                
                # 计算每个当前团与每个前一时间步团之间的关联概率
                for i in range(num_curr_cliques):
                    # 当前团特征
                    curr_feature = curr_clique_features_proj[i].unsqueeze(0)  # [1, attention_dim]
                    
                    # 使用多头注意力计算当前团与所有前一时间步团之间的注意力
                    # 确保形状正确 - 修复：将形状调整为[batch=1, seq_len=1, embed_dim]
                    # 注意力维度必须与self.attention_dim一致
                    q = curr_feature.unsqueeze(1)  # [1, 1, attention_dim]
                    
                    # 调整prev_clique_features_proj形状为[batch=1, seq_len=num_prev_cliques, embed_dim]
                    k = prev_clique_features_proj.unsqueeze(0)  # [1, num_prev_cliques, attention_dim]
                    v = k.clone()  # 值与键相同，使用clone确保独立的内存
                    
                    # 检查q和k的维度，确保都是attention_dim（64）
                    assert q.size(-1) == self.attention_dim, f"查询的嵌入维度 {q.size(-1)} 与注意力层的嵌入维度 {self.attention_dim} 不匹配"
                    assert k.size(-1) == self.attention_dim, f"键的嵌入维度 {k.size(-1)} 与注意力层的嵌入维度 {self.attention_dim} 不匹配"
                    
                    # 应用多头注意力，得到加权后的特征
                    attn_output, attn_weights = self.evolution_attention(q, k, v)
                    
                    # 对每个前一时间步的团，计算与当前团的关联概率
                    for j in range(num_prev_cliques):
                        # 连接当前团特征和前一时间步团特征
                        prev_feature = prev_clique_features_proj[j]
                        combined_feature = torch.cat([curr_feature.squeeze(0), prev_feature], dim=0)
                        
                        # 预测演化概率
                        prob = self.evolution_predictor(combined_feature.unsqueeze(0))
                        evolution_prob_matrix[i, j] = prob.item()
                
                return evolution_prob_matrix
            else:
                # 如果没有提供前一时间步的团列表，使用全局特征平均值作为参考
                global_prev_feature = torch.mean(prev_features, dim=0, keepdim=True)
                global_prev_feature_proj = self.feature_projector(global_prev_feature)
                
                # 创建一个单一的"前一时间步全局状态"作为参考
                # 确保shape正确 [batch=1, seq_len=1, embed_dim=attention_dim]
                k = global_prev_feature_proj.unsqueeze(0)  # [1, 1, attention_dim]
                v = k.clone()  # 值与键相同
                
                # 初始化演化概率列表
                evolution_probs = []
                
                # 为每个当前团计算演化概率
                for i in range(num_curr_cliques):
                    # 当前团的投影特征 [attention_dim]
                    curr_feature = curr_clique_features_proj[i]
                    
                    # 调整形状为多头注意力的输入要求 [batch=1, seq_len=1, embed_dim]
                    q = curr_feature.unsqueeze(0).unsqueeze(0)  # [1, 1, attention_dim]
                    
                    # 检查q和k的维度
                    assert q.size(-1) == self.attention_dim, f"查询的嵌入维度 {q.size(-1)} 与注意力层的嵌入维度 {self.attention_dim} 不匹配"
                    assert k.size(-1) == self.attention_dim, f"键的嵌入维度 {k.size(-1)} 与注意力层的嵌入维度 {self.attention_dim} 不匹配"
                    
                    # 应用多头注意力机制
                    attn_output, _ = self.evolution_attention(q, k, v)
                    
                    # 取出注意力输出特征 [attention_dim]
                    attn_feat = attn_output.squeeze(0).squeeze(0)  # 从[1, 1, attention_dim]到[attention_dim]
                    
                    # 连接当前团特征和注意力加权特征
                    combined_feature = torch.cat([curr_feature, attn_feat], dim=0)
                    
                    # 预测演化概率
                    prob = self.evolution_predictor(combined_feature.unsqueeze(0))
                    evolution_probs.append(prob)
                
                # 堆叠所有概率为形状[num_curr_cliques, 1]的张量
                if evolution_probs:
                    return torch.cat(evolution_probs, dim=0)
                else:
                    return torch.zeros((num_curr_cliques, 1), device=curr_features.device)
        else:
            # 如果是第一个时间步，无法预测演化
            return torch.zeros((num_curr_cliques, 1), device=curr_features.device)