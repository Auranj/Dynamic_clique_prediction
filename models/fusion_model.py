"""融合模型实现，结合SNN和EvolveGCN的优势，专注于节点分类任务。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from .snn import SimplexNeuralNetwork
from .evolvegcn import EvolveGCN

class SNNEvolveGCN(nn.Module):
    """融合SNN和EvolveGCN的动态图学习模型，专注于节点分类任务。"""
    
    def __init__(self, in_channels: int, hidden_channels: int, output_dim: int, dropout: float = 0.2):
        """初始化融合模型。

        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            output_dim: 输出维度（分类任务中为类别数）
            dropout: Dropout概率
        """
        super().__init__()
        
        # SNN模块，用于提取结构特征
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
        
        # 融合层 - 结合SNN和EvolveGCN的输出
        # SNN输出维度为output_dim(64)，EvolveGCN输出维度为hidden_channels(128)
        # 因此连接后的特征维度为64+128=192
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类器层，用于节点分类
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)  # 二分类任务，输出维度为2
        )
        
        # 添加注意力机制用于团演化预测
        # 使用16作为嵌入维度，这样可以被头数4整除
        self.attention = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        
        # 用于团演化预测的层
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16)  # 降维到注意力层的输入维度
        )
        
        # 演化概率输出层
        self.evolution_prob = nn.Sequential(
            nn.Linear(16, 8), 
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # 输出0-1之间的概率
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入特征，形状为[num_nodes, output_dim + hidden_channels]，包含SNN和EvolveGCN的输出特征

        Returns:
            output: 分类预测结果，形状为[num_nodes, 2]（二分类）
        """
        # 检查输入维度
        if x.size(1) != self.fusion_layer[0].in_features:
            raise ValueError(f"输入张量的特征维度 {x.size(1)} 与融合层期望的维度 {self.fusion_layer[0].in_features} 不匹配")
            
        # 融合SNN和EvolveGCN的特征
        fused_features = self.fusion_layer(x)
        
        # 节点分类
        output = self.classifier(fused_features)
        
        return output
        
    def predict_evolution(self, curr_features: torch.Tensor, curr_cliques: List[List[int]], prev_features: torch.Tensor) -> List[torch.Tensor]:
        """预测团演化概率。
        
        使用注意力机制计算当前时间步中的团与上一时间步特征之间的关系，以预测团演化。
        
        Args:
            curr_features: 当前时间步的节点特征，形状为[num_nodes, feature_dim]
            curr_cliques: 当前时间步检测到的团列表，每个团是一个节点索引列表
            prev_features: 上一时间步的节点特征，形状为[prev_num_nodes, feature_dim]
            
        Returns:
            evolution_probs: 每个当前团与上一时间步状态的关联概率列表
        """
        # 如果没有当前团，返回空列表
        if not curr_cliques:
            return []
            
        # 获取每个团的平均特征表示
        clique_features = []
        for clique in curr_cliques:
            if not clique:  # 跳过空团
                continue
                
            # 提取团中节点的特征
            try:
                nodes_features = curr_features[clique, :]
                # 计算团的平均特征表示
                clique_repr = torch.mean(nodes_features, dim=0, keepdim=True)
                clique_features.append(clique_repr)
            except Exception as e:
                print(f"处理团特征时出错: {str(e)}")
                # 使用零向量作为备用
                clique_repr = torch.zeros((1, curr_features.shape[1]), device=curr_features.device)
                clique_features.append(clique_repr)
        
        # 如果没有有效团，返回空列表
        if not clique_features:
            return []
            
        # 将团特征堆叠为张量
        clique_features = torch.cat(clique_features, dim=0)  # [num_cliques, feature_dim]
        
        # 使用预测器层处理团特征，降维至注意力机制的输入维度
        clique_embeddings = self.predictor(clique_features)  # [num_cliques, embed_dim]
        
        # 准备注意力层的输入
        # 将前一时间步特征也降维
        prev_embeddings = self.predictor(prev_features)  # [prev_num_nodes, embed_dim]
        
        # 添加批次维度，准备进行注意力计算
        query = clique_embeddings.unsqueeze(0)  # [1, num_cliques, embed_dim]
        key = prev_embeddings.unsqueeze(0)  # [1, prev_num_nodes, embed_dim]
        value = prev_embeddings.unsqueeze(0)  # [1, prev_num_nodes, embed_dim]
        
        # 使用注意力机制计算当前团与上一时间步节点之间的关系
        try:
            attn_output, attn_weights = self.attention(query, key, value)
            # attn_output: [1, num_cliques, embed_dim]
            # attn_weights: [1, num_cliques, prev_num_nodes]
            
            # 压缩注意力输出，移除批次维度
            attn_output = attn_output.squeeze(0)  # [num_cliques, embed_dim]
            
            # 计算演化概率
            evolution_probs = [self.evolution_prob(output) for output in attn_output]
            
            return evolution_probs
        except Exception as e:
            print(f"计算注意力时出错: {str(e)}")
            # 返回默认概率（低概率值表示不太可能演化）
            return [torch.tensor([[0.1]], device=curr_features.device) for _ in range(len(clique_features))]