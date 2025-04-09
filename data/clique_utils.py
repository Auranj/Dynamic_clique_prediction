"""团结构检测与标注模块，用于检测和处理图中的团结构。"""

import networkx as nx
import numpy as np
from typing import List, Set, Dict, Tuple

class CliqueDetector:
    """团结构检测器，用于检测和处理图中的团结构。"""
    
    def __init__(self, min_clique_size: int = 3):
        """初始化团结构检测器。

        Args:
            min_clique_size: 最小团大小，默认为3
        """
        self.min_clique_size = min_clique_size
        
    def find_maximal_cliques(self, graph: nx.Graph) -> List[Set[int]]:
        """查找图中的所有最大团。

        Args:
            graph: NetworkX图对象

        Returns:
            最大团列表，每个团是一个节点集合
        """
        cliques = list(nx.find_cliques(graph))
        # 过滤掉小于最小团大小的团
        cliques = [set(c) for c in cliques if len(c) >= self.min_clique_size]
        return cliques
    
    def get_clique_features(self, graph: nx.Graph, clique: Set[int]) -> np.ndarray:
        """计算团的特征。

        Args:
            graph: NetworkX图对象
            clique: 团中的节点集合

        Returns:
            团的特征向量
        """
        # 计算团的大小
        size = len(clique)
        
        # 计算团的密度（实际边数/可能的最大边数）
        subgraph = graph.subgraph(clique)
        density = nx.density(subgraph)
        
        # 计算团的平均边权重
        edge_weights = [w['weight'] for _, _, w in subgraph.edges(data=True)]
        avg_weight = np.mean(edge_weights) if edge_weights else 0
        
        # 计算团的标准差
        weight_std = np.std(edge_weights) if edge_weights else 0
        
        return np.array([size, density, avg_weight, weight_std])
    
    def get_clique_evolution(self, prev_cliques: List[Set[int]], curr_cliques: List[Set[int]]) -> List[Tuple[int, int]]:
        """检测团的演化关系。

        Args:
            prev_cliques: 前一时间步的团列表
            curr_cliques: 当前时间步的团列表

        Returns:
            团的演化关系列表，每个元素为(prev_idx, curr_idx)元组
        """
        evolution = []
        
        for i, prev_c in enumerate(prev_cliques):
            for j, curr_c in enumerate(curr_cliques):
                # 计算两个团的Jaccard相似度
                similarity = len(prev_c & curr_c) / len(prev_c | curr_c)
                
                # 如果相似度超过阈值，认为存在演化关系
                if similarity > 0.5:  # 可调整的阈值
                    evolution.append((i, j))
        
        return evolution
    
    def label_clique_nodes(self, graph: nx.Graph, cliques: List[Set[int]]) -> Dict[int, List[int]]:
        """为节点标注所属的团。

        Args:
            graph: NetworkX图对象
            cliques: 团列表

        Returns:
            节点到团标签的映射，键为节点ID，值为团索引列表
        """
        node_labels = {node: [] for node in graph.nodes()}
        
        for i, clique in enumerate(cliques):
            for node in clique:
                node_labels[node].append(i)
                
        return node_labels
    
    def get_clique_transition_matrix(self, prev_cliques: List[Set[int]], curr_cliques: List[Set[int]]) -> np.ndarray:
        """计算团的转移矩阵。

        Args:
            prev_cliques: 前一时间步的团列表
            curr_cliques: 当前时间步的团列表

        Returns:
            转移矩阵，表示团之间的演化概率
        """
        n_prev = len(prev_cliques)
        n_curr = len(curr_cliques)
        
        # 初始化转移矩阵
        transition_matrix = np.zeros((n_prev, n_curr))
        
        for i, prev_c in enumerate(prev_cliques):
            for j, curr_c in enumerate(curr_cliques):
                # 计算相似度作为转移概率
                similarity = len(prev_c & curr_c) / len(prev_c | curr_c)
                transition_matrix[i, j] = similarity
        
        # 归一化每一行
        row_sums = transition_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # 避免除以0
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        return transition_matrix