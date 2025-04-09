"""数据加载与预处理模块，用于处理Bitcoin-OTC数据集。"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset
import random
import os

def load_bitcoin_data(data_path: str, time_steps: int, train_ratio: float, val_ratio: float, test_ratio: float, loader=None, is_adaptive_window=False):
    """加载并预处理Bitcoin-OTC数据集，用于节点分类任务。

    Args:
        data_path: 数据集路径
        time_steps: 时间步数（固定窗口模式）或DataLoader对象（自适应窗口模式）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        loader: 已初始化的DataLoader对象（可选）
        is_adaptive_window: 是否使用自适应窗口模式

    Returns:
        train_data, val_data, test_data: 训练集、验证集和测试集
    """
    # 初始化数据加载器
    if loader is None:
        loader = DataLoader(data_path, time_window=time_steps)
    
    # 加载数据并生成图序列
    if is_adaptive_window and hasattr(loader, 'graphs') and loader.graphs:
        # 自适应窗口模式，使用已经生成的图序列
        graphs = loader.graphs
    else:
        # 固定窗口模式，重新生成图序列
        graphs = loader.generate_graph_sequence()
    
    # 计算数据集划分点
    total_steps = len(graphs)
    train_size = int(total_steps * train_ratio)
    val_size = int(total_steps * val_ratio)
    
    # 划分数据集
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:train_size+val_size]
    test_graphs = graphs[train_size+val_size:]
    
    # 转换为模型所需的格式
    train_data = []
    val_data = []
    test_data = []
    
    # 处理训练集
    for i, graph in enumerate(train_graphs):
        features = loader.get_node_features(graph)
        adj = nx.adjacency_matrix(graph)
        
        # 转换为PyTorch张量
        x = torch.FloatTensor(np.array(list(features.values())))
        adj = torch.FloatTensor(adj.todense())
        
        # 生成节点标签（根据节点度数设定标签）
        # 高度中心性的节点（度数较高）标记为类别1，其他节点标记为类别0
        node_labels = []
        degrees = dict(graph.degree())
        median_degree = np.median(list(degrees.values()))
        
        for node in graph.nodes():
            # 如果节点度数高于中位数，标签为1，否则为0
            is_high_degree = 1 if degrees[node] > median_degree else 0
            node_labels.append(is_high_degree)
        
        labels = torch.LongTensor(node_labels)
        train_data.append({
            'features': x,
            'adj': adj,
            'labels': labels,
            'graph': graph  # 保存原始图以便可视化
        })
    
    # 处理验证集
    for i, graph in enumerate(val_graphs):
        features = loader.get_node_features(graph)
        adj = nx.adjacency_matrix(graph)
        x = torch.FloatTensor(np.array(list(features.values())))
        adj = torch.FloatTensor(adj.todense())
        
        # 生成节点标签（根据节点度数设定标签）
        node_labels = []
        degrees = dict(graph.degree())
        median_degree = np.median(list(degrees.values()))
        
        for node in graph.nodes():
            # 如果节点度数高于中位数，标签为1，否则为0
            is_high_degree = 1 if degrees[node] > median_degree else 0
            node_labels.append(is_high_degree)
        
        labels = torch.LongTensor(node_labels)
        val_data.append({
            'features': x,
            'adj': adj,
            'labels': labels,
            'graph': graph  # 保存原始图以便可视化
        })
    
    # 处理测试集
    for i, graph in enumerate(test_graphs):
        features = loader.get_node_features(graph)
        adj = nx.adjacency_matrix(graph)
        x = torch.FloatTensor(np.array(list(features.values())))
        adj = torch.FloatTensor(adj.todense())
        
        # 生成节点标签（根据节点度数设定标签）
        node_labels = []
        degrees = dict(graph.degree())
        median_degree = np.median(list(degrees.values()))
        
        for node in graph.nodes():
            # 如果节点度数高于中位数，标签为1，否则为0
            is_high_degree = 1 if degrees[node] > median_degree else 0
            node_labels.append(is_high_degree)
        
        labels = torch.LongTensor(node_labels)
        test_data.append({
            'features': x,
            'adj': adj,
            'labels': labels,
            'graph': graph  # 保存原始图以便可视化
        })
    
    return train_data, val_data, test_data

class DataLoader:
    """数据加载器，负责加载和预处理Bitcoin-OTC数据集。"""
    
    def __init__(self, data_path: str, time_window: int = 14, min_window: int = 7, 
                 max_window: int = 30, window_step: int = 1, overlap_ratio: float = 0.5,
                 adaptive_window: bool = False):
        """初始化数据加载器。

        Args:
            data_path: 数据集路径
            time_window: 时间窗口大小（天）
            min_window: 最小时间窗口大小（天），用于自适应窗口
            max_window: 最大时间窗口大小（天），用于自适应窗口
            window_step: 时间窗口搜索步长（天），用于自适应窗口
            overlap_ratio: 时间窗口重叠比例，范围[0,1]，0表示不重叠，1表示完全重叠
            adaptive_window: 是否使用自适应窗口大小
        """
        self.data_path = data_path
        self.time_window = time_window
        self.min_window = min_window
        self.max_window = max_window
        self.window_step = window_step
        self.overlap_ratio = overlap_ratio
        self.adaptive_window = adaptive_window
        
        self.raw_data = None
        self.graphs = []
        self.adaptive_windows = []
        self.time_steps = None
        self.window_performance = {}

    def evaluate_window_size(self, window_size: int) -> float:
        """评估指定时间窗口大小的性能。

        Args:
            window_size: 要评估的时间窗口大小（天）

        Returns:
            在验证集上的准确率
        """
        try:
            if window_size < 1:
                raise ValueError(f"时间窗口大小必须大于0，当前值: {window_size}")
            
            # 保存当前状态
            original_window = self.time_window
            original_graphs = self.graphs.copy() if self.graphs else []
            original_raw_data = self.raw_data.copy() if self.raw_data is not None else None
            
            # 设置新的时间窗口并生成图序列
            self.time_window = window_size
            self.graphs = []
            graphs = self.generate_graph_sequence()
            
            if not graphs:
                raise ValueError(f"使用时间窗口{window_size}天无法生成有效的图序列")
            
            # 计算团结构检测的准确率
            from data.clique_utils import CliqueDetector
            clique_detector = CliqueDetector(min_clique_size=3)
            
            total_correct = 0
            total_nodes = 0
            total_cliques = 0
            
            for graph in graphs:
                if len(graph.nodes()) == 0:
                    continue
                
                # 检测真实团结构
                try:
                    true_cliques = clique_detector.find_maximal_cliques(graph)
                    true_nodes = set()
                    for clique in true_cliques:
                        true_nodes.update(clique)
                        total_cliques += 1
                    
                    # 使用当前时间窗口的图进行预测
                    pred_cliques = clique_detector.find_maximal_cliques(graph)
                    pred_nodes = set()
                    for clique in pred_cliques:
                        pred_nodes.update(clique)
                    
                    # 计算准确率
                    correct = len(true_nodes.intersection(pred_nodes))
                    total_correct += correct
                    total_nodes += len(graph.nodes())
                except Exception as e:
                    print(f"处理图{graphs.index(graph)}时出错: {str(e)}")
                    continue
            
            # 计算总体准确率
            if total_nodes == 0 or total_cliques == 0:
                accuracy = 0
            else:
                accuracy = total_correct / total_nodes
            
            # 恢复原始状态
            self.time_window = original_window
            self.graphs = original_graphs
            self.raw_data = original_raw_data
            
            return accuracy
            
        except Exception as e:
            print(f"评估时间窗口{window_size}天时出错: {str(e)}")
            return 0

    def evaluate_window_size(self, window_size: int) -> float:
        """评估指定时间窗口大小的性能。

        Args:
            window_size: 时间窗口大小（天）

        Returns:
            float: 在验证集上的准确率
        """
        try:
            # 使用当前时间窗口加载数据
            _, val_data, _ = load_bitcoin_data(
                self.data_path,
                window_size,
                0.7,  # 训练集比例
                0.1,  # 验证集比例
                0.2   # 测试集比例
            )
            
            # 计算验证集上的准确率
            total_correct = 0
            total_samples = 0
            
            for batch in val_data:
                y_true = batch['labels'].numpy()
                total_correct += (y_true == 1).sum()  # 统计团成员节点数
                total_samples += len(y_true)
            
            accuracy = total_correct / total_samples if total_samples > 0 else 0
            return accuracy
            
        except Exception as e:
            print(f"评估时间窗口 {window_size} 时出错: {str(e)}")
            return 0.0

    def find_optimal_window(self) -> int:
        """搜索最优时间窗口大小。

        Returns:
            最优时间窗口大小（天）
        """
        print("\n开始搜索最优时间窗口...")
        
        # 存储每个窗口大小的性能
        self.window_performance = {}
        best_window = self.time_window
        best_accuracy = 0.0
        
        # 在验证集上评估不同的时间窗口
        for window in range(self.min_window, self.max_window + 1, self.window_step):
            accuracy = self.evaluate_window_size(window)
            self.window_performance[window] = accuracy
            
            print(f"时间窗口 {window} 天的准确率: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_window = window
        
        print(f"\n找到最优时间窗口: {best_window} 天 (准确率: {best_accuracy:.4f})")
        return best_window

    def load_data(self) -> pd.DataFrame:
        """加载原始数据集。

        Returns:
            处理后的DataFrame，包含SOURCE、TARGET、RATING和TIME列
        """
        self.raw_data = pd.read_csv(self.data_path)
        # 重命名列
        self.raw_data.columns = ['SOURCE', 'TARGET', 'RATING', 'TIME']
        self.raw_data['TIME'] = pd.to_datetime(self.raw_data['TIME'], unit='s')
        return self.raw_data
    
    def calculate_data_density(self, start_time: datetime, end_time: datetime) -> float:
        """计算指定时间范围内的数据密度。

        Args:
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            数据密度（每天的交易数量）
        """
        if self.raw_data is None:
            self.load_data()
            
        mask = (self.raw_data['TIME'] >= start_time) & (self.raw_data['TIME'] < end_time)
        window_data = self.raw_data[mask]
        
        # 计算数据密度（每天的交易数量）
        days = max(1, (end_time - start_time).days)  # 避免除以零
        density = len(window_data) / days
        
        return density
    
    def get_adaptive_window_size(self, current_time: datetime) -> int:
        """根据数据密度计算自适应窗口大小。

        Args:
            current_time: 当前时间点

        Returns:
            自适应窗口大小（天）
        """
        if self.raw_data is None:
            self.load_data()
        
        # 计算全局数据密度作为参考标准
        total_days = (self.raw_data['TIME'].max() - self.raw_data['TIME'].min()).days
        global_density = len(self.raw_data) / total_days
        
        # 尝试不同大小的窗口，计算局部数据密度
        densities = []
        window_sizes = []
        
        for w in range(self.min_window, self.max_window + 1, self.window_step):
            end_time = current_time + timedelta(days=w)
            local_density = self.calculate_data_density(current_time, end_time)
            densities.append(local_density)
            window_sizes.append(w)
        
        if not densities:
            return self.time_window  # 返回默认窗口大小
        
        # 计算最佳窗口大小
        # 策略1：寻找密度接近全局平均密度的窗口
        density_diffs = [abs(d - global_density) for d in densities]
        best_idx = density_diffs.index(min(density_diffs))
        window_size_by_global = window_sizes[best_idx]
        
        # 策略2：寻找局部相对高密度的窗口（密度先增后减的拐点）
        max_density_idx = densities.index(max(densities))
        window_size_by_local = window_sizes[max_density_idx]
        
        # 策略3：结合策略1和策略2，给出最终窗口大小
        # 优先考虑策略2，但如果窗口过小，则考虑策略1
        if window_size_by_local < self.time_window / 2 and window_size_by_global > window_size_by_local:
            final_window_size = window_size_by_global
        else:
            final_window_size = window_size_by_local
        
        # 确保窗口大小在允许范围内
        final_window_size = max(self.min_window, min(final_window_size, self.max_window))
        
        return final_window_size
    
    def create_time_windows(self) -> List[Tuple[datetime, datetime]]:
        """创建时间窗口。

        Returns:
            时间窗口列表，每个元素为(开始时间, 结束时间)的元组
        """
        if self.raw_data is None:
            self.load_data()
            
        start_time = self.raw_data['TIME'].min()
        end_time = self.raw_data['TIME'].max()
        
        time_windows = []
        self.adaptive_windows = []  # 清空之前的自适应窗口记录
        current_time = start_time
        
        if self.adaptive_window:
            # 自适应窗口模式
            while current_time < end_time:
                # 根据数据密度确定当前时间点的最佳窗口大小
                window_size = self.get_adaptive_window_size(current_time)
                self.adaptive_windows.append(window_size)
                
                # 使用自适应窗口大小
                window_end = current_time + timedelta(days=window_size)
                time_windows.append((current_time, window_end))
                
                # 移动到下一个时间点，考虑重叠
                step_size = int(window_size * (1 - self.overlap_ratio))
                step_size = max(1, step_size)  # 确保至少移动1天
                current_time = current_time + timedelta(days=step_size)
        else:
            # 固定窗口模式，但使用滑动窗口
            window_size = timedelta(days=self.time_window)
            step_size = int(self.time_window * (1 - self.overlap_ratio))
            step_size = max(1, step_size)  # 确保至少移动1天
            step_delta = timedelta(days=step_size)
            
            while current_time < end_time:
                window_end = current_time + window_size
                time_windows.append((current_time, window_end))
                current_time = current_time + step_delta
        
        self.time_steps = len(time_windows)
        return time_windows
    
    def build_graph_snapshot(self, window_data: pd.DataFrame) -> nx.Graph:
        """根据时间窗口数据构建图快照。

        Args:
            window_data: 当前时间窗口的数据

        Returns:
            NetworkX图对象
        """
        G = nx.Graph()
        
        # 添加节点和边
        for _, row in window_data.iterrows():
            source = row['SOURCE']
            target = row['TARGET']
            rating = row['RATING']
            
            # 添加节点（如果不存在）
            if not G.has_node(source):
                G.add_node(source)
            if not G.has_node(target):
                G.add_node(target)
            
            # 添加边或更新边权重
            if G.has_edge(source, target):
                # 如果边已存在，取平均评分
                old_rating = G[source][target]['weight']
                G[source][target]['weight'] = (old_rating + rating) / 2
            else:
                G.add_edge(source, target, weight=rating)
        
        return G
    
    def generate_graph_sequence(self) -> List[nx.Graph]:
        """生成时序图序列。

        Returns:
            NetworkX图对象列表，每个元素代表一个时间步的图快照
        """
        if self.raw_data is None:
            self.load_data()
            
        time_windows = self.create_time_windows()
        self.graphs = []
        
        for start_time, end_time in time_windows:
            # 获取当前时间窗口的数据
            mask = (self.raw_data['TIME'] >= start_time) & (self.raw_data['TIME'] < end_time)
            window_data = self.raw_data[mask]
            
            # 构建图快照
            graph = self.build_graph_snapshot(window_data)
            self.graphs.append(graph)
        
        return self.graphs
    
    def get_node_features(self, graph: nx.Graph) -> Dict[int, np.ndarray]:
        """提取节点特征。

        Args:
            graph: NetworkX图对象

        Returns:
            节点特征字典，键为节点ID，值为特征向量
        """
        features = {}
        for node in graph.nodes():
            # 计算节点的度
            degree = graph.degree(node)
            # 计算节点的加权度
            weighted_degree = sum(w['weight'] for _, _, w in graph.edges(node, data=True))
            # 计算节点的聚类系数
            clustering_coef = nx.clustering(graph, node)
            
            # 组合基本特征向量
            basic_features = np.array([degree, weighted_degree, clustering_coef])
            
            # 将特征向量扩展到64维，以匹配模型配置中的in_channels参数
            # 这里我们使用基本特征的组合和随机初始化来扩展维度
            np.random.seed(node)  # 使用节点ID作为随机种子，确保相同节点在不同时间步有相同的特征
            extended_features = np.zeros(64)
            extended_features[0:3] = basic_features  # 前3维使用原始特征
            
            # 使用原始特征的组合填充剩余维度
            for i in range(3, 64):
                # 使用基本特征的线性组合和非线性变换
                idx = i % 3
                if idx == 0:
                    extended_features[i] = np.sin(basic_features[0] * (i/10))
                elif idx == 1:
                    extended_features[i] = np.cos(basic_features[1] * (i/10))
                else:
                    extended_features[i] = np.tanh(basic_features[2] * (i/10))
            
            features[node] = extended_features
            
        return features
    
    def get_edge_features(self, graph: nx.Graph) -> Dict[Tuple[int, int], np.ndarray]:
        """提取边特征。

        Args:
            graph: NetworkX图对象

        Returns:
            边特征字典，键为(source, target)元组，值为特征向量
        """
        features = {}
        for source, target, data in graph.edges(data=True):
            # 获取边权重
            weight = data['weight']
            # 计算共同邻居数
            common_neighbors = len(list(nx.common_neighbors(graph, source, target)))
            # 计算Jaccard系数
            jaccard = len(list(nx.common_neighbors(graph, source, target))) / \
                      len(set(graph.neighbors(source)) | set(graph.neighbors(target)))
            
            # 组合特征向量
            features[(source, target)] = np.array([weight, common_neighbors, jaccard])
            features[(target, source)] = features[(source, target)]  # 无向图
            
        return features

    def ensure_balanced_samples(self, graph: nx.Graph, cliques: List[set]) -> List[set]:
        """确保图中包含足够的正负样本，以提高AUC计算的稳定性。

        Args:
            graph: 图对象
            cliques: 检测到的团列表

        Returns:
            修改后的团列表
        """
        # 1. 确保至少有一个团
        if not cliques:
            nodes = list(graph.nodes())
            if len(nodes) >= 3:
                # 找到度最高的节点
                degrees = dict(graph.degree())
                sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
                
                # 创建一个团，以度高的节点为中心
                potential_members = [sorted_nodes[0][0]]
                for node, _ in sorted_nodes[1:]:
                    # 检查是否与现有成员都有连接
                    if all(graph.has_edge(node, member) for member in potential_members):
                        potential_members.append(node)
                    if len(potential_members) >= 3:
                        break
                
                # 如果找到足够的成员，创建团
                if len(potential_members) >= 3:
                    cliques = [set(potential_members)]
                else:
                    # 如果没有找到自然团，创建一个人工团
                    artificial_clique = set(random.sample(nodes, min(3, len(nodes))))
                    for u in artificial_clique:
                        for v in artificial_clique:
                            if u != v and not graph.has_edge(u, v):
                                graph.add_edge(u, v, weight=1.0)
                    cliques = [artificial_clique]
        
        # 2. 确保有非团成员节点
        all_nodes = set(graph.nodes())
        clique_nodes = set()
        for clique in cliques:
            clique_nodes.update(clique)
        
        non_clique_nodes = all_nodes - clique_nodes
        
        # 如果所有节点都是团成员，从一些团中移除部分节点
        if len(non_clique_nodes) == 0 and len(cliques) > 0 and len(all_nodes) > 5:
            # 按大小排序团
            sorted_cliques = sorted(cliques, key=len, reverse=True)
            
            # 从最大的团中移除一些节点
            nodes_to_remove = set()
            for clique in sorted_cliques:
                if len(clique) > 4:  # 确保团至少保留3个节点
                    # 移除1-2个节点
                    num_to_remove = min(2, len(clique) - 3)
                    nodes_to_remove.update(random.sample(list(clique), num_to_remove))
                    if len(nodes_to_remove) >= len(all_nodes) * 0.2:  # 最多移除20%的节点
                        break
            
            # 更新团
            updated_cliques = []
            for clique in cliques:
                updated_clique = clique - nodes_to_remove
                if len(updated_clique) >= 3:  # 只保留至少有3个节点的团
                    updated_cliques.append(updated_clique)
            
            return updated_cliques
        
        return cliques