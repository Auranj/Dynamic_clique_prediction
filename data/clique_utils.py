"""团结构检测与标注模块，用于检测和处理图中的团结构。"""

import networkx as nx
import numpy as np
from typing import List, Set, Dict, Tuple, Callable
import itertools

class CliqueDetector:
    """团结构检测器，用于检测和处理图中的团结构。支持多种团定义。"""
    
    def __init__(self, min_clique_size: int = 3, detection_method: str = 'maximal_clique', 
                 k_value: int = 3, min_similarity: float = 0.3):
        """初始化团结构检测器。

        Args:
            min_clique_size: 最小团大小，默认为3
            detection_method: 团检测方法，可选值：'maximal_clique', 'k_clique_community', 'k_core', 'quasi_clique'
            k_value: k值，用于k-core和k-clique-communities算法
            min_similarity: 最小相似度阈值，用于团演化检测
        """
        self.min_clique_size = min_clique_size
        self.detection_method = detection_method
        self.k_value = k_value
        self.min_similarity = min_similarity
        
    def find_cliques(self, graph: nx.Graph) -> List[Set[int]]:
        """根据指定方法查找图中的团结构。

        Args:
            graph: NetworkX图对象

        Returns:
            团列表，每个团是一个节点集合
        """
        if self.detection_method == 'maximal_clique':
            return self.find_maximal_cliques(graph)
        elif self.detection_method == 'k_clique_community':
            return self.find_k_clique_communities(graph)
        elif self.detection_method == 'k_core':
            return self.find_k_cores(graph)
        elif self.detection_method == 'quasi_clique':
            return self.find_quasi_cliques(graph)
        else:
            # 默认使用最大团
            return self.find_maximal_cliques(graph)
        
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
    
    def find_k_clique_communities(self, graph: nx.Graph) -> List[Set[int]]:
        """使用k-clique-communities算法查找社区。

        Args:
            graph: NetworkX图对象

        Returns:
            社区列表，每个社区是一个节点集合
        """
        try:
            # 使用NetworkX的k-clique-communities算法
            communities = list(nx.k_clique_communities(graph, self.k_value))
            # 转换为集合列表并过滤小团
            communities = [set(c) for c in communities if len(c) >= self.min_clique_size]
            return communities
        except Exception as e:
            print(f"查找k-clique-communities时出错: {str(e)}")
            # 如果失败，回退到最大团
            return self.find_maximal_cliques(graph)
    
    def find_k_cores(self, graph: nx.Graph) -> List[Set[int]]:
        """查找图中的k-core子图。

        Args:
            graph: NetworkX图对象

        Returns:
            k-core列表，每个k-core是一个节点集合
        """
        try:
            # 获取k-core子图
            k_core_graph = nx.k_core(graph, self.k_value)
            
            # 如果k-core为空，尝试降低k值
            if len(k_core_graph.nodes()) == 0:
                for k in range(self.k_value-1, 1, -1):
                    k_core_graph = nx.k_core(graph, k)
                    if len(k_core_graph.nodes()) >= self.min_clique_size:
                        break
            
            # 如果k-core太大，可以进一步划分
            cores = []
            if len(k_core_graph.nodes()) > 0:
                # 使用连通分量划分大的k-core
                for component in nx.connected_components(k_core_graph):
                    if len(component) >= self.min_clique_size:
                        cores.append(set(component))
                
                # 如果只有一个大连通分量，可以尝试使用社区检测进一步细分
                if len(cores) == 1 and len(cores[0]) > 20:  # 如果k-core太大
                    try:
                        # 使用Louvain社区检测
                        import community as community_louvain
                        subgraph = graph.subgraph(cores[0])
                        partition = community_louvain.best_partition(subgraph)
                        
                        # 根据分区划分节点
                        communities = {}
                        for node, comm_id in partition.items():
                            if comm_id not in communities:
                                communities[comm_id] = set()
                            communities[comm_id].add(node)
                        
                        # 过滤太小的社区
                        cores = [comm for comm in communities.values() if len(comm) >= self.min_clique_size]
                    except:
                        # 如果社区检测失败，保留原来的k-core
                        pass
            
            return cores
        except Exception as e:
            print(f"查找k-cores时出错: {str(e)}")
            # 如果失败，回退到最大团
            return self.find_maximal_cliques(graph)
    
    def find_quasi_cliques(self, graph: nx.Graph, gamma: float = 0.6) -> List[Set[int]]:
        """查找准团(Quasi-Clique)，即图中密度大于等于gamma的子图。

        Args:
            graph: NetworkX图对象
            gamma: 最小密度阈值，范围[0,1]，默认0.6

        Returns:
            准团列表，每个准团是一个节点集合
        """
        try:
            # 首先查找潜在的种子节点(高度节点)
            degrees = dict(graph.degree())
            high_degree_nodes = [node for node, degree in degrees.items() 
                                if degree >= self.k_value]
            
            quasi_cliques = []
            visited = set()
            
            # 从每个高度节点开始，寻找准团
            for start_node in high_degree_nodes:
                if start_node in visited:
                    continue
                
                # 初始化准团为该节点及其邻居
                candidate = {start_node} | set(graph.neighbors(start_node))
                
                # 迭代优化准团
                while True:
                    # 计算子图密度
                    subgraph = graph.subgraph(candidate)
                    density = nx.density(subgraph)
                    
                    if density < gamma:
                        # 密度太低，移除度最小的节点
                        sub_degrees = dict(subgraph.degree())
                        min_node = min(sub_degrees.items(), key=lambda x: x[1])[0]
                        candidate.remove(min_node)
                        
                        # 如果准团太小，终止
                        if len(candidate) < self.min_clique_size:
                            break
                    else:
                        # 密度达标，检查是否可以添加更多节点
                        neighbors = set()
                        for node in candidate:
                            neighbors.update(graph.neighbors(node))
                        
                        # 找出不在准团中但有许多准团内邻居的节点
                        potential_nodes = []
                        for node in neighbors - candidate:
                            if node in visited:
                                continue
                            
                            # 计算该节点与准团的连接数
                            connections = sum(1 for n in candidate if graph.has_edge(node, n))
                            connection_ratio = connections / len(candidate)
                            
                            if connection_ratio >= gamma:
                                potential_nodes.append((node, connection_ratio))
                        
                        # 如果没有可添加节点，完成这个准团
                        if not potential_nodes:
                            if len(candidate) >= self.min_clique_size:
                                quasi_cliques.append(candidate)
                                visited.update(candidate)
                            break
                        
                        # 添加连接最多的节点
                        best_node = max(potential_nodes, key=lambda x: x[1])[0]
                        candidate.add(best_node)
            
            return [set(qc) for qc in quasi_cliques]
        except Exception as e:
            print(f"查找准团时出错: {str(e)}")
            # 如果失败，回退到最大团
            return self.find_maximal_cliques(graph)
    
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
    
    def get_clique_evolution(self, prev_cliques: List[Set[int]], curr_cliques: List[Set[int]]) -> List[Tuple[int, int, str]]:
        """检测团的演化关系，增强版本，支持检测多种演化模式。

        Args:
            prev_cliques: 前一时间步的团列表
            curr_cliques: 当前时间步的团列表

        Returns:
            团的演化关系列表，每个元素为(prev_idx, curr_idx, evolution_type)元组
            evolution_type可以是：'continue', 'grow', 'shrink', 'split', 'merge'
        """
        # 降低相似度阈值，使其更容易捕捉到团演化关系
        jaccard_threshold = self.min_similarity  # 使用初始化时设置的阈值
        overlap_threshold = max(0.3, self.min_similarity - 0.1)  # 稍微低于主阈值
        containment_threshold = max(0.5, self.min_similarity + 0.1)  # 稍微高于主阈值
        
        # 初始化演化关系记录，保存每种演化类型
        evolution = []  # (prev_idx, curr_idx, evolution_type)
        
        # 跟踪每个团的演化关系
        prev_to_curr = {}  # 前一时间步团到当前时间步团的映射
        curr_to_prev = {}  # 当前时间步团到前一时间步团的映射
        
        # 添加调试计数
        total_comparisons = 0
        detected_evolutions = 0
        
        # 阶段1：计算所有团对之间的相似度，检测基本延续关系
        for i, prev_c in enumerate(prev_cliques):
            for j, curr_c in enumerate(curr_cliques):
                total_comparisons += 1
                
                # 计算多种相似度度量
                
                # 1. Jaccard相似度: |A∩B|/|A∪B|
                jaccard = len(prev_c & curr_c) / len(prev_c | curr_c) if len(prev_c | curr_c) > 0 else 0
                
                # 2. 重叠系数: |A∩B|/min(|A|,|B|)
                overlap = len(prev_c & curr_c) / min(len(prev_c), len(curr_c)) if min(len(prev_c), len(curr_c)) > 0 else 0
                
                # 3. 包含系数: |A∩B|/|A|，表示前一个团有多少被包含在当前团中
                containment_prev = len(prev_c & curr_c) / len(prev_c) if len(prev_c) > 0 else 0
                
                # 4. 包含系数: |A∩B|/|B|，表示当前团有多少来自前一个团
                containment_curr = len(prev_c & curr_c) / len(curr_c) if len(curr_c) > 0 else 0
                
                # 如果相似度高，认为存在演化关系
                if jaccard > jaccard_threshold or overlap > overlap_threshold:
                    # 确定演化类型
                    evolution_type = 'continue'  # 默认为继续
                    
                    # 比较团大小来判断增长或萎缩
                    size_diff_ratio = (len(curr_c) - len(prev_c)) / len(prev_c) if len(prev_c) > 0 else 0
                    
                    if size_diff_ratio > 0.3:  # 增长超过30%
                        evolution_type = 'grow'
                    elif size_diff_ratio < -0.3:  # 萎缩超过30%
                        evolution_type = 'shrink'
                    
                    # 记录演化关系
                    evolution.append((i, j, evolution_type))
                    detected_evolutions += 1
                    
                    # 记录基本映射关系（用于后续检测分裂和合并）
                    if i not in prev_to_curr:
                        prev_to_curr[i] = []
                    prev_to_curr[i].append((j, jaccard))
                    
                    if j not in curr_to_prev:
                        curr_to_prev[j] = []
                    curr_to_prev[j].append((i, jaccard))
        
        # 阶段2：检测分裂和合并模式
        # 分裂：一个前一时间步的团映射到多个当前时间步的团
        for prev_idx, connections in prev_to_curr.items():
            if len(connections) > 1:  # 一个前一时间步的团映射到多个当前团
                # 按相似度排序
                connections.sort(key=lambda x: x[1], reverse=True)
                
                # 主要的延续关系保持为"continue"或之前的类型
                # 其余连接标记为"split"
                main_curr_idx = connections[0][0]
                
                # 查找原始连接的类型
                original_type = None
                for idx, (p_idx, c_idx, e_type) in enumerate(evolution):
                    if p_idx == prev_idx and c_idx == main_curr_idx:
                        original_type = e_type
                        break
                
                # 标记次要连接为"split"
                for curr_idx, _ in connections[1:]:
                    # 检查这个连接是否已经在evolution中
                    connection_exists = False
                    for idx, (p_idx, c_idx, _) in enumerate(evolution):
                        if p_idx == prev_idx and c_idx == curr_idx:
                            # 更新为split类型
                            evolution[idx] = (p_idx, c_idx, 'split')
                            connection_exists = True
                            break
                    
                    # 如果连接不存在，添加新的split连接
                    if not connection_exists:
                        evolution.append((prev_idx, curr_idx, 'split'))
        
        # 合并：多个前一时间步的团映射到一个当前时间步的团
        for curr_idx, connections in curr_to_prev.items():
            if len(connections) > 1:  # 多个前一时间步的团映射到一个当前团
                # 按相似度排序
                connections.sort(key=lambda x: x[1], reverse=True)
                
                # 主要的延续关系保持为"continue"或之前的类型
                # 其余连接标记为"merge"
                main_prev_idx = connections[0][0]
                
                # 查找原始连接的类型
                original_type = None
                for idx, (p_idx, c_idx, e_type) in enumerate(evolution):
                    if p_idx == main_prev_idx and c_idx == curr_idx:
                        original_type = e_type
                        break
                
                # 标记次要连接为"merge"
                for prev_idx, _ in connections[1:]:
                    # 检查这个连接是否已经在evolution中
                    connection_exists = False
                    for idx, (p_idx, c_idx, _) in enumerate(evolution):
                        if p_idx == prev_idx and c_idx == curr_idx:
                            # 更新为merge类型
                            evolution[idx] = (p_idx, c_idx, 'merge')
                            connection_exists = True
                            break
                    
                    # 如果连接不存在，添加新的merge连接
                    if not connection_exists:
                        evolution.append((prev_idx, curr_idx, 'merge'))
        
        # 打印统计信息
        if total_comparisons > 0:
            detection_rate = detected_evolutions / total_comparisons
            # print(f"团演化检测: 共比较{total_comparisons}对团, 检测到{detected_evolutions}个演化关系, "
            #       f"检测率:{detection_rate:.3f}")
            
            # 按类型统计
            evolution_types = {}
            for _, _, e_type in evolution:
                if e_type not in evolution_types:
                    evolution_types[e_type] = 0
                evolution_types[e_type] += 1
            
            # print("团演化类型统计:")
            # for e_type, count in evolution_types.items():
            #     print(f"  - {e_type}: {count}个")
        
        # 为了向后兼容，返回简化的结果格式 (prev_idx, curr_idx)
        simplified_evolution = [(p_idx, c_idx) for p_idx, c_idx, _ in evolution]
        return simplified_evolution
    
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