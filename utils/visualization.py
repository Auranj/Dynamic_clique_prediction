import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_clique_prediction(G, clique_nodes, pred_nodes, time_step=None, save_path=None):
    """
    可视化团结构预测结果
    
    Args:
        G: NetworkX图对象
        clique_nodes: 真实团结构节点列表
        pred_nodes: 预测团结构节点列表
        time_step: 时间步（可选）
        save_path: 保存图表的文件路径（可选）
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    
    # 绘制所有节点和边
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=300)
    
    # 绘制真实团结构节点
    nx.draw_networkx_nodes(G, pos, nodelist=clique_nodes,
                         node_color='green', node_size=500,
                         label='True Clique')
    
    # 绘制预测团结构节点
    nx.draw_networkx_nodes(G, pos, nodelist=pred_nodes,
                         node_color='red', node_size=400,
                         label='Predicted Clique')
    
    plt.title(f'Clique Prediction Results {f"at t={time_step}" if time_step is not None else ""}')
    plt.legend()
    plt.axis('off')
    
    # 如果提供了保存路径，则保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
def plot_metrics_over_time(metrics_list, save_path=None):
    """
    绘制随时间变化的性能指标
    
    Args:
        metrics_list: 包含每个时间步评估指标的列表
        save_path: 保存图表的文件路径（可选）
    """
    time_steps = range(len(metrics_list))
    metrics = metrics_list[0].keys()
    
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        values = [m[metric] for m in metrics_list]
        plt.plot(time_steps, values, marker='o', label=metric)
    
    plt.xlabel('Time Step')
    plt.ylabel('Score')
    plt.title('Performance Metrics over Time')
    plt.legend()
    plt.grid(True)
    
    # 如果提供了保存路径，则保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 标签名称（可选）
        save_path: 保存图表的文件路径（可选）
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    if labels:
        plt.xticks(np.arange(len(labels)) + 0.5, labels)
        plt.yticks(np.arange(len(labels)) + 0.5, labels)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # 如果提供了保存路径，则保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")