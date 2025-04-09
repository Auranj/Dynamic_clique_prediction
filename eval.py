import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm import tqdm
from data.data_loader import DataLoader, load_bitcoin_data
from models.snn import SimplexNeuralNetwork as SNN
from models.evolvegcn import EvolveGCN
from models.fusion_model import SNNEvolveGCN as FusionModel
from utils.evaluation import evaluate_temporal_prediction, evaluate_clique_prediction, evaluate_clique_evolution
from utils.visualization import plot_clique_prediction, plot_confusion_matrix, plot_metrics_over_time
from config import DATASET_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, EVAL_CONFIG

def evaluate_model():
    # 创建测试结果目录
    test_results_dir = 'test_results'
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)
    
    # 初始化数据加载器
    loader = DataLoader(
        DATASET_CONFIG['data_path'],
        time_window=DATASET_CONFIG['time_steps'],
        min_window=DATASET_CONFIG['min_window'],
        max_window=DATASET_CONFIG['max_window'],
        window_step=DATASET_CONFIG['window_step'],
        adaptive_window=DATASET_CONFIG.get('adaptive_window', False)  # 支持自适应窗口
    )
    
    # 加载原始数据，计算全局统计信息
    raw_data = loader.load_data()
    print(f"\n数据集信息:")
    print(f"- 总记录数: {len(raw_data)}")
    print(f"- 时间范围: {raw_data['TIME'].min()} 至 {raw_data['TIME'].max()}")
    print(f"- 时间跨度: {(raw_data['TIME'].max() - raw_data['TIME'].min()).days} 天")
    print(f"- 全局数据密度: {len(raw_data) / (raw_data['TIME'].max() - raw_data['TIME'].min()).days:.2f} 条/天")
    
    if DATASET_CONFIG.get('adaptive_window', False):
        print(f"\n使用自适应时间窗口进行评估...")
        # 直接生成图序列，内部会自动计算自适应窗口
        loader.generate_graph_sequence()
        
        # 保存自适应窗口信息
        window_df = pd.DataFrame({
            'time_step': range(len(loader.adaptive_windows)),
            'window_size': loader.adaptive_windows
        })
        window_df.to_csv(os.path.join(test_results_dir, 'adaptive_windows.csv'), index=False)
        
        # 绘制自适应窗口大小
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(loader.adaptive_windows)), loader.adaptive_windows, marker='o')
        plt.xlabel('Time Step')
        plt.ylabel('Window Size (days)')
        plt.title('Adaptive Window Size by Time Step')
        plt.grid(True)
        plt.savefig(os.path.join(test_results_dir, 'adaptive_windows.png'), dpi=300, bbox_inches='tight')
        
        print(f"自适应窗口信息:")
        print(f"- 平均窗口大小: {np.mean(loader.adaptive_windows):.2f} 天")
        print(f"- 最小窗口大小: {min(loader.adaptive_windows)} 天")
        print(f"- 最大窗口大小: {max(loader.adaptive_windows)} 天")
        print(f"- 窗口大小标准差: {np.std(loader.adaptive_windows):.2f} 天")
        
        # 使用生成的图序列进行评估
        _, _, test_data = load_bitcoin_data(
            DATASET_CONFIG['data_path'],
            loader.time_steps,  # 使用自适应窗口生成的时间步数
            DATASET_CONFIG['train_ratio'],
            DATASET_CONFIG['val_ratio'],
            DATASET_CONFIG['test_ratio'],
            loader=loader,
            is_adaptive_window=True
        )
    else:
        # 搜索最优时间窗口
        best_window = loader.find_optimal_window()
        print(f"\n使用最优时间窗口 {best_window} 天进行评估...")
        
        # 使用最优时间窗口加载测试数据
        _, _, test_data = load_bitcoin_data(
            DATASET_CONFIG['data_path'],
            best_window,  # 使用找到的最优时间窗口
            DATASET_CONFIG['train_ratio'],
            DATASET_CONFIG['val_ratio'],
            DATASET_CONFIG['test_ratio']
        )
        
        # 保存时间窗口性能评估结果
        window_performance_df = pd.DataFrame([
            {'window_size': w, 'accuracy': acc}
            for w, acc in loader.window_performance.items()
        ])
        window_performance_df.to_csv(
            os.path.join(test_results_dir, 'window_performance.csv'),
            index=False
        )
        
        # 绘制时间窗口性能曲线
        plt.figure(figsize=(10, 6))
        plt.plot(window_performance_df['window_size'],
                window_performance_df['accuracy'],
                marker='o')
        plt.xlabel('Time Window Size (days)')
        plt.ylabel('Accuracy')
        plt.title('Performance vs Time Window Size')
        plt.grid(True)
        plt.savefig(
            os.path.join(test_results_dir, 'window_performance.png'),
            dpi=300,
            bbox_inches='tight'
        )
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    snn = SNN(**MODEL_CONFIG['snn']).to(device)
    evolvegcn = EvolveGCN(**MODEL_CONFIG['evolvegcn']).to(device)
    fusion_model = FusionModel(**MODEL_CONFIG['fusion']).to(device)
    
    # 加载最佳模型
    checkpoint_path = os.path.join(TRAIN_CONFIG['save_dir'], 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        snn.load_state_dict(checkpoint['snn_state_dict'])
        evolvegcn.load_state_dict(checkpoint['evolvegcn_state_dict'])
        fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
        print(f"加载最佳模型，Epoch: {checkpoint['epoch']}, 验证F1值: {checkpoint['best_val_f1']:.4f}")
    else:
        print("未找到模型检查点，使用随机初始化的模型")
    
    # 设置为评估模式
    snn.eval()
    evolvegcn.eval()
    fusion_model.eval()
    
    # 记录测试结果
    test_pred_list = []
    test_true_list = []
    test_score_list = []
    test_metrics_list = []
    
    # 记录团大小等信息
    clique_sizes = []
    
    # 团演化预测评估
    clique_evolution_true = []
    clique_evolution_pred = []
    clique_evolution_prob = []  # 保存预测概率，用于绘制PR曲线和ROC曲线
    
    with torch.no_grad():
        prev_features = None
        
        for t in range(len(test_data)):
            batch = test_data[t]
            x, adj, y = batch['features'], batch['adj'], batch['labels']
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            
            # 前向传播
            up_adj = adj.clone()
            down_adj = adj.clone()
            snn_out = snn(x, up_adj, down_adj)
            evolvegcn_out = evolvegcn(x, adj)
            combined_features = torch.cat([snn_out, evolvegcn_out], dim=1)
            outputs = fusion_model(combined_features)
            
            # 收集预测结果
            _, preds = torch.max(outputs, 1)
            test_pred_list.append(preds.cpu().numpy())
            test_true_list.append(y.cpu().numpy())
            test_score_list.append(outputs[:, 1].detach().cpu().numpy())
            
            # 计算当前时间步的指标
            current_metrics = evaluate_temporal_prediction(
                [y.cpu().numpy()], [preds.cpu().numpy()], [outputs[:, 1].detach().cpu().numpy()])
            test_metrics_list.append(current_metrics)
            
            # 获取预测为团成员的节点
            pred_nodes = []
            for i, pred in enumerate(preds.cpu().numpy()):
                if pred == 1:  # 预测为团成员
                    # 获取节点ID（假设节点ID与索引对应）
                    node_id = list(batch['graph'].nodes())[i]
                    pred_nodes.append(node_id)
            
            # 记录真实团和预测团的大小
            clique_sizes.append({
                'time_step': t,
                'true_clique_size': len(batch['clique_nodes']),
                'pred_clique_size': len(pred_nodes),
                'overlap_size': len(set(batch['clique_nodes']).intersection(set(pred_nodes)))
            })
            
            # 团演化预测评估（从第二个时间步开始）
            if batch['cliques'] and t > 0:
                # 将团列表从集合转换为列表
                curr_cliques = [list(clique) for clique in batch['cliques']]
                
                # 预测团的演化
                evolution_prob = fusion_model.predict_evolution(
                    x, curr_cliques, prev_features)
                
                # 如果有真实的团演化数据，收集评估数据
                if batch['clique_evolution'] is not None:
                    # 真实的团演化关系
                    true_evolution = batch['clique_evolution']
                    
                    # 根据概率阈值确定预测的演化关系
                    pred_evolution = []
                    for i, prob in enumerate(evolution_prob):
                        if prob[0].item() > 0.5:  # 使用0.5作为阈值
                            # 简化处理，假设与前一时间步的第一个团有关联
                            pred_evolution.append((0, i))
                    
                    clique_evolution_true.append(true_evolution)
                    clique_evolution_pred.append(pred_evolution)
                    
                    # 保存预测概率，用于后续分析
                    probs_list = [(i, prob[0].item()) for i, prob in enumerate(evolution_prob)]
                    clique_evolution_prob.append(probs_list)
            
            # 可视化预测结果（每10个时间步可视化一次）
            if t % 10 == 0 or t == len(test_data) - 1:  # 每10个时间步可视化一次，以及最后一个时间步
                # 可视化真实团和预测团并保存
                plot_clique_prediction(
                    batch['graph'],
                    batch['clique_nodes'],
                    pred_nodes,
                    time_step=t,
                    save_path=os.path.join(test_results_dir, f'clique_prediction_t{t}.png')
                )
            
            # 更新前一时间步的特征
            prev_features = x.clone()
    
    # 计算测试指标
    test_metrics = evaluate_temporal_prediction(
        test_true_list, test_pred_list, test_score_list)
    
    print("\n节点级别测试指标:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 评估团演化预测
    if clique_evolution_true and clique_evolution_pred:
        evolution_metrics = evaluate_clique_evolution(clique_evolution_true, clique_evolution_pred)
        
        print("\n团演化预测指标:")
        for metric, value in evolution_metrics.items():
            print(f"{metric.replace('evolution_', '')}: {value:.4f}")
        
        # 更新测试指标，包括团演化预测指标
        test_metrics.update(evolution_metrics)
        
        # 可视化团演化预测结果
        # 1. 计算每个时间步的演化指标
        time_step_metrics = []
        for i, (true_evol, pred_evol) in enumerate(zip(clique_evolution_true, clique_evolution_pred)):
            time_metrics = evaluate_clique_evolution([true_evol], [pred_evol])
            time_metrics['time_step'] = i + 1  # 从第1个时间步开始计数
            time_step_metrics.append(time_metrics)
        
        # 2. 绘制随时间变化的团演化预测指标
        if time_step_metrics:
            metrics_df = pd.DataFrame(time_step_metrics)
            plt.figure(figsize=(12, 6))
            
            metrics_to_plot = ['evolution_accuracy', 'evolution_precision', 'evolution_recall', 'evolution_f1']
            for metric in metrics_to_plot:
                plt.plot(metrics_df['time_step'], metrics_df[metric], marker='o', label=metric.replace('evolution_', ''))
            
            plt.xlabel('Time Step')
            plt.ylabel('Score')
            plt.title('Evolution Prediction Metrics Over Time')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(test_results_dir, 'evolution_metrics_over_time.png'), dpi=300, bbox_inches='tight')
    
    # 保存测试指标到CSV文件
    metrics_df = pd.DataFrame([test_metrics])
    metrics_df.to_csv(os.path.join(test_results_dir, 'test_metrics.csv'), index=False)
    
    # 保存每个时间步的指标到CSV文件
    timestep_metrics = []
    for t, metrics in enumerate(test_metrics_list):
        metrics_dict = {'time_step': t}
        metrics_dict.update(metrics)
        timestep_metrics.append(metrics_dict)
    
    timestep_df = pd.DataFrame(timestep_metrics)
    timestep_df.to_csv(os.path.join(test_results_dir, 'test_metrics_by_timestep.csv'), index=False)
    
    # 保存团大小信息到CSV文件
    clique_sizes_df = pd.DataFrame(clique_sizes)
    clique_sizes_df.to_csv(os.path.join(test_results_dir, 'clique_sizes.csv'), index=False)
    
    # 绘制混淆矩阵并保存
    if EVAL_CONFIG['visualization']['plot_confusion_matrix']:
        plot_confusion_matrix(
            test_true_list[-1],  # 使用最后一个时间步的结果
            test_pred_list[-1],
            labels=['Non-Clique', 'Clique'],
            save_path=os.path.join(test_results_dir, 'confusion_matrix.png')
        )
    
    # 绘制测试指标随时间变化的曲线
    if EVAL_CONFIG['visualization']['plot_metrics_over_time']:
        # 收集每个时间步的指标
        metrics_by_time = []
        for t, metrics in enumerate(test_metrics_list):
            metrics_dict = {'time_step': t}
            metrics_dict.update(metrics)
            metrics_by_time.append(metrics_dict)
        
        metrics_df = pd.DataFrame(metrics_by_time)
        
        # 绘制各个指标随时间变化的曲线
        plt.figure(figsize=(12, 8))
        
        # 要绘制的指标
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'ap']
        
        for metric in metrics_to_plot:
            if metric in metrics_df.columns:
                plt.plot(metrics_df['time_step'], metrics_df[metric], marker='.', label=metric)
        
        plt.xlabel('Time Step')
        plt.ylabel('Score')
        plt.title('Test Metrics Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(test_results_dir, 'test_metrics_over_time.png'), dpi=300, bbox_inches='tight')
    
    # 分析重叠率
    if clique_sizes:
        # 计算重叠率（真实团和预测团的交集/并集）
        overlap_rates = []
        
        for item in clique_sizes:
            true_size = item['true_clique_size']
            pred_size = item['pred_clique_size']
            overlap = item['overlap_size']
            
            # 计算重叠率（Jaccard相似度）
            union_size = true_size + pred_size - overlap
            if union_size > 0:
                overlap_rate = overlap / union_size
            else:
                overlap_rate = 1.0  # 如果两者都为空，则认为完全匹配
            
            overlap_rates.append({
                'time_step': item['time_step'],
                'overlap_rate': overlap_rate
            })
        
        # 绘制重叠率随时间变化的曲线
        overlap_df = pd.DataFrame(overlap_rates)
        
        plt.figure(figsize=(10, 6))
        plt.plot(overlap_df['time_step'], overlap_df['overlap_rate'], marker='o')
        plt.xlabel('Time Step')
        plt.ylabel('Overlap Rate')
        plt.title('Clique Prediction Overlap Rate Over Time')
        plt.grid(True)
        plt.savefig(os.path.join(test_results_dir, 'overlap_rate.png'), dpi=300, bbox_inches='tight')
        
        # 保存重叠率数据
        overlap_df.to_csv(os.path.join(test_results_dir, 'overlap_rate.csv'), index=False)
    
    print(f"\n测试完成，结果已保存到 {test_results_dir} 目录")

if __name__ == '__main__':
    evaluate_model()