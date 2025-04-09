import torch
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

# 添加项目根目录到路径
sys.path.append('/root/DynamicCliqueNet')
from data.data_loader import DataLoader, load_bitcoin_data
from utils.evaluation import evaluate_temporal_prediction
from utils.visualization import plot_confusion_matrix, plot_metrics_over_time

# 导入当前目录的文件
from evolvegcn_classifier import EvolveGCNClassifier
from config import DATASET_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, EVAL_CONFIG, RESULTS_CONFIG

# 使用配置中的目录
RESULTS_BASE_DIR = RESULTS_CONFIG['base_dir']
TEST_RESULTS_DIR = RESULTS_CONFIG['test_dir']
CHECKPOINTS_DIR = RESULTS_CONFIG['checkpoints_dir']

def evaluate_model():
    # 确保结果目录存在
    if not os.path.exists(RESULTS_BASE_DIR):
        os.makedirs(RESULTS_BASE_DIR)
    if not os.path.exists(TEST_RESULTS_DIR):
        os.makedirs(TEST_RESULTS_DIR)
    
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
        window_df.to_csv(os.path.join(TEST_RESULTS_DIR, 'adaptive_windows.csv'), index=False)
        
        # 绘制自适应窗口大小
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(loader.adaptive_windows)), loader.adaptive_windows, marker='o')
        plt.xlabel('Time Step')
        plt.ylabel('Window Size (days)')
        plt.title('Adaptive Window Size by Time Step')
        plt.grid(True)
        plt.savefig(os.path.join(TEST_RESULTS_DIR, 'adaptive_windows.png'), dpi=300, bbox_inches='tight')
        
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
            os.path.join(TEST_RESULTS_DIR, 'window_performance.csv'),
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
            os.path.join(TEST_RESULTS_DIR, 'window_performance.png'),
            dpi=300,
            bbox_inches='tight'
        )
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化EvolveGCNClassifier模型
    model = EvolveGCNClassifier(**MODEL_CONFIG['evolvegcn']).to(device)
    
    # 加载最佳模型 - 使用配置中定义的checkpoints目录
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, 'best_model.pth')
    print(f"尝试从路径加载模型: {checkpoint_path}")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载最佳模型，Epoch: {checkpoint['epoch']}, 验证F1值: {checkpoint['best_val_f1']:.4f}")
    else:
        print(f"未找到模型检查点 {checkpoint_path}，使用随机初始化的模型")
    
    # 设置为评估模式
    model.eval()
    
    # 记录测试结果
    test_pred_list = []
    test_true_list = []
    test_score_list = []
    test_metrics_list = []
    
    with torch.no_grad():
        for t in range(len(test_data)):
            batch = test_data[t]
            x, adj, y = batch['features'], batch['adj'], batch['labels']
            x = x.to(device).float()
            adj = adj.to(device).float()
            y = y.to(device).long()
            
            # 前向传播
            outputs = model(x, adj)
            
            # 收集预测结果
            _, preds = torch.max(outputs, 1)
            test_pred_list.append(preds.cpu().numpy())
            test_true_list.append(y.cpu().numpy())
            test_score_list.append(F.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
            
            # 计算当前时间步的指标
            current_metrics = evaluate_temporal_prediction(
                [y.cpu().numpy()], [preds.cpu().numpy()], [F.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()])
            test_metrics_list.append(current_metrics)
    
    # 计算整体测试指标
    test_metrics = evaluate_temporal_prediction(
        test_true_list, test_pred_list, test_score_list)
    
    print("\n节点级别测试指标:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 保存测试指标到CSV文件
    metrics_df = pd.DataFrame([test_metrics])
    metrics_df.to_csv(os.path.join(TEST_RESULTS_DIR, 'test_metrics.csv'), index=False)
    
    # 保存每个时间步的指标到CSV文件
    timestep_metrics = []
    for t, metrics in enumerate(test_metrics_list):
        metrics_dict = {'time_step': t}
        metrics_dict.update(metrics)
        timestep_metrics.append(metrics_dict)
    
    timestep_df = pd.DataFrame(timestep_metrics)
    timestep_df.to_csv(os.path.join(TEST_RESULTS_DIR, 'test_metrics_by_timestep.csv'), index=False)
    
    # 绘制混淆矩阵并保存
    if EVAL_CONFIG['visualization'].get('plot_confusion_matrix', False):
        plot_confusion_matrix(
            test_true_list[-1],  # 使用最后一个时间步的结果
            test_pred_list[-1],
            labels=['Non-Clique', 'Clique'],
            save_path=os.path.join(TEST_RESULTS_DIR, 'confusion_matrix.png')
        )
    
    # 绘制测试指标随时间变化的曲线
    if EVAL_CONFIG['visualization'].get('plot_metrics_over_time', False):
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
        plt.savefig(os.path.join(TEST_RESULTS_DIR, 'test_metrics_over_time.png'), dpi=300, bbox_inches='tight')
    
    print(f"\n测试完成，结果已保存到 {TEST_RESULTS_DIR} 目录")

if __name__ == '__main__':
    evaluate_model() 