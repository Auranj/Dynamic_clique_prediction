import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.data_loader import DataLoader, load_bitcoin_data
from models.snn import SimplexNeuralNetwork as SNN
from models.evolvegcn import EvolveGCN
from models.fusion_model import SNNEvolveGCN as FusionModel
from utils.evaluation import evaluate_temporal_prediction, evaluate_clique_prediction
from utils.visualization import plot_metrics_over_time
from config import DATASET_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_model():
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
        print(f"\n使用自适应时间窗口进行训练...")
        # 直接生成图序列，内部会自动计算自适应窗口
        loader.generate_graph_sequence()
        
        # 保存自适应窗口信息
        if not os.path.exists('train_results'):
            os.makedirs('train_results')
        
        # 绘制自适应窗口大小
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(loader.adaptive_windows)), loader.adaptive_windows, marker='o')
        plt.xlabel('Time Step')
        plt.ylabel('Window Size (days)')
        plt.title('Adaptive Window Size by Time Step')
        plt.grid(True)
        plt.savefig('train_results/adaptive_windows.png', dpi=300, bbox_inches='tight')
        
        # 保存窗口大小数据
        window_df = pd.DataFrame({
            'time_step': range(len(loader.adaptive_windows)),
            'window_size': loader.adaptive_windows
        })
        window_df.to_csv('train_results/adaptive_windows.csv', index=False)
        
        print(f"自适应窗口信息:")
        print(f"- 平均窗口大小: {np.mean(loader.adaptive_windows):.2f} 天")
        print(f"- 最小窗口大小: {min(loader.adaptive_windows)} 天")
        print(f"- 最大窗口大小: {max(loader.adaptive_windows)} 天")
        print(f"- 窗口大小标准差: {np.std(loader.adaptive_windows):.2f} 天")
        
        # 使用生成的图序列进行训练
        train_data, val_data, test_data = load_bitcoin_data(
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
        print(f"\n使用最优时间窗口 {best_window} 天进行训练...")
        
        # 使用最优时间窗口加载数据
        train_data, val_data, _ = load_bitcoin_data(
            DATASET_CONFIG['data_path'],
            best_window,  # 使用找到的最优时间窗口
            DATASET_CONFIG['train_ratio'],
            DATASET_CONFIG['val_ratio'],
            DATASET_CONFIG['test_ratio']
        )
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    snn = SNN(**MODEL_CONFIG['snn']).to(device)
    evolvegcn = EvolveGCN(**MODEL_CONFIG['evolvegcn']).to(device)
    fusion_model = FusionModel(**MODEL_CONFIG['fusion']).to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam([
        {'params': snn.parameters()},
        {'params': evolvegcn.parameters()},
        {'params': fusion_model.parameters()}
    ], lr=TRAIN_CONFIG['learning_rate'], weight_decay=TRAIN_CONFIG['weight_decay'])
    
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    best_val_f1 = 0
    patience_counter = 0
    train_metrics_history = []
    val_metrics_history = []
    
    for epoch in range(TRAIN_CONFIG['num_epochs']):
        # 训练阶段
        snn.train()
        evolvegcn.train()
        fusion_model.train()
        
        train_pred_list = []
        train_true_list = []
        train_score_list = []
        
        # 团演化预测评估
        train_clique_evolution_true = []
        train_clique_evolution_pred = []
        
        prev_features = None
        for t in range(len(train_data)):
            batch = train_data[t]
            x, adj, y = batch['features'], batch['adj'], batch['labels']
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            
            # 前向传播
            up_adj = adj.clone()
            down_adj = adj.clone()
            snn_out = snn(x, up_adj, down_adj)
            evolvegcn_out = evolvegcn(x, adj)
            
            # 融合输出
            combined_features = torch.cat([snn_out, evolvegcn_out], dim=1)
            outputs = fusion_model(combined_features)
            
            # 计算节点级别的团成员预测损失
            node_loss = criterion(outputs, y)
            
            # 团演化预测损失
            evolution_loss = 0.0
            if batch['cliques'] and t > 0 and batch['clique_evolution'] is not None:
                # 将团列表从集合转换为列表
                curr_cliques = [list(clique) for clique in batch['cliques']]
                
                # 预测团的演化
                evolution_prob = fusion_model.predict_evolution(
                    x, curr_cliques, prev_features)
                
                # 创建真实的演化标签
                true_evolution = torch.zeros(len(curr_cliques), device=device)
                for prev_idx, curr_idx in batch['clique_evolution']:
                    if curr_idx < len(curr_cliques):
                        true_evolution[curr_idx] = 1.0
                
                # 计算二元交叉熵损失
                if len(evolution_prob) > 0 and len(true_evolution) > 0:
                    evolution_criterion = nn.BCELoss()
                    # 使用第一列作为预测概率，确保维度匹配
                    evolution_loss = evolution_criterion(evolution_prob[:, 0], true_evolution)
                    
                    # 收集团演化预测结果
                    pred_evolution = []
                    for i, prob in enumerate(evolution_prob):
                        if prob[0].item() > 0.5:  # 使用0.5作为阈值，取第一个元素作为预测概率
                            # 简化处理，假设与前一时间步的第一个团有关联
                            pred_evolution.append((0, i))
                    
                    train_clique_evolution_true.append(batch['clique_evolution'])
                    train_clique_evolution_pred.append(pred_evolution)
            
            # 总损失 = 节点级别损失 + 团演化损失
            loss = node_loss
            if evolution_loss > 0:
                loss += evolution_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 收集预测结果
            _, preds = torch.max(outputs, 1)
            train_pred_list.append(preds.cpu().numpy())
            train_true_list.append(y.cpu().numpy())
            train_score_list.append(outputs[:, 1].detach().cpu().numpy())
            
            # 更新前一时间步的特征
            prev_features = x.clone()
        
        # 计算训练指标
        train_metrics = evaluate_temporal_prediction(
            train_true_list, train_pred_list, train_score_list)
        train_metrics_history.append(train_metrics)
        
        # 验证阶段
        snn.eval()
        evolvegcn.eval()
        fusion_model.eval()
        
        val_pred_list = []
        val_true_list = []
        val_score_list = []
        
        # 团演化预测评估
        clique_evolution_true = []
        clique_evolution_pred = []
        
        with torch.no_grad():
            prev_features = None
            for t in range(len(val_data)):
                batch = val_data[t]
                x, adj, y = batch['features'], batch['adj'], batch['labels']
                x, adj, y = x.to(device), adj.to(device), y.to(device)
                
                up_adj = adj.clone()
                down_adj = adj.clone()
                snn_out = snn(x, up_adj, down_adj)
                evolvegcn_out = evolvegcn(x, adj)
                combined_features = torch.cat([snn_out, evolvegcn_out], dim=1)
                outputs = fusion_model(combined_features)
                
                # 节点级别的团成员预测
                _, preds = torch.max(outputs, 1)
                val_pred_list.append(preds.cpu().numpy())
                val_true_list.append(y.cpu().numpy())
                val_score_list.append(outputs[:, 1].detach().cpu().numpy())
                
                # 团演化预测
                if batch['cliques'] and t > 0:  # 确保有团且不是第一个时间步
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
                            if prob[0].item() > 0.5:  # 使用0.5作为阈值，取第一个元素作为预测概率
                                # 找到对应的前一时间步的团索引
                                # 这里简化处理，假设与前一时间步的第一个团有关联
                                pred_evolution.append((0, i))
                        
                        clique_evolution_true.append(true_evolution)
                        clique_evolution_pred.append(pred_evolution)
                
                # 更新前一时间步的特征
                prev_features = x.clone()
        
        # 计算验证指标
        val_metrics = evaluate_temporal_prediction(
            val_true_list, val_pred_list, val_score_list)
        val_metrics_history.append(val_metrics)
        
        # 评估团演化预测
        if clique_evolution_true and clique_evolution_pred:
            # 计算团演化预测的准确率
            evolution_accuracy = 0.0
            evolution_total = 0
            
            for true_evol, pred_evol in zip(clique_evolution_true, clique_evolution_pred):
                # 将预测和真实的演化关系转换为集合以便比较
                true_set = set([(t[0], t[1]) for t in true_evol])
                pred_set = set([(p[0], p[1]) for p in pred_evol])
                
                # 计算正确预测的数量
                correct = len(true_set.intersection(pred_set))
                total = len(true_set)
                
                if total > 0:
                    evolution_accuracy += correct / total
                    evolution_total += 1
            
            if evolution_total > 0:
                evolution_accuracy /= evolution_total
                print(f'团演化预测准确率: {evolution_accuracy:.4f}')
                
                # 将团演化预测准确率添加到验证指标中
                val_metrics['evolution_accuracy'] = evolution_accuracy
        
        # 早停检查
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            # 保存最佳模型
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            
            torch.save({
                'snn_state_dict': snn.state_dict(),
                'evolvegcn_state_dict': evolvegcn.state_dict(),
                'fusion_model_state_dict': fusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_f1': best_val_f1
            }, os.path.join('checkpoints', 'best_model.pth'))
            
            # 保存训练和验证指标图表
            if not os.path.exists('train_results'):
                os.makedirs('train_results')
            
            # 绘制训练指标图表
            plot_metrics_over_time(
                train_metrics_history,
                save_path=os.path.join('train_results', 'train_metrics.png')
            )
            
            # 绘制验证指标图表
            plot_metrics_over_time(
                val_metrics_history,
                save_path=os.path.join('train_results', 'val_metrics.png')
            )
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1}/{TRAIN_CONFIG["num_epochs"]}')
        print(f'Train Metrics: {train_metrics}')
        print(f'Val Metrics: {val_metrics}')
        
        # 添加AUC特定输出，以便更清晰地跟踪
        if 'auc' in train_metrics:
            train_auc = train_metrics['auc']
            val_auc = val_metrics['auc'] if 'auc' in val_metrics else float('nan')
            print(f'AUC Scores - Train: {train_auc:.4f}, Val: {val_auc:.4f}')
        
        if patience_counter >= TRAIN_CONFIG['early_stopping_patience']:
            print('Early stopping triggered')
            break
    
    # 创建检查点目录
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # 创建训练结果目录
    train_results_dir = 'train_results'
    if not os.path.exists(train_results_dir):
        os.makedirs(train_results_dir)
    
    # 绘制训练过程中的指标变化并保存
    plot_metrics_over_time(train_metrics_history, 
                          save_path=os.path.join(train_results_dir, 'train_metrics.png'))
    plot_metrics_over_time(val_metrics_history,
                          save_path=os.path.join(train_results_dir, 'val_metrics.png'))

if __name__ == '__main__':
    train_model()