import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data.data_loader import DataLoader, load_bitcoin_data
from models.snn import SimplexNeuralNetwork as SNN
from models.evolvegcn import EvolveGCN
from models.fusion_model import SNNEvolveGCN as FusionModel
from utils.evaluation import evaluate_temporal_prediction
from utils.visualization import plot_metrics_over_time
from config import DATASET_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.clique_utils import CliqueDetector

def train_model():
    # 初始化数据加载器
    loader = DataLoader(
        DATASET_CONFIG['data_path'],
        time_window=DATASET_CONFIG['time_steps'],
        min_window=DATASET_CONFIG['min_window'],
        max_window=DATASET_CONFIG['max_window'],
        window_step=DATASET_CONFIG['window_step'],
        overlap_ratio=DATASET_CONFIG.get('overlap_ratio', 0.5),  # 新增：窗口重叠比例
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
    
    # 分析各时间步的类别分布
    print("\n类别分布分析:")
    class_counts = {}
    for t in range(len(train_data)):
        batch = train_data[t]
        y_numpy = batch['labels'].numpy()  # 在移动到GPU前获取numpy数组
        positive_count = np.sum(y_numpy == 1)
        negative_count = np.sum(y_numpy == 0)
        total_count = len(y_numpy)
        class_ratio = positive_count / total_count if total_count > 0 else 0
        
        class_counts[t] = {
            'positive': positive_count,
            'negative': negative_count,
            'total': total_count,
            'positive_ratio': class_ratio
        }
        
        print(f"时间步 {t}: 正样本 {positive_count}, 负样本 {negative_count}, "
              f"正样本比例 {class_ratio:.2f}")
    
    # 计算全局类别权重用于损失函数
    all_labels = []
    for t in range(len(train_data)):
        all_labels.extend(train_data[t]['labels'].numpy())
    
    all_labels = np.array(all_labels)
    positive_weight = 1.0
    negative_weight = 1.0
    
    # 根据类别比例计算权重
    if len(all_labels) > 0:
        positive_count = np.sum(all_labels == 1)
        negative_count = np.sum(all_labels == 0)
        total_count = len(all_labels)
        
        if positive_count > 0 and negative_count > 0:
            # 计算反比权重
            positive_weight = total_count / (2 * positive_count)
            negative_weight = total_count / (2 * negative_count)
    
    # 防止权重过大导致训练不稳定
    max_weight = 10.0
    positive_weight = min(positive_weight, max_weight)
    negative_weight = min(negative_weight, max_weight)
    
    print(f"\n类别权重: 正样本 {positive_weight:.2f}, 负样本 {negative_weight:.2f}")
    
    # 创建类别权重张量
    class_weights = torch.tensor([negative_weight, positive_weight], device=device, dtype=torch.float32)
    
    # 定义带权重的交叉熵损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 定义优化器和学习率调度器
    optimizer = optim.Adam([
        {'params': snn.parameters()},
        {'params': evolvegcn.parameters()},
        {'params': fusion_model.parameters()}
    ], lr=TRAIN_CONFIG['learning_rate'], weight_decay=TRAIN_CONFIG['weight_decay'])
    
    # 学习率调度器，使用余弦退火策略
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=TRAIN_CONFIG['num_epochs'],
        eta_min=TRAIN_CONFIG['learning_rate'] * 0.01
    )
    
    # 训练循环
    best_val_f1 = 0
    best_val_auc = 0
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
        
        epoch_loss = 0.0
        
        for t in range(len(train_data)):
            batch = train_data[t]
            x, adj, y = batch['features'], batch['adj'], batch['labels']
            x = x.to(device).float()
            adj = adj.to(device).float()
            y = y.to(device).long()
            
            # 应用标签平滑
            if TRAIN_CONFIG.get('label_smoothing', False):
                smoothing_factor = 0.1
                # 将硬标签转换为软标签
                y_smooth = torch.zeros(y.size(0), 2, device=device)
                y_smooth.fill_(smoothing_factor / 2)  # 平滑分布
                y_smooth.scatter_(1, y.unsqueeze(1), 1.0 - smoothing_factor)  # 主要类别仍有较高概率
            
            # 前向传播
            up_adj = adj.clone()
            down_adj = adj.clone()
            snn_out = snn(x, up_adj, down_adj)
            evolvegcn_out = evolvegcn(x, adj)
            
            # 融合输出
            combined_features = torch.cat([snn_out, evolvegcn_out], dim=1)
            outputs = fusion_model(combined_features)
            
            # 计算节点级别的分类损失
            if TRAIN_CONFIG.get('label_smoothing', False):
                # 使用KL散度损失
                loss = -torch.sum(y_smooth * F.log_softmax(outputs, dim=1), dim=1).mean()
            else:
                loss = criterion(outputs, y)
            
            # L2正则化项
            l2_lambda = TRAIN_CONFIG.get('l2_lambda', 0.001)
            l2_reg = torch.tensor(0., device=device, dtype=torch.float32)
            for param in snn.parameters():
                l2_reg += torch.norm(param, p=2)
            for param in evolvegcn.parameters():
                l2_reg += torch.norm(param, p=2)
            for param in fusion_model.parameters():
                l2_reg += torch.norm(param, p=2)
            
            loss += l2_lambda * l2_reg
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            max_grad_norm = TRAIN_CONFIG.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(snn.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(evolvegcn.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 收集预测结果
            _, preds = torch.max(outputs, 1)
            train_pred_list.append(preds.cpu().numpy())
            train_true_list.append(y.cpu().numpy())
            train_score_list.append(F.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
        
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
        
        with torch.no_grad():
            for t in range(len(val_data)):
                batch = val_data[t]
                x, adj, y = batch['features'], batch['adj'], batch['labels']
                x = x.to(device).float()
                adj = adj.to(device).float()
                y = y.to(device).long()
                
                up_adj = adj.clone()
                down_adj = adj.clone()
                snn_out = snn(x, up_adj, down_adj)
                evolvegcn_out = evolvegcn(x, adj)
                combined_features = torch.cat([snn_out, evolvegcn_out], dim=1)
                outputs = fusion_model(combined_features)
                
                # 节点级别的分类
                _, preds = torch.max(outputs, 1)
                val_pred_list.append(preds.cpu().numpy())
                val_true_list.append(y.cpu().numpy())
                val_score_list.append(F.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
        
        # 计算验证指标
        val_metrics = evaluate_temporal_prediction(
            val_true_list, val_pred_list, val_score_list)
        val_metrics_history.append(val_metrics)
        
        # 更新学习率
        lr_scheduler.step()
        
        # 判断是否保存当前模型
        save_model = False
        
        # 使用F1作为主要评估指标
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            save_model = True
            print(f"[信息] 保存新的最佳F1模型，F1: {best_val_f1:.4f}")
            
        # 同时监控AUC值
        if 'auc' in val_metrics and val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            # 如果F1没有提升，但AUC提升了，也保存模型
            if not save_model:
                save_model = True
                patience_counter = 0
                print(f"[信息] 保存新的最佳AUC模型，AUC: {best_val_auc:.4f}")
        
        if save_model:
            # 保存最佳模型
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            
            torch.save({
                'snn_state_dict': snn.state_dict(),
                'evolvegcn_state_dict': evolvegcn.state_dict(),
                'fusion_model_state_dict': fusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_f1': best_val_f1,
                'best_val_auc': best_val_auc
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
            
            # 保存完整训练历史
            train_hist_df = pd.DataFrame(train_metrics_history)
            val_hist_df = pd.DataFrame(val_metrics_history)
            
            train_hist_df.to_csv(os.path.join('train_results', 'train_history.csv'), index=False)
            val_hist_df.to_csv(os.path.join('train_results', 'val_history.csv'), index=False)
        else:
            patience_counter += 1
        
        # 输出当前epoch的训练信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{TRAIN_CONFIG["num_epochs"]} - LR: {current_lr:.6f} - Loss: {epoch_loss/len(train_data):.4f}')
        print(f'Train Metrics: {train_metrics}')
        print(f'Val Metrics: {val_metrics}')
        
        # 添加AUC特定输出，以便更清晰地跟踪
        if 'auc' in train_metrics:
            train_auc = train_metrics['auc']
            val_auc = val_metrics['auc'] if 'auc' in val_metrics else float('nan')
            train_ap = train_metrics.get('ap', float('nan'))
            val_ap = val_metrics.get('ap', float('nan'))
            print(f'AUC Scores - Train: {train_auc:.4f}, Val: {val_auc:.4f}')
            print(f'AP Scores - Train: {train_ap:.4f}, Val: {val_ap:.4f}')
        
        # 早停检查
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
    
    # 单独绘制AUC和AP指标的变化趋势
    if train_metrics_history and 'auc' in train_metrics_history[0]:
        # 提取AUC和AP数据
        epochs = range(1, len(train_metrics_history) + 1)
        train_auc = [m.get('auc', float('nan')) for m in train_metrics_history]
        val_auc = [m.get('auc', float('nan')) for m in val_metrics_history]
        train_ap = [m.get('ap', float('nan')) for m in train_metrics_history]
        val_ap = [m.get('ap', float('nan')) for m in val_metrics_history]
        
        # 绘制AUC变化趋势
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_auc, 'b-', label='Train AUC')
        plt.plot(epochs, val_auc, 'r-', label='Val AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('AUC Scores Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(train_results_dir, 'auc_over_time.png'), dpi=300, bbox_inches='tight')
        
        # 绘制AP变化趋势
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_ap, 'b-', label='Train AP')
        plt.plot(epochs, val_ap, 'r-', label='Val AP')
        plt.xlabel('Epoch')
        plt.ylabel('AP')
        plt.title('Average Precision Scores Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(train_results_dir, 'ap_over_time.png'), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    train_model()