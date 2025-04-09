import torch
import sys
sys.path.append('/root/DynamicCliqueNet')

from models.fusion_model import SNNEvolveGCN

# 初始化模型，使用与config.py中相同的参数
model = SNNEvolveGCN(in_channels=64, hidden_channels=128, output_dim=64, dropout=0.3)

# 创建测试数据
x = torch.randn(10, 64)  # 当前时间步的节点特征
curr_cliques = [[0, 1, 2], [3, 4, 5]]  # 当前时间步的团
prev_x = torch.randn(10, 64)  # 前一时间步的节点特征
prev_cliques = [[0, 1, 2]]  # 前一时间步的团

print("开始测试多头注意力机制...")

try:
    # 测试predict_evolution方法
    result = model.predict_evolution(x, curr_cliques, prev_x, prev_cliques)
    print(f"测试成功! 输出形状: {result.shape}")
    print(f"输出值: {result}")
except Exception as e:
    print(f"测试失败，错误: {e}")
    import traceback
    traceback.print_exc() 