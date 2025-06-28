import pandas as pd
import matplotlib
# 设置后端为Agg，这是一个非交互式后端，可以避免显示问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题

# 读取CSV文件
data = pd.read_csv('all_loss_910.csv')

# 分离train和eval数据
train_data = data[data['type'] == 'train']
eval_data = data[data['type'] == 'eval']

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(train_data['step'], train_data['loss'], 'o-', color='#8884d8', linewidth=2, markersize=8, label='train loss')
plt.plot(eval_data['step'], eval_data['loss'], 's-', color='#82ca9d', linewidth=2, markersize=8, label='test loss')

# 添加图表标题和轴标签
plt.title('train vs. test', fontsize=16)
plt.xlabel('Step', fontsize=14)
plt.ylabel('Loss', fontsize=14)

# 不添加网格线
# plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例
plt.legend(fontsize=12)

# 美化图表
plt.tight_layout()

# 保存图表为PNG文件，而不是显示
plt.savefig('train_eval_loss910.png', dpi=300, bbox_inches='tight')
print("图表已保存为 train_eval_loss.png")