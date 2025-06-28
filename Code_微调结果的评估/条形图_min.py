import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np

# 设置后端（解决显示问题）
matplotlib.use('Agg')

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def create_loss_comparison():
    """创建训练和验证损失对比图"""

    # 获取所有CSV文件
    csv_files = [f for f in os.listdir('.') if f.startswith('all_loss_') and f.endswith('.csv')]

    if not csv_files:
        print("❌ 未找到符合条件的CSV文件")
        return

    print(f"📊 找到 {len(csv_files)} 个CSV文件")

    # 初始化数据存储列表
    dataset_sizes = []
    min_train_losses = []
    min_eval_losses = []

    # 处理每个CSV文件
    for csv_file in sorted(csv_files):
        try:
            # 从文件名中提取数据集大小
            size = csv_file.split('_')[2].split('.')[0]

            # 读取CSV文件
            df = pd.read_csv(csv_file)

            # 检查必要的列是否存在
            if 'type' not in df.columns or 'loss' not in df.columns:
                print(f"⚠️ 跳过文件 {csv_file}: 缺少必要的列")
                continue

            # 分离train和eval数据
            train_data = df[df['type'] == 'train']
            eval_data = df[df['type'] == 'eval']

            if train_data.empty or eval_data.empty:
                print(f"⚠️ 跳过文件 {csv_file}: 缺少train或eval数据")
                continue

            # 获取最小loss值
            min_train_loss = train_data['loss'].min()
            min_eval_loss = eval_data['loss'].min()

            # 存储数据
            dataset_sizes.append(size)
            min_train_losses.append(min_train_loss)
            min_eval_losses.append(min_eval_loss)

            print(f"✅ 处理完成: {csv_file} (size: {size})")

        except Exception as e:
            print(f"❌ 处理文件 {csv_file} 时出错: {e}")
            continue

    if not dataset_sizes:
        print("❌ 没有成功处理的数据文件")
        return

    # 创建条形图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 设置条形图的位置
    x = np.arange(len(dataset_sizes))
    width = 0.35

    # 绘制条形图
    rects1 = ax.bar(x - width / 2, min_train_losses, width,
                    label='Train Min Loss', color='skyblue', alpha=0.8)
    rects2 = ax.bar(x + width / 2, min_eval_losses, width,
                    label='Eval Min Loss', color='lightcoral', alpha=0.8)

    # 添加标签和标题
    ax.set_xlabel('Dataset Size', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Train vs. Eval Min Loss Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_sizes)
    ax.legend(fontsize=10)

    # 添加网格
    ax.grid(True, alpha=0.3, axis='y')

    # 在条形上方添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    # 调整布局并保存图片
    plt.tight_layout()

    # 保存为多种格式
    plt.savefig('min_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('min_loss_comparison.pdf', bbox_inches='tight')

    # 关闭图形释放内存
    plt.close()

    print("✅ 图表已成功生成并保存:")
    print("   📁 min_loss_comparison.png")
    print("   📁 min_loss_comparison.pdf")

    # 打印数据摘要
    print("\n📈 数据摘要:")
    for i, size in enumerate(dataset_sizes):
        print(f"   Dataset {size}: Train={min_train_losses[i]:.4f}, Eval={min_eval_losses[i]:.4f}")


if __name__ == "__main__":
    create_loss_comparison()