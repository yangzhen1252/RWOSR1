import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
# 柱子的名称
categories = ['ResNet18', 'ResNet34', 'ResNet50']

# 第一组柱子的高度
values1 = [61.56, 62.11, 58.95]

# 第二组柱子的高度
values2 = [84.17, 84.33, 85.21]

# 设置每个柱子的宽度
bar_width = 0.25

# 设置每个柱子的位置
bar_positions1 = np.arange(len(categories))
bar_positions2 = bar_positions1 + bar_width

# 绘制第一组柱状图
plt.bar(bar_positions1, values1, width=bar_width, label='Original')

# 绘制第二组柱状图
plt.bar(bar_positions2, values2, width=bar_width, label='OUR')
plt.ylim(top=120)
# 设置图例
plt.legend()

# 设置图表标题和轴标签
plt.title('AUROC of the proposed method on different backbone networks')
plt.xlabel('Backbone networks')
plt.ylabel('AUROC')

for i, v in enumerate(values1):
    plt.text(bar_positions1[i], v + 0.5, str(v), ha='center')

for i, v in enumerate(values2):
    plt.text(bar_positions2[i], v + 0.5, str(v), ha='center')

# 调整 x 轴刻度标签
plt.xticks(bar_positions1 + bar_width / 2, categories)

# 显示图表
plt.show()