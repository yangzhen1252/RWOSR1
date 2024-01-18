import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

categories = ['ResNet18', 'ResNet34', 'ResNet50']


values1 = [61.56, 62.11, 58.95]


values2 = [84.17, 84.33, 85.21]


bar_width = 0.25


bar_positions1 = np.arange(len(categories))
bar_positions2 = bar_positions1 + bar_width


plt.bar(bar_positions1, values1, width=bar_width, label='Original')


plt.bar(bar_positions2, values2, width=bar_width, label='OUR')
plt.ylim(top=120)

plt.legend()


plt.title('AUROC of the proposed method on different backbone networks')
plt.xlabel('Backbone networks')
plt.ylabel('AUROC')

for i, v in enumerate(values1):
    plt.text(bar_positions1[i], v + 0.5, str(v), ha='center')

for i, v in enumerate(values2):
    plt.text(bar_positions2[i], v + 0.5, str(v), ha='center')


plt.xticks(bar_positions1 + bar_width / 2, categories)


plt.show()