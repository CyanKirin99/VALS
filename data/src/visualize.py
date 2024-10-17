import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# 读取CSV文件
file_path = '../with_label/AllTraitsFilted.csv'  # 请替换为你的CSV文件路径
data = pd.read_csv(file_path)

# 选择数值列
num_data = data.select_dtypes(include=[np.number])

# 创建一个相关性矩阵
correlation_matrix = num_data.corr()

# 创建一个遮罩以只显示下三角
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# 绘制热度图
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', annot=True, fmt=".2f",
            square=True, cbar_kws={"shrink": .8}, linewidths=0.5)

plt.title('Traits Correlation Heatmap', fontsize=18)
plt.savefig('C:/file/Research/projects/TCAF/fig/correlation_heatmap.png',
            bbox_inches='tight', dpi=300)
plt.show()
