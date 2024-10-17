import pandas as pd

# 读取过滤后的 CSV 文件
file_path = '../with_label/AllTraitsFilted.csv'  # 替换为您的过滤后文件路径
data = pd.read_csv(file_path)

# 创建一个字典来存储统计结果
stats = {}

# 对每一列进行统计分析
for column in data.columns:
    if pd.api.types.is_numeric_dtype(data[column]):  # 检查是否为数值类型
        valid_count = data[column].count()  # 有效样本数量
        max_value = data[column].max()  # 最大值
        min_value = data[column].min()  # 最小值
        mean_value = data[column].mean()  # 平均值
        std_dev = data[column].std()  # 标准差

        # 将结果存储到字典中
        stats[column] = {
            'count': valid_count,
            'mean': mean_value,
            'std': std_dev,
            'max': max_value,
            'min': min_value,
        }
    else:  # 如果不是数值类型，统计各个种类的数量
        value_counts = data[column].value_counts()  # 统计每种类的数量
        for value, count in value_counts.items():
            # 将结果存储到字典中
            stats[f"{column}_{value}"] = {
                'count': count
            }

# 将统计结果转换为 DataFrame
stats_df = pd.DataFrame(stats).T  # 转置，使列名为列，行名为统计项

# 输出统计结果
print(stats_df)

# 将统计结果保存为新的 CSV 文件
stats_df.to_csv('../with_label/statistics_traits.csv', index_label='trait')
