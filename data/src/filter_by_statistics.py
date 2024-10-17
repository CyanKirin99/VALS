import pandas as pd

# 读取合并后的 CSV 文件
file_path = '../with_label/AllTraits.csv'  # 替换为您的合并文件路径
data = pd.read_csv(file_path)


# 定义一个函数，筛选超出 3σ 的数值
def filter_outliers(series):
    if pd.api.types.is_numeric_dtype(series):
        mean = series.mean()  # 计算均值
        std_dev = series.std()  # 计算标准差
        # 计算上下限
        lower_limit = mean - 3 * std_dev
        upper_limit = mean + 3 * std_dev
        # 返回筛选后的 Series
        return series[(series >= lower_limit) & (series <= upper_limit)]
    return series  # 非数值列不做修改


# 对每一列应用筛选函数
filtered_data = data.apply(filter_outliers)

# 输出处理后的 DataFrame
print(filtered_data)

# 将处理后的 DataFrame 保存为新的 CSV 文件
filtered_data.to_csv('../with_label/AllTraitsFilted.csv', index=False)
