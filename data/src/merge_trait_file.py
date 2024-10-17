import pandas as pd
import os

# 定义要合并的 CSV 文件所在的文件夹路径
folder_path = '../with_label/trait'  # 替换为你的文件夹路径

# 创建一个空的 DataFrame
merged_df = pd.DataFrame()

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否是 CSV 文件
    if filename.endswith('.csv'):
        # 生成文件的完整路径
        file_path = os.path.join(folder_path, filename)

        # 读取 CSV 文件，设置 uid 列为索引
        df = pd.read_csv(file_path, index_col=0)  # 第一列作为索引
        df.columns = [filename[:-4]]  # 取文件名（去掉 .csv），作为列名

        # 合并到主 DataFrame 中
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = merged_df.join(df, how='outer')  # 使用 outer join 合并

# 输出合并后的 DataFrame
print(merged_df)

# 可选：将合并后的 DataFrame 保存为新的 CSV 文件
merged_df.to_csv('../with_label/AllTraits.csv')
