import pandas as pd
import os

# 设置要合并的CSV文件所在的文件夹路径
folder_path = '../with_label/refl'  # 修改为你的文件夹路径

# 获取该文件夹下所有CSV文件的路径
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 初始化一个空的列表，用于存储每个DataFrame
df_list = []

# 循环读取每个CSV文件，并将其添加到列表中
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(file_path)  # 读取CSV文件
    df_list.append(df)  # 将DataFrame添加到列表中

# 合并所有DataFrame，使用outer join确保所有列都对齐
merged_df = pd.concat(df_list, ignore_index=True, sort=False)

# 输出合并后的DataFrame
print(merged_df)

# 保存合并后的DataFrame到新的CSV文件
merged_df.to_csv('../with_label/AllRefl.csv', index=False)
