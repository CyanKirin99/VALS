import pandas as pd

# 读取文件
spec_df = pd.read_csv('../with_label/lma_good/refl.csv')
trait_df = pd.read_csv('../with_label/lma_good/trait.csv')

# 合并数据，保留共有的 'uid'
merged_df = pd.merge(spec_df, trait_df, on='uid', how='inner')

# 根据 'Tree' 列的值进行分割
tree_0_df = merged_df[merged_df['Tree'] == 0]
tree_1_df = merged_df[merged_df['Tree'] == 1]

# 将结果分别保存到新的CSV文件
tree_0_df.to_csv('../with_label/lma_good/refl_is_tree.csv', index=False, columns=spec_df.columns)
tree_0_df.to_csv('../with_label/lma_good/trait_is_tree.csv', index=False, columns=trait_df.columns)

tree_1_df.to_csv('../with_label/lma_good/refl_not_tree.csv', index=False, columns=spec_df.columns)
tree_1_df.to_csv('../with_label/lma_good/trait_not_tree.csv', index=False, columns=trait_df.columns)
