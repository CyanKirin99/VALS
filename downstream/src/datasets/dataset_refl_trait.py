import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import random_split, Dataset, DataLoader
from torch.utils.data import DataLoader, SubsetRandomSampler


class SupervisedDataset(Dataset):
    def __init__(self, spec_path, trait_path, tasks, output_dims):
        self.spec_df_origin = pd.read_csv(spec_path)
        self.trait_df_origin = pd.read_csv(trait_path)
        self.trait_df_origin = self.trait_df_origin[self.trait_df_origin['uid'].isin(self.spec_df_origin['uid'])]

        self.tasks = tasks
        self.output_dims = output_dims

        valid_mask = self.trait_df_origin[tasks].notna().squeeze()
        self.trait_df = self.trait_df_origin[valid_mask].reset_index(drop=True)
        self.spec_df = self.spec_df_origin[self.spec_df_origin['uid'].isin(self.trait_df['uid'])].reset_index(drop=True)

        self.uid = self.spec_df['uid']
        self.scaler_dict = self._compute_scaler()
        self.label_dict = self._map_label()

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        uid = self.uid[idx]
        spec = torch.from_numpy(np.array(self.spec_df.iloc[idx]['start':]).astype(np.float32)).squeeze().float()

        trait = {}
        t = self.trait_df.iloc[idx]
        for tk in self.tasks:
            if self.output_dims[tk] == 1:
                tk_values = torch.tensor(t[tk]).squeeze().float()
                trait[tk] = (tk_values - self.scaler_dict[tk]['Median']) / self.scaler_dict[tk]['IQR']
            else:
                tk_values = t[tk]
                trait[tk] = torch.tensor(self.label_dict[tk][tk_values], dtype=torch.long)

        return spec, trait

    def _compute_scaler(self):
        scaler_dict = {}
        for tk in self.tasks:
            if self.output_dims[tk] == 1:
                tk_series = self.trait_df[tk]

                valid_count = tk_series.count()

                median = tk_series.median()
                Q1 = tk_series.quantile(0.25)
                Q3 = tk_series.quantile(0.75)
                IQR = Q3 - Q1

                tk_scaler = {'Median': median, 'Q1': Q1, 'Q3': Q3, 'IQR': IQR, 'Count': valid_count}
                scaler_dict[tk] = tk_scaler

        return scaler_dict

    def _map_label(self):
        label_dict = {}
        for tk in self.tasks:
            if self.output_dims[tk] > 1:
                unique_labels = self.trait_df[tk].unique()
                label_dict[tk] = {label: idx for idx, label in enumerate(unique_labels)}
        return label_dict


def make_dataset(spec_path, trait_path, tasks, output_dims, split_ratio, batch_size, pin_mem, num_workers):
    dataset = SupervisedDataset(spec_path, trait_path, tasks, output_dims)

    # 找到一个分类任务
    task_to_sample = next((tk for tk in tasks if output_dims[tk] > 1), None)
    if task_to_sample:
        # 获取每个类别的样本索引
        class_indices = {}
        for class_label, idx in dataset.label_dict[task_to_sample].items():
            class_indices[class_label] = dataset.trait_df.index[
                dataset.trait_df[task_to_sample] == class_label].tolist()

        # 找到最小的类别样本数，欠采样
        min_class_size = min(len(indices) for indices in class_indices.values())
        sampled_indices = [idx for indices in class_indices.values() for idx in random.sample(indices, min_class_size)]
        sampler = SubsetRandomSampler(sampled_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=pin_mem, num_workers=num_workers)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem, num_workers=num_workers)

        return dataset, train_loader, test_loader
    else:
        train_ratio, test_ratio = split_ratio
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem, num_workers=num_workers)

        return dataset, train_loader, test_loader

