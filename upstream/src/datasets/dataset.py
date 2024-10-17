import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SelfSupervisedDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path).iloc[:, 1:]
        self.data = np.array(df)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x


def make_dataset(
        file_path,
        batch_size,
        pin_mem,
        num_workers
):
    dataset = SelfSupervisedDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem, num_workers=num_workers)
    return dataset, dataloader
