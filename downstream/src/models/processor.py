import math

import torch
from torch import nn

from downstream.src.models.cross_attention import CrossAttention
from upstream.src.utils.tensors import trunc_normal_


class Head(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(Head, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, input_dim//4)
        self.fc2 = nn.Linear(input_dim//4, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.dropout(self.gelu(self.fc1(x)))
        x = self.fc2(x)
        return x


class Output(nn.Module):
    def __init__(self, input_dim, output_dim, init_std=0.02):
        super(Output, self).__init__()
        self.init_std = init_std

        self.task_token = nn.Parameter(torch.zeros(1, 1, input_dim), requires_grad=True)
        trunc_normal_(self.task_token, std=self.init_std)

        self.head = Head(input_dim, output_dim)
        self.cross_attention = CrossAttention(input_dim)

    def forward(self, x):
        B, P, D = x.shape

        batch_task_tokens = self.task_token.expand(B, -1, -1)
        y, attn = self.cross_attention(batch_task_tokens, x)
        y += batch_task_tokens

        y = self.head(y)

        return y


class TaskProcessor(nn.Module):
    def __init__(self, tasks, input_dim, output_dims, init_std=0.02):
        super(TaskProcessor, self).__init__()
        self.input_dim = input_dim
        self.output_dims = output_dims

        self.tasks = tasks
        self.output = nn.ModuleDict({tk: Output(input_dim, output_dims[tk]) for tk in tasks})

        self.init_std = init_std
        self.apply(self._init_weights)

    def forward(self, x, mask=None):
        B, P, D = x.shape

        # 如果 mask 为 None，直接进行标准处理
        if mask is None:
            outputs = {}
            for tk in self.tasks:
                outputs[tk] = self.output[tk](x).squeeze()
            return outputs

        # Step 1: 根据 mask 中值为 true 的元素数量分组
        group_indices = {}
        valid_counts = mask.sum(dim=1)
        for i, count in enumerate(valid_counts):
            if count.item() not in group_indices:
                group_indices[count.item()] = []
            group_indices[count.item()].append(i)

        outputs = {}

        # Step 2: 按组提取出有效数据组成新张量
        for count, indices in group_indices.items():
            group_x = x[indices]
            group_mask = mask[indices]

            # 过滤掉无效的数据点
            valid_indices = [torch.nonzero(~group_mask[i]).squeeze(1) for i in range(len(indices))]
            valid_indices = torch.stack(valid_indices).unsqueeze(-1).expand(-1, -1, D)
            filtered_x = torch.gather(group_x, 1, valid_indices)

            # Step 3: 将每组的张量送入后续神经网络处理
            for tk in self.tasks:
                task_output = self.output[tk](filtered_x).squeeze()
                if tk not in outputs:
                    outputs[tk] = {}
                outputs[tk][count] = task_output

        # Step 4: 按分组索引将处理后的张量数据正确排序
        sorted_outputs = {}
        for tk in self.tasks:
            sorted_outputs[tk] = [None] * B
            for count, group_index in group_indices.items():
                for i, idx in enumerate(group_index):
                    sorted_outputs[tk][idx] = outputs[tk][count][i]
            sorted_outputs[tk] = torch.stack(sorted_outputs[tk])
        outputs = sorted_outputs
        return outputs

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

