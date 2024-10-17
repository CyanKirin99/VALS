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


class Regressor(nn.Module):
    def __init__(self, tasks, input_dim, output_dims, init_std=0.02):
        super(Regressor, self).__init__()
        self.input_dim = input_dim
        self.output_dims = output_dims

        self.tasks = tasks
        self.output = nn.ModuleDict({tk: Output(input_dim, output_dims[tk]) for tk in tasks})

        self.init_std = init_std
        self.apply(self._init_weights)

    def forward(self, x, mask=None):
        B, P, D = x.shape
        if mask is not None:
            valid_indices = torch.nonzero(~mask)
            x = x[valid_indices[:, 0], valid_indices[:, 1], :].unsqueeze(1).view(B, -1, D)
        outputs = {}
        for tk in self.tasks:
            outputs[tk] = self.output[tk](x).squeeze()
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

