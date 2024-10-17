import torch
import torch.nn as nn
from upstream.src.utils.tensors import trunc_normal_


class PE(nn.Module):
    def __init__(self, dim, max_len=1000, pe_type='learnable', init_std=0.02):
        super(PE, self).__init__()
        self.max_len = max_len
        self.dim = dim
        self.init_std = init_std

        pe = torch.zeros(max_len, dim)
        if pe_type == 'learnable':
            self.pe = nn.Parameter(pe.unsqueeze(0))  # (1, max_len, dim)
            trunc_normal_(self.pe, std=self.init_std)
        elif pe_type == 'fixed':
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, dim)
            self.register_buffer('pe', pe)

    def forward(self, x, indices=None):
        """
        :param x: 输入序列，(torch tensor)，尺寸为(batch_size, seq_length, dim)
        :param indices: 每个序列的索引位置，(torch tensor)，尺寸为(batch_size, seq_length)
        :return: x 带有位置编码的序列，(torch tensor)，尺寸为(batch_size, seq_length, dim)
        """
        B, P, D = x.size()
        assert D == self.dim, "数据张量的最后一个维度必须等于位置编码的维度"

        if indices is None:
            indices = torch.arange(P).unsqueeze(0).expand(B, -1)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D).to(x.device)

        pe = self.pe.expand(B, -1, -1)
        pos_encoding = torch.gather(pe, 1, indices_expanded)

        x = x + pos_encoding
        return x


