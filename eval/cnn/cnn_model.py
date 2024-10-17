import torch.nn as nn
from upstream.src.models.embedding import FilterResampler
from upstream.src.utils.tensors import trunc_normal_


class CNNModel(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, kernel_size, stride, padding,
                 dropout=0.1, init_std=0.02):
        super(CNNModel, self).__init__()
        self.init_std = init_std
        self.resampler = FilterResampler()

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.total_len = self.resampler.total_len
        self.l1_dim = self._compute_len() * self.hid_channels

        self.conv1 = nn.Conv1d(in_channels, hid_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(hid_channels, hid_channels, kernel_size, stride, padding)

        self.linear1 = nn.Linear(self.l1_dim, self.l1_dim // 2)
        self.linear2 = nn.Linear(self.l1_dim // 2, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

        self.apply(self._init_weights)

    def forward(self, x):
        B, L = x.shape
        x = self.resampler(x)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(B, -1)
        x = self.dropout(self.gelu(self.linear1(x)))
        x = self.linear2(x)
        return x.squeeze()

    def _compute_len(self):
        len_conv1 = (self.total_len + 2 * self.padding - self.kernel_size) // self.stride + 1
        len_conv2 = (len_conv1 + 2 * self.padding - self.kernel_size) // self.stride + 1
        return len_conv2

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
