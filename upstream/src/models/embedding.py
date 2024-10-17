import torch
import torch.nn as nn
import numpy as np
from scipy.signal import savgol_filter
from upstream.src.utils.tensors import trunc_normal_


class FilterResampler(nn.Module):
    def __init__(self, window_size=11, poly_order=2, resampling_rate=1, spec_len=2151, total_len=2160):
        # 初始化函数，设置滤波窗口大小、多项式阶数、重采样率和序列长度
        super(FilterResampler, self).__init__()
        self.resampling_rate = resampling_rate
        self.window_size = window_size
        self.poly_order = poly_order
        self.spec_len = spec_len
        self.total_len = total_len
        self.front_endpoint = 350
        self.back_endpoint = 2500

    def forward(self, x):
        """
        对输入的批量数据进行滤波和重采样
        :param x ，尺寸为(batch_size, spec_len)
        :return: x_tensor (torch tensor)，形状为(batch_size, spec_len+pad_len)
        """
        x_np = x.detach().cpu().numpy()

        x_resampled = np.zeros((len(x_np), self.spec_len))
        for i, x_ in enumerate(x_np):
            x_filtered = self.sg_filter(x_[~np.isnan(x_)])
            x_resampled[i] = self.resample(x_filtered, self.resampling_rate)
        # 填充到指定长度total_len
        x_padded = np.pad(x_resampled, ((0, 0), (0, self.total_len - self.spec_len)), mode='constant',
                          constant_values=0)
        x_tensor = torch.tensor(x_padded, dtype=torch.float32)
        return x_tensor.to(x.device)

    @staticmethod
    def sg_filter(x):
        """
        对输入的数据进行Savitzky-Golay滤波
        :param x: (numpy array)，尺寸为(spec_len + 3,)
        :return: y (numpy array)，形状为(spec_len + 3,)
        """
        start, end, spec_len, refl = x[0], x[1], x[2], x[3:]

        sampling_rate = (end - start + 1) / spec_len
        if sampling_rate <= 1.1:
            window_size, poly_order = 21, 2
        elif sampling_rate >= 9.9:
            window_size, poly_order = 5, 2
        else:
            window_size, poly_order = 11, 2
        refl_filtered = savgol_filter(refl, window_size, poly_order)

        y = np.concatenate([x[:3], refl_filtered])
        return y

    def resample(self, x, resampling_rate):
        """
        对滤波后的数据进行重采样
        :param x: (numpy array)，尺寸为(spec_len+3,)
        :param resampling_rate: int，采样分辨率
        :return: y (numpy array)，尺寸为(spec_len,)
        """
        start, end, spec_len, refl = x[0], x[1], x[2], x[3:]

        original_wavelengths = np.linspace(start, end, refl.shape[0])
        target_wavelengths = np.linspace(start, end, int((end - start) / resampling_rate) + 1)
        refl_resampled = np.interp(target_wavelengths, original_wavelengths, refl)

        if len(target_wavelengths) == self.spec_len:
            y = refl_resampled
        else:
            padding_front = np.zeros(int(target_wavelengths[0]) - self.front_endpoint, dtype=np.float32)
            padding_back = np.zeros(self.back_endpoint - int(target_wavelengths[-1]), dtype=np.float32)
            y = np.concatenate([padding_front, refl_resampled, padding_back])
        return y


class Diff(nn.Module):
    def __init__(self):
        super(Diff, self).__init__()
        self.diff = torch.diff

    def compute_diff(self, x, dim=-1):
        """
        计算输入数据的差分导数，复制最后一个值以保持与原始数据相同长度
        :param x: (torch tensor)，尺寸为(batch_size, spec_len)
        :param dim: int，计算的维度
        :return: x (torch tensor)，形状为(batch_size, spec_len)
        """
        x = self.diff(x, dim=dim)
        x = torch.cat((x, x[:, -1:]), dim=dim)
        return x

    def forward(self, x):
        """
        计算输入光谱数据的一阶和二阶差分导数
        :param x: (torch tensor), 尺寸为(batch_size, spec_len)
        :return: fd_x (torch tensor)，尺寸为(batch_size, spec_len)
        :return: sd_x (torch tensor)，尺寸为(batch_size, spec_len)
        """
        fd_x = self.compute_diff(x)
        sd_x = self.compute_diff(fd_x)
        return fd_x, sd_x


class LinearLayerNorm(nn.Module):
    def __init__(self, patch_size, hid_dim):
        super(LinearLayerNorm, self).__init__()
        self.proj = nn.Linear(patch_size, hid_dim)
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x):
        """
        将输入数据用线性层映射到特征维度，并层归一化
        :param x: (torch tensor)，尺寸为(batch_size, patch_len, patch_size)
        :return x: (torch tensor)，尺寸为(batch_size, patch_len, hid_dim)
        """
        x = self.proj(x)
        x = self.norm(x)
        return x


class ConvLayerNorm(nn.Module):
    def __init__(self, patch_size, hid_dim):
        super(ConvLayerNorm, self).__init__()
        self.proj = nn.Conv1d(in_channels=1, out_channels=hid_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x):
        """
        将输入数据用卷积层映射到特征维度，并层归一化
        :param x: (torch tensor)，尺寸为(batch_size, spec_len)
        :return x: (torch tensor)，尺寸为(batch_size, patch_len, hid_dim)
        """
        x = x.unsqueeze(1)  # ->(batch_size, 1, spec_len)
        x = self.proj(x)    # ->(batch_size, hid_dim, patch_len)
        x = x.transpose(1, 2)   # ->(batch_size, patch_len, hid_dim)
        x = self.norm(x)
        return x


class SplitProject(nn.Module):
    def __init__(self, hid_dim, patch_size, proj_type=None, pre_num=3):
        super(SplitProject, self).__init__()
        self.hid_dim = hid_dim
        self.patch_size = patch_size
        self.proj_type = proj_type
        self.pre_num = pre_num

        if proj_type == 'linear':
            self.proj = nn.ModuleList([LinearLayerNorm(patch_size, hid_dim // pre_num) for _ in range(pre_num)])
        elif proj_type == 'conv':
            self.proj = nn.ModuleList([ConvLayerNorm(patch_size, hid_dim // pre_num) for _ in range(pre_num)])
        else:
            raise ValueError('proj_type should be linear or conv')

    def forward(self, x, pre_code):
        """
        按照数据类型，用线性层或卷积层将数据映射到特征维度
        :param x: (torch tensor)，尺寸为(batch_size, spec_len)
        :param pre_code: 0, 1, 2分别代表sg, fd, sd
        :return: x: 映射后的数据，(torch tensor)，尺寸为(batch_size, patch_len, hid_dim)
                 mask:无效数据掩码，(torch tensor)，内含元素为bool，True表示无效，尺寸为(batch_size, patch_len)
        """
        # 按patch_size分割为小块
        x_patched = x.unfold(dimension=1, size=self.patch_size, step=self.patch_size)
        x_patched = x_patched.contiguous().view(x_patched.size(0), -1, self.patch_size)
        # 检测有0值的patch，使mask为True
        mask = (x_patched == 0).any(dim=-1)

        if self.proj_type == 'linear':
            y = self.proj[pre_code](x_patched)
        elif self.proj_type == 'conv':
            y = self.proj[pre_code](x)

        return y, mask


class ManualEmbedding(nn.Module):
    def __init__(self, hid_dim, patch_size, total_len=2160, proj_type='conv', pre_num=3, init_std=0.02):
        super(ManualEmbedding, self).__init__()
        self.hid_dim = hid_dim
        self.patch_size = patch_size
        self.init_std = init_std
        self.num_patches = total_len // patch_size

        self.filter = FilterResampler()
        self.diff = Diff()
        self.proj = SplitProject(hid_dim, patch_size, proj_type, pre_num)

        self.apply(self._init_weights)

    def forward(self, x):
        sg_x = self.filter(x)
        fd_x, sd_x = self.diff(sg_x)

        sg_x, mask = self.proj(sg_x, 0)
        fd_x, _ = self.proj(fd_x, 1)
        sd_x, _ = self.proj(sd_x, 2)

        x = torch.cat([sg_x, fd_x, sd_x], dim=-1)
        return x, mask

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


class AutomaticEmbedding(nn.Module):
    def __init__(self, hid_dim, patch_size, total_len=2160, proj_type='conv', pre_num=1, init_std=0.02):
        super(AutomaticEmbedding, self).__init__()
        self.hid_dim = hid_dim
        self.patch_size = patch_size
        self.init_std = init_std
        self.num_patches = total_len // patch_size

        self.filter = FilterResampler()
        self.proj = SplitProject(hid_dim, patch_size, proj_type, pre_num)

        self.apply(self._init_weights)

    def forward(self, x):
        sg_x = self.filter(x)
        x, mask = self.proj(sg_x, 0)
        return x, mask

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

