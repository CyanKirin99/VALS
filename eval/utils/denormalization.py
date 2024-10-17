import torch.nn as nn


class Denormalize(nn.Module):
    def __init__(self, scaler_dict):
        super().__init__()
        self.scaler_dict = scaler_dict

    def forward(self, o, t, output_dims):
        o_denorm, t_denorm = {}, {}
        for tk in o.keys():
            if output_dims[tk] == 1:
                o_denorm[tk] = self.denorm(o[tk], tk)
                t_denorm[tk] = self.denorm(t[tk], tk)
            else:
                o_denorm[tk] = o[tk]
                t_denorm[tk] = t[tk]
        return o_denorm, t_denorm

    def denorm(self, x, tk):
        median = self.scaler_dict[tk]['Median']
        IQR = self.scaler_dict[tk]['IQR']
        x_denorm = x * IQR + median
        return x_denorm
