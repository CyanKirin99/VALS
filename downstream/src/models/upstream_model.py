import torch
import torch.nn as nn
import torch.nn.functional as F


class CompleteUpstreamModel(nn.Module):
    def __init__(self, embedding, encoder, predictor):
        super(CompleteUpstreamModel, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, x):
        x, mask = self.embedding(x)
        x_like = torch.zeros_like(x)
        grouped_indices, grouped_x, grouped_mask = self.group_samples(x, mask)

        for (i_, x_, m_) in zip(grouped_indices, grouped_x, grouped_mask):
            x_ = x_.to(x.device)
            m_ = m_.to(x.device)
            x_like_ = self.encode_predict(x_, m_)
            for (x_one, i_one) in zip(x_like_, i_):
                x_like[i_one] = x_one

        x_like = x_like.to(x.device)
        return x_like, mask

    def encode_predict(self, x_, m_):
        B, P, D = x_.shape
        mask_indices = torch.nonzero(m_)[:, 1].reshape(B, -1)
        valid_indices_extract = torch.nonzero(~m_)
        valid_indices = valid_indices_extract[:, 1].reshape(B, -1)

        valid_x = x_[valid_indices_extract[:, 0], valid_indices_extract[:, 1], :].unsqueeze(1).view(B, -1, D)
        tgt_shp = [mask_indices.size(0), mask_indices.size(1)]

        ctx_rep = self.encoder(valid_x, indices=valid_indices)
        ctx_rep_ = F.layer_norm(ctx_rep, (ctx_rep.size(-1),))

        pred_rep = self.predictor(ctx_rep_, tgt_shp, valid_indices, mask_indices, num_tgt_blk=1)

        x_like_ = torch.zeros_like(x_)
        x_like_[:, valid_indices] = ctx_rep
        x_like_[:, mask_indices] = pred_rep
        return x_like_

    @staticmethod
    def group_samples(x, mask):
        # 计算每个样本中 True 值的数量
        true_counts = mask.sum(dim=1)

        # 使用一个字典来存储分组
        grouped_samples = {}
        for idx, count in enumerate(true_counts):
            count = count.item()  # 转换为 Python int
            if count not in grouped_samples:
                grouped_samples[count] = ([], [], [])
            grouped_samples[count][0].append(idx)  # 添加原索引
            grouped_samples[count][1].append(x[idx])  # 添加数据样本
            grouped_samples[count][2].append(mask[idx])  # 添加掩码样本

        # 生成重组的新张量和索引
        grouped_indices, grouped_x, grouped_mask = [], [], []
        for count, (i_, d_, m_) in grouped_samples.items():
            grouped_indices.append(i_)  # 保存原索引
            grouped_x.append(torch.stack(d_))  # 创建新张量
            grouped_mask.append(torch.stack(m_))  # 创建新掩码

        return grouped_indices, grouped_x, grouped_mask


class SimpleUpstreamModel(nn.Module):
    def __init__(self, embedding, encoder, predictor=None):
        super(SimpleUpstreamModel, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, x):
        x, mask = self.embedding(x)
        x = self.encoder(x, mask=mask)
        x = F.layer_norm(x, (x.size(-1), ))
        return x, mask
