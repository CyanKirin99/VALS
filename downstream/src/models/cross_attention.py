import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)  # Query
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  # Key and Value
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value):
        B, N_q, C = query.shape  # Query shape: (batch_size, query_length, channels)
        B, N_kv, C = key_value.shape  # Key/Value shape: (batch_size, key_value_length, channels)

        q = self.q(query).reshape(B, N_q, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(key_value).reshape(B, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # Attention scores

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)  # Apply attention
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
