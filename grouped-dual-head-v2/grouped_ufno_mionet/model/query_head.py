from __future__ import annotations
import torch
from torch import nn
from .mionet import mionet_contract


class AttentionQueryHead(nn.Module):
    def __init__(self, width: int, rank: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads if width % heads == 0 else 1, batch_first=True)
        self.out = nn.Sequential(nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1))
        self.base_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, query_hidden, trunk_rank, medium_tokens, medium_rank, source_rank, source_hidden):
        q = query_hidden + source_hidden[:, None, :]
        attn, _ = self.attn(q, medium_tokens, medium_tokens, need_weights=False)
        base = mionet_contract(medium_rank, source_rank, trunk_rank)
        return base + self.base_scale * self.out(attn).squeeze(-1)
