"""Multimodal fusion layers."""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, ts_emb: torch.Tensor, event_emb: torch.Tensor) -> torch.Tensor:
        # ts_emb: (B, d_model), event_emb: (B, d_model)
        q = ts_emb.unsqueeze(1)  # (B, 1, d_model)
        kv = event_emb.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        attn_out = attn_out.squeeze(1)
        g = self.gate(torch.cat([ts_emb, attn_out], dim=-1))
        fused = g * attn_out + (1 - g) * ts_emb
        return self.norm(fused)
