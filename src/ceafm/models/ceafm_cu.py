"""Main model: Causal Event-Aware Financial Model with Calibrated Uncertainty."""

import torch
import torch.nn as nn
from ..encoders.timeseries import PatchTSTEncoder
from ..encoders.event_text import EventTextEncoder
from ..fusion.cross_attn import CrossAttentionFusion
from ..uncertainty.heads import QuantileHead, GaussianHead


class CEAFMCU(nn.Module):
    def __init__(self, ts_dim=6, embed_dim=384, event_meta_dim=8, d_model=128, quantiles=None):
        super().__init__()
        self.ts_enc = PatchTSTEncoder(input_dim=ts_dim, d_model=d_model)
        self.ev_enc = EventTextEncoder(embed_dim=embed_dim, d_model=d_model, event_meta_dim=event_meta_dim)
        self.fusion = CrossAttentionFusion(d_model=d_model)
        self.quantile_head = QuantileHead(d_model=d_model, quantiles=quantiles)
        self.gaussian_head = GaussianHead(d_model=d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, ts_x: torch.Tensor, text_emb: torch.Tensor, event_meta: torch.Tensor):
        ts_z = self.ts_enc(ts_x)
        ev_z = self.ev_enc(text_emb, event_meta)
        fused = self.fusion(ts_z, ev_z)
        fused = self.dropout(fused)
        q_pred = self.quantile_head(fused)
        mu, logvar = self.gaussian_head(fused)
        return {"quantiles": q_pred, "mu": mu, "logvar": logvar, "fused": fused}
