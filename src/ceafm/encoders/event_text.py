"""Event and text encoder using embeddings."""

import torch
import torch.nn as nn


class EventTextEncoder(nn.Module):
    def __init__(self, embed_dim: int = 384, d_model: int = 128, event_meta_dim: int = 8):
        super().__init__()
        self.text_proj = nn.Linear(embed_dim, d_model)
        self.meta_proj = nn.Linear(event_meta_dim, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, text_emb: torch.Tensor, event_meta: torch.Tensor) -> torch.Tensor:
        # text_emb: (B, embed_dim), event_meta: (B, event_meta_dim)
        t = self.text_proj(text_emb)
        m = self.meta_proj(event_meta)
        z = self.fuse(torch.cat([t, m], dim=-1))
        return z  # (B, d_model)
