"""Time series encoder for price/volatility history."""

import torch
import torch.nn as nn


class PatchTSTEncoder(nn.Module):
    def __init__(self, input_dim: int = 6, d_model: int = 128, n_heads: int = 4, n_layers: int = 3, patch_len: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.proj = nn.Linear(input_dim * patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        # simple patching
        if T % self.patch_len != 0:
            pad = self.patch_len - (T % self.patch_len)
            x = torch.cat([x, x[:, -pad:, :]], dim=1)
            T = x.shape[1]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_len)  # (B, num_patches, patch_len, C)
        patches = patches.reshape(B, patches.shape[1], -1)
        z = self.proj(patches)
        z = self.encoder(z)
        z = self.norm(z)
        return z.mean(dim=1)  # (B, d_model)
