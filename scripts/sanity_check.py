"""Quick sanity test for CEAFM-CU."""
import torch
from ceafm.models.ceafm_cu import CEAFMCU

model = CEAFMCU(ts_dim=6, embed_dim=384, event_meta_dim=8, d_model=128, quantiles=[0.1,0.5,0.9])
B, T = 4, 60
ts_x = torch.randn(B, T, 6)
text_emb = torch.randn(B, 384)
event_meta = torch.randn(B, 8)
out = model(ts_x, text_emb, event_meta)
print({k: v.shape for k,v in out.items()})
