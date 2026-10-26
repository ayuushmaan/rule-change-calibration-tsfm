"""Training script for CEAFM-CU."""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from ceafm.models.ceafm_cu import CEAFMCU
from ceafm.data.dataset import EventAwareDataset
from ceafm.data.fetch import fetch_ohlcv
from ceafm.data.synthesize import synthesize_events, events_to_meta
from ceafm.training.losses import quantile_loss, gaussian_nll
from ceafm.causal.augment import EventAugmenter


def train(cfg_path: str = "configs/default.yaml"):
    cfg = OmegaConf.load(cfg_path)
    tickers = cfg.data.tickers
    data = fetch_ohlcv(tickers, cfg.data.start_date, cfg.data.end_date)
    if not data:
        raise ValueError("No data fetched")

    t = tickers[0]
    df = data[t]
    events = synthesize_events(df)
    event_meta = events_to_meta(events, df.index)
    text_emb = np.random.randn(len(df), cfg.model.embed_dim).astype(np.float32)

    ds = EventAwareDataset(df, event_meta=event_meta, text_emb=text_emb, lookback=cfg.data.lookback, horizon=cfg.data.horizon, target_col="log_ret")
    dl = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=True)

    model = CEAFMCU(ts_dim=cfg.model.ts_dim, embed_dim=cfg.model.embed_dim, event_meta_dim=cfg.model.event_meta_dim, d_model=cfg.model.d_model, quantiles=cfg.model.quantiles)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    aug = EventAugmenter(p_mask=cfg.training.p_mask, p_swap=cfg.training.p_swap)

    model.train()
    for epoch in range(cfg.training.epochs):
        total = 0.0
        for batch in dl:
            ts_x = batch["ts_x"]
            text_emb_b = batch["text_emb"]
            event_meta_b = batch["event_meta"]
            y = batch["y"].unsqueeze(-1)

            em, te = event_meta_b, text_emb_b
            if cfg.training.use_causal_aug:
                em, te = aug.augment(em, te)

            out = model(ts_x, te, em)
            qloss = quantile_loss(out["quantiles"], y, cfg.model.quantiles)
            nll = gaussian_nll(out["mu"], out["logvar"], y)
            loss = cfg.training.lambda_quantile * qloss + cfg.training.lambda_nll * nll
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if epoch % 10 == 0:
            print(f"epoch {epoch} loss {total/len(dl):.4f}")

    os.makedirs("experiments", exist_ok=True)
    torch.save(model.state_dict(), "experiments/ceafm_cu.pt")
    print("saved experiments/ceafm_cu.pt")


if __name__ == "__main__":
    train()
