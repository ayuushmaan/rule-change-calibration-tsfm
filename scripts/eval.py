"""Evaluation script."""

import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from ceafm.models.ceafm_cu import CEAFMCU
from ceafm.data.dataset import EventAwareDataset
from ceafm.data.fetch import fetch_ohlcv
from ceafm.data.synthesize import synthesize_events, events_to_meta
from ceafm.evaluation.metrics import expected_calibration_error, prediction_interval_coverage


def evaluate(cfg_path: str = "configs/default.yaml", ckpt: str = "experiments/ceafm_cu_test.pt"):
    cfg = OmegaConf.load(cfg_path)
    data = fetch_ohlcv(cfg.data.tickers, cfg.data.start_date, cfg.data.end_date)
    t = cfg.data.tickers[0]
    df = data[t].dropna()
    events = synthesize_events(df, t)
    event_meta = events_to_meta(events, df.index)
    text_emb = np.random.randn(len(df), cfg.model.embed_dim).astype(np.float32)

    ds = EventAwareDataset(df, event_meta=event_meta, text_emb=text_emb, lookback=cfg.data.lookback, horizon=cfg.data.horizon)
    dl = DataLoader(ds, batch_size=64)

    model = CEAFMCU(ts_dim=cfg.model.ts_dim, embed_dim=cfg.model.embed_dim, event_meta_dim=cfg.model.event_meta_dim, d_model=cfg.model.d_model, quantiles=cfg.model.quantiles)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    preds = []
    ys = []
    lowers = []
    uppers = []
    with torch.no_grad():
        for batch in dl:
            out = model(batch["ts_x"], batch["text_emb"], batch["event_meta"])
            q = out["quantiles"].numpy()
            preds.append(q[:, 1])
            lowers.append(q[:, 0])
            uppers.append(q[:, 2])
            ys.append(batch["y"].numpy())

    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    lowers = np.concatenate(lowers)
    uppers = np.concatenate(uppers)

    mae = np.mean(np.abs(preds - ys))
    rmse = np.sqrt(np.mean((preds - ys) ** 2))
    cov = prediction_interval_coverage(lowers, uppers, ys)
    probs = 1 / (1 + np.exp(-preds * 10))
    ece = expected_calibration_error(probs, (ys > 0).astype(int))
    print(f"MAE={mae:.6f} RMSE={rmse:.6f} PICP(10-90)={cov:.3f} ECE={ece:.4f}")


if __name__ == "__main__":
    evaluate()
