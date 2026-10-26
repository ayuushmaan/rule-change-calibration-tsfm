"""Torch dataset for event-aware forecasting."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EventAwareDataset(Dataset):
    def __init__(
        self,
        market_df: pd.DataFrame,
        event_meta: np.ndarray = None,
        text_emb: np.ndarray = None,
        lookback: int = 60,
        horizon: int = 1,
        target_col: str = "log_ret",
    ):
        self.df = market_df.copy()
        self.lookback = lookback
        self.horizon = horizon
        self.target_col = target_col

        feats = ["open", "high", "low", "close", "volume"]
        if "realized_vol" in self.df.columns:
            feats.append("realized_vol")
        self.feat_cols = [c for c in feats if c in self.df.columns]
        self.X = self.df[self.feat_cols].values.astype(np.float32)

        self.y = self.df[target_col].shift(-horizon).values.astype(np.float32)
        if event_meta is None:
            self.event_meta = np.zeros((len(self.df), 8), dtype=np.float32)
        else:
            self.event_meta = event_meta.astype(np.float32)
        if text_emb is None:
            self.text_emb = np.zeros((len(self.df), 384), dtype=np.float32)
        else:
            self.text_emb = text_emb.astype(np.float32)

        self.valid = np.arange(lookback, len(self.df) - horizon)

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        i = self.valid[idx]
        ts_x = self.X[i - self.lookback : i]
        return {
            "ts_x": torch.from_numpy(ts_x),
            "text_emb": torch.from_numpy(self.text_emb[i]),
            "event_meta": torch.from_numpy(self.event_meta[i]),
            "y": torch.tensor(self.y[i]),
            "index": i,
        }
