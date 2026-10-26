"""Data fetching utilities."""

import os
import math
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_ohlcv(tickers: List[str], start: str, end: str, interval: str = "1d") -> dict:
    data = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, interval=interval, progress=False)
        if df.empty:
            continue
        df = df.rename(columns=str.lower)
        df.index = pd.to_datetime(df.index)
        logret = []
        closes = df["close"].values
        for i in range(len(closes)):
            if i == 0:
                logret.append(np.nan)
                continue
            prev = float(closes[i - 1])
            cur = float(closes[i])
            if prev > 0 and cur > 0:
                logret.append(math.log(cur / prev))
            else:
                logret.append(np.nan)
        df["log_ret"] = logret
        df["realized_vol"] = df["log_ret"].rolling(20).std()
        data[t] = df
    return data


def save_parquet(data: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for t, df in data.items():
        df.to_parquet(os.path.join(out_dir, f"{t}.parquet"))
