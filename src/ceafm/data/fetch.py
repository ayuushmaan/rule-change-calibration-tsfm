"""Data fetching utilities."""

import os
from datetime import datetime
from typing import List, Optional

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
        df["log_ret"] = (df["close"] / df["close"].shift(1)).apply(lambda x: __import__("math").log(x) if x > 0 else float("nan"))
        df["realized_vol"] = df["log_ret"].rolling(20).std()
        data[t] = df
    return data


def save_parquet(data: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for t, df in data.items():
        df.to_parquet(os.path.join(out_dir, f"{t}.parquet"))
