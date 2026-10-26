"""Data loading and alignment utilities."""

import pandas as pd
import numpy as np
from typing import Tuple, List
from .schemas import Event


def align_events_to_windows(
    market_df: pd.DataFrame,
    events: List[Event],
    pre_window: str = "1D",
    post_window: str = "1D",
) -> pd.DataFrame:
    """Align events to market timestamps with pre/post windows.
    Returns market_df with event flags and metadata.
    """
    df = market_df.copy()
    df["has_event"] = False
    df["event_type"] = "none"
    df["event_importance"] = 0
    df["event_source_cred"] = 0.0
    df["event_count"] = 0

    for ev in events:
        mask = (df.index >= ev.timestamp - pd.Timedelta(pre_window)) & (
            df.index <= ev.timestamp + pd.Timedelta(post_window)
        )
        if mask.any():
            df.loc[mask, "has_event"] = True
            df.loc[mask, "event_type"] = ev.event_type.value
            df.loc[mask, "event_importance"] = max(
                df.loc[mask, "event_importance"], ev.importance.value
            )
            df.loc[mask, "event_source_cred"] = max(
                df.loc[mask, "event_source_cred"], ev.source_cred
            )
            df.loc[mask, "event_count"] += 1

    return df


def compute_regimes(df: pd.DataFrame, vol_window: int = 20, q: float = 0.7) -> pd.Series:
    """Label regimes as calm/volatile based on rolling realized volatility."""
    if "log_ret" not in df.columns:
        df = df.copy()
        df["log_ret"] = np.log(df["close"]).diff()

    rv = df["log_ret"].rolling(vol_window).std()
    thresh = rv.quantile(q)
    regime = pd.Series("calm", index=df.index)
    regime[rv >= thresh] = "volatile"
    regime[rv.isna()] = "calm"
    return regime
