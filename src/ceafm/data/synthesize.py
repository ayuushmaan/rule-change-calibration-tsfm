"""Event synthesis for prototyping."""

import numpy as np
import pandas as pd
from ..data.schemas import Event, EventType, EventImportance


def synthesize_events(market_df: pd.DataFrame, ticker: str, prob: float = 0.05) -> list:
    events = []
    rng = np.random.RandomState(123)
    for idx, row in market_df.iterrows():
        if rng.rand() < prob:
            etype = rng.choice([EventType.EARNINGS, EventType.MACRO, EventType.COMPANY_NEWS])
            imp = rng.choice([EventImportance.MEDIUM, EventImportance.HIGH])
            cred = rng.uniform(0.4, 0.9)
            sent = rng.uniform(-0.5, 0.5)
            events.append(
                Event(
                    event_id=f"{ticker}_{idx}",
                    ticker=ticker,
                    timestamp=idx if isinstance(idx, pd.Timestamp) else pd.to_datetime(idx),
                    event_type=etype,
                    title=f"Synth {etype.value}",
                    source_cred=cred,
                    importance=imp,
                    raw_sentiment=sent,
                )
            )
    return events


def events_to_meta(events: list, index: pd.DatetimeIndex) -> np.ndarray:
    meta = np.zeros((len(index), 8), dtype=np.float32)
    for ev in events:
        # find nearest index
        try:
            pos = index.get_indexer([ev.timestamp], method="nearest")[0]
        except Exception:
            continue
        if pos < 0 or pos >= len(index):
            continue
        meta[pos, 0] = 1.0  # has_event
        meta[pos, 1] = float(ev.importance.value)
        meta[pos, 2] = float(ev.source_cred)
        meta[pos, 3] = float(ev.raw_sentiment) if ev.raw_sentiment is not None else 0.0
        # encode type
        type_map = {
            "earnings": 1.0,
            "macro": 0.8,
            "policy": 0.7,
            "company_news": 0.5,
            "corporate_action": 0.6,
            "other": 0.3,
        }
        meta[pos, 4] = type_map.get(ev.event_type.value, 0.3)
    return meta
