"""Event taxonomy and data structures for event-aware forecasting."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import pandas as pd


class EventType(Enum):
    EARNINGS = "earnings"
    MACRO = "macro"
    POLICY = "policy"
    COMPANY_NEWS = "company_news"
    CORPORATE_ACTION = "corporate_action"
    OTHER = "other"


class EventImportance(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class Event:
    event_id: str
    ticker: str
    timestamp: pd.Timestamp
    event_type: EventType
    title: str
    description: Optional[str] = None
    source: Optional[str] = None
    source_cred: float = 0.5  # [0, 1]
    importance: EventImportance = EventImportance.MEDIUM
    raw_sentiment: Optional[float] = None  # [-1, 1]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketWindow:
    ticker: str
    start: pd.Timestamp
    end: pd.Timestamp
    features: pd.DataFrame
    target: pd.Series
    events: list[Event] = field(default_factory=list)
