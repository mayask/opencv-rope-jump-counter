from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Optional


class SessionEventType(Enum):
    """Types of session events."""

    SESSION_STARTED = auto()
    SESSION_ENDED = auto()
    MILESTONE_REACHED = auto()
    DAILY_RESET = auto()


@dataclass
class SessionEvent:
    """Represents a session event."""

    event_type: SessionEventType
    timestamp: datetime
    session_jumps: int
    daily_total: int
    milestone: Optional[int] = None
    session_duration_seconds: Optional[float] = None
