from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    stream_connected: bool
    uptime_seconds: float


class SessionInfo(BaseModel):
    active: bool
    start_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    jumps: int


class StreamInfo(BaseModel):
    connected: bool
    fps: float
    frames_processed: int


class StatusResponse(BaseModel):
    session: SessionInfo
    daily_total: int
    stream: StreamInfo


class CountResponse(BaseModel):
    session_count: int
    daily_total: int
    last_milestone: int
    next_milestone: int


class SessionActionResponse(BaseModel):
    status: str
    message: str
    count: Optional[int] = None


class ResetResponse(BaseModel):
    status: str
    message: str
