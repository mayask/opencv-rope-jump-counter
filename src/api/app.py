import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException

from ..config.settings import get_config
from .models import (
    CountResponse,
    HealthResponse,
    ResetResponse,
    SessionActionResponse,
    SessionInfo,
    StatusResponse,
    StreamInfo,
)

if TYPE_CHECKING:
    from ..processor.video import VideoProcessor

logger = logging.getLogger(__name__)

# Global reference to processor (set during lifespan)
_processor: "VideoProcessor | None" = None
_start_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _processor, _start_time
    from ..processor.video import VideoProcessor

    _start_time = time.time()
    config = get_config()

    # Create and start the video processor
    _processor = VideoProcessor(config)
    _processor.start()

    logger.info("Application started")

    yield

    # Shutdown
    if _processor:
        await _processor.stop()
    logger.info("Application stopped")


app = FastAPI(
    title="Rope Skipping Counter",
    description="Counts rope skipping exercises using camera feed",
    version="1.0.0",
    lifespan=lifespan,
)


def get_processor() -> "VideoProcessor":
    """Get the video processor instance."""
    if _processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    return _processor


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Docker/Portainer."""
    processor = get_processor()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        stream_connected=processor.capture.is_connected(),
        uptime_seconds=time.time() - _start_time,
    )


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get full system status."""
    processor = get_processor()
    session_status = processor.session_manager.get_status()

    return StatusResponse(
        session=SessionInfo(
            active=session_status.active,
            start_time=session_status.start_time.isoformat()
            if session_status.start_time
            else None,
            duration_seconds=session_status.duration_seconds,
            jumps=session_status.session_jumps,
        ),
        daily_total=session_status.daily_total,
        stream=StreamInfo(
            connected=processor.capture.is_connected(),
            fps=processor.current_fps,
            frames_processed=processor.frames_processed,
        ),
    )


@app.get("/count", response_model=CountResponse)
async def get_count():
    """Get current jump count."""
    processor = get_processor()
    session_status = processor.session_manager.get_status()
    config = get_config()

    milestone_interval = config.webhook.milestone_interval
    next_milestone = (
        (session_status.session_jumps // milestone_interval) + 1
    ) * milestone_interval

    return CountResponse(
        session_count=session_status.session_jumps,
        daily_total=session_status.daily_total,
        last_milestone=session_status.last_milestone,
        next_milestone=next_milestone,
    )


@app.post("/session/start", response_model=SessionActionResponse)
async def start_session():
    """Manually start a session."""
    processor = get_processor()

    if processor.session_manager.session_active:
        return SessionActionResponse(
            status="already_active",
            message="Session is already active",
            count=processor.session_manager.session_jumps,
        )

    processor.session_manager.force_start()
    return SessionActionResponse(
        status="started",
        message="Session started manually",
        count=0,
    )


@app.post("/session/stop", response_model=SessionActionResponse)
async def stop_session():
    """Manually stop the current session."""
    processor = get_processor()

    if not processor.session_manager.session_active:
        return SessionActionResponse(
            status="not_active",
            message="No active session",
        )

    event = processor.session_manager.force_stop()
    return SessionActionResponse(
        status="stopped",
        message="Session stopped",
        count=event.session_jumps if event else 0,
    )


@app.post("/reset", response_model=ResetResponse)
async def reset_counts():
    """Reset all counts."""
    processor = get_processor()
    processor.session_manager.reset_all()
    processor.jump_detector.reset_all()

    return ResetResponse(
        status="reset",
        message="All counts have been reset",
    )
