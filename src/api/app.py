import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

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


@app.get("/snapshot")
async def get_snapshot():
    """Capture and return current camera frame for debugging."""
    processor = get_processor()
    frame = processor.capture.get_frame(timeout=1.0)

    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available")

    # Save to file
    snapshot_path = "/app/logs/snapshot.jpg"
    cv2.imwrite(snapshot_path, frame)

    return FileResponse(
        snapshot_path,
        media_type="image/jpeg",
        filename="snapshot.jpg",
    )


# MediaPipe drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def generate_debug_frames():
    """Generate MJPEG frames with pose overlay from main processor."""
    processor = get_processor()
    import time
    last_frame_id = 0

    while True:
        # Use the frame already processed by the main processor
        if processor.last_frame is None:
            time.sleep(0.01)
            continue

        # Only send new frames (avoid duplicates)
        if processor.frames_processed == last_frame_id:
            time.sleep(0.01)
            continue
        last_frame_id = processor.frames_processed

        frame = processor.last_frame.copy()
        results = processor.pose_detector.last_results

        # Debug: show frame dimensions
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Frame: {w}x{h}", (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Overlay segmentation mask if available (shows what MediaPipe thinks is "person")
        if results and results.segmentation_mask is not None:
            mask = results.segmentation_mask
            # Resize mask to match frame dimensions if needed
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            # Create colored overlay where person is detected
            mask_3ch = np.stack([mask, mask, mask], axis=-1)
            # Blue tint for segmentation
            overlay = (mask_3ch * np.array([100, 50, 0])).astype(np.uint8)
            frame = cv2.addWeighted(frame, 1.0, overlay, 0.5, 0)

        # Always draw skeleton if pose detected (no filtering)
        if results and results.pose_landmarks:
            nose = results.pose_landmarks.landmark[0]
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            )
            cv2.putText(frame, f"POSE: nose=({nose.x:.2f}, {nose.y:.2f})", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO POSE DETECTED", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

        # Add jump count
        jump_count = processor.jump_detector.session_jumps
        cv2.putText(frame, f"Jumps: {jump_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.get("/debug/stream")
async def debug_stream():
    """Live video stream with pose detection overlay. Open in browser."""
    return StreamingResponse(
        generate_debug_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
