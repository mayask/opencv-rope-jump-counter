import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from ..config.settings import get_config

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
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


@app.get("/reset", response_model=ResetResponse)
async def reset_counts():
    """Reset all counts."""
    processor = get_processor()
    processor.session_manager.reset_all()
    processor.jump_detector.reset_all()
    processor.reset_counters()

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


# YOLO pose skeleton connections (pairs of keypoint indices)
YOLO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 12),  # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]


def draw_yolo_skeleton(frame: np.ndarray, keypoints: np.ndarray, color: tuple = (0, 255, 0)) -> None:
    """Draw YOLO pose skeleton on frame."""
    h, w = frame.shape[:2]

    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:  # Only draw visible keypoints
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    # Draw skeleton connections
    for i, j in YOLO_SKELETON:
        x1, y1 = keypoints[i]
        x2, y2 = keypoints[j]
        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)


def generate_debug_frames():
    """Generate MJPEG frames with YOLO pose overlay from main processor."""
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
        orig_h, orig_w = frame.shape[:2]

        # Resize for lower bandwidth debug stream
        scale = 1.0
        if orig_w > 640:
            scale = 640 / orig_w
            frame = cv2.resize(frame, (640, int(orig_h * scale)))
        h, w = frame.shape[:2]

        # Debug: show frame dimensions
        cv2.putText(frame, f"Frame: {w}x{h}", (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Get YOLO detection info (scale keypoints to resized frame)
        keypoints = processor.pose_detector.last_keypoints
        if keypoints is not None and scale != 1.0:
            keypoints = keypoints * scale
        bbox = processor.pose_detector.last_bbox
        if bbox is not None and scale != 1.0:
            bbox = bbox * scale
        rejection = processor.pose_detector.last_rejection_reason

        if keypoints is not None:
            nose_x, nose_y = keypoints[0]
            if rejection:
                # Pose detected but rejected - show in red
                cv2.putText(frame, f"REJECTED: {rejection}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                draw_yolo_skeleton(frame, keypoints, color=(0, 0, 255))
            else:
                # Valid pose - draw green skeleton
                draw_yolo_skeleton(frame, keypoints, color=(0, 255, 0))
                cv2.putText(frame, f"VALID POSE: nose=({nose_x/w:.2f}, {nose_y/h:.2f})", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw bounding box if available
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                color = (0, 0, 255) if rejection else (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        else:
            cv2.putText(frame, f"NO PERSON: {rejection or 'waiting'}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

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
