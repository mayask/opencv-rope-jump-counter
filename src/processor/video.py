import asyncio
import logging
import threading
import time
from collections import deque
from typing import Optional

from ..config.settings import AppConfig
from ..detection.jump import JumpDetector
from ..detection.pose import PoseDetector
from ..notifications.webhook import WebhookSender
from ..session.events import SessionEvent
from ..session.manager import SessionManager
from ..stream.capture import RTSPCapture

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Main video processing orchestrator."""

    def __init__(self, config: AppConfig):
        self.config = config

        # Initialize components
        self.capture = RTSPCapture(
            rtsp_url=config.stream.rtsp_url,
            target_fps=config.stream.target_fps,
            reconnect_delay=config.stream.reconnect_delay,
        )

        self.pose_detector = PoseDetector(
            model_complexity=config.detection.model_complexity,
            min_detection_confidence=config.detection.min_detection_confidence,
            min_tracking_confidence=config.detection.min_tracking_confidence,
            track_point="nose",  # Head movement is more pronounced for rope skipping
        )

        self.jump_detector = JumpDetector()

        self.session_manager = SessionManager(
            start_threshold=config.session.start_threshold,
            stop_timeout=config.session.stop_timeout,
            milestone_interval=config.webhook.milestone_interval,
        )

        self.webhook_sender = WebhookSender(
            webhook_url=config.webhook.url,
            max_retries=config.webhook.max_retries,
        )

        # Processing state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Metrics
        self.frames_processed = 0
        self.current_fps = 0.0
        self._fps_samples: deque[float] = deque(maxlen=30)
        self._last_fps_time = time.time()

        # Store last frame for debug stream
        self.last_frame = None

        # Track last jump count to compute batch sizes
        self._last_jump_count = 0

    def start(self) -> None:
        """Start the video processor."""
        if self._running:
            return

        self._running = True
        self.capture.start()

        # Start processing thread
        self._thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._thread.start()

        logger.info("VideoProcessor started")

    async def stop(self) -> None:
        """Stop the video processor."""
        self._running = False
        self.capture.stop()

        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        self.pose_detector.close()
        await self.webhook_sender.close()

        logger.info("VideoProcessor stopped")

    def _processing_loop(self) -> None:
        """Main processing loop running in separate thread."""
        # Create event loop for async webhook calls
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)

        last_timeout_check = time.time()
        timeout_check_interval = 1.0  # Check every second

        try:
            while self._running:
                # Get frame from capture
                frame = self.capture.get_frame(timeout=0.5)

                if frame is not None:
                    self._process_frame(frame)

                # Periodic timeout check
                now = time.time()
                if now - last_timeout_check > timeout_check_interval:
                    self._check_session_timeout()
                    last_timeout_check = now

        except Exception as e:
            logger.error(f"Processing loop error: {e}")
        finally:
            self._event_loop.close()

    def _process_frame(self, frame) -> None:
        """Process a single frame."""
        self.frames_processed += 1
        self._update_fps()

        # Detect pose FIRST
        body_position = self.pose_detector.process_frame(frame)

        # Store frame AFTER pose detection so frame and results are in sync
        self.last_frame = frame.copy()

        if body_position is None:
            return

        # Debug: log position every 50 frames
        if self.frames_processed % 50 == 0:
            jd = self.jump_detector
            logger.info(
                f"[DEBUG] frame={self.frames_processed}, y={body_position.y:.3f}, "
                f"jumps={jd.session_jumps}"
            )

        # Detect jump
        jump_event = self.jump_detector.process(body_position)

        if jump_event is None:
            return

        # Calculate batch size (usually 1, but 3+ at rhythm confirmation)
        batch_size = jump_event.session_count - self._last_jump_count
        self._last_jump_count = jump_event.session_count

        # Record jump(s) in session manager
        session_event = self.session_manager.record_jump(count=batch_size)

        if session_event is not None:
            self._handle_session_event(session_event)

    def _check_session_timeout(self) -> None:
        """Check for session timeout."""
        event = self.session_manager.check_timeout()
        if event is not None:
            self._handle_session_event(event)

    def _handle_session_event(self, event: SessionEvent) -> None:
        """Handle a session event (send webhook if needed)."""
        # Run async webhook in the event loop
        if self._event_loop:
            try:
                self._event_loop.run_until_complete(
                    self.webhook_sender.send_event(event)
                )
            except Exception as e:
                logger.error(f"Webhook error: {e}")

    def _update_fps(self) -> None:
        """Update FPS calculation."""
        now = time.time()
        elapsed = now - self._last_fps_time

        if elapsed > 0:
            self._fps_samples.append(1.0 / elapsed)
            self.current_fps = sum(self._fps_samples) / len(self._fps_samples)

        self._last_fps_time = now

    def reset_counters(self) -> None:
        """Reset jump count tracking."""
        self._last_jump_count = 0
