import logging
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RTSPCapture:
    """Threaded RTSP stream capture with automatic reconnection."""

    def __init__(
        self,
        rtsp_url: str,
        target_fps: int = 15,
        reconnect_delay: int = 5,
        queue_size: int = 2,
    ):
        self.rtsp_url = rtsp_url
        self.target_fps = target_fps
        self.reconnect_delay = reconnect_delay
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)

        self._running = False
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_frame_time: Optional[float] = None
        self._frame_interval = 1.0 / target_fps

    def start(self) -> None:
        """Start the capture thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started RTSP capture for {self.rtsp_url}")

    def stop(self) -> None:
        """Stop the capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Stopped RTSP capture")

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get the latest frame from the queue."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_connected(self) -> bool:
        """Check if stream is currently connected."""
        with self._lock:
            return self._connected

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        while self._running:
            cap = self._connect()
            if cap is None:
                time.sleep(self.reconnect_delay)
                continue

            try:
                self._read_frames(cap)
            except Exception as e:
                logger.error(f"Error reading frames: {e}")
            finally:
                cap.release()
                with self._lock:
                    self._connected = False

            if self._running:
                logger.info(f"Reconnecting in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)

    def _connect(self) -> Optional[cv2.VideoCapture]:
        """Attempt to connect to the RTSP stream."""
        logger.info(f"Connecting to {self.rtsp_url}...")

        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        # Reduce buffer to minimize latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            logger.error("Failed to open RTSP stream")
            return None

        with self._lock:
            self._connected = True

        logger.info("Successfully connected to RTSP stream")
        return cap

    def _read_frames(self, cap: cv2.VideoCapture) -> None:
        """Read frames from the capture device."""
        while self._running and cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                logger.warning("Failed to read frame")
                break

            # Non-blocking put, drop old frames if queue is full
            try:
                # Clear old frames to keep only latest
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # Drop frame if queue is full
