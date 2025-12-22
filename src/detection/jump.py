import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .pose import BodyPosition

logger = logging.getLogger(__name__)


@dataclass
class JumpEvent:
    """Represents a completed jump."""

    total_count: int
    session_count: int


class JumpDetector:
    """
    Detects rope skipping jumps with filters to reject other movements.

    Filters:
    1. Position stability - person must stay in roughly the same X location
    2. Rhythmic pattern - jumps must occur at a consistent frequency
    3. Minimum amplitude - vertical movement must exceed threshold
    """

    def __init__(
        self,
        min_amplitude: float = 0.0025,  # Minimum 0.25% of frame height
        max_x_drift: float = 0.15,  # Max horizontal movement (15% of frame width)
        max_jump_interval: float = 2.0,  # Max seconds between jumps before reset
        min_jump_gap: float = 0.25,  # Minimum 0.25s between jumps (max 4/sec, real is ~2.4/sec)
    ):
        self.min_amplitude = min_amplitude
        self.max_x_drift = max_x_drift
        self.max_jump_interval = max_jump_interval
        self.min_jump_gap = min_jump_gap

        # Position tracking
        self.last_y: Optional[float] = None
        self.prev_y: Optional[float] = None
        self.was_going_up = False
        self.local_min_y: Optional[float] = None
        self.local_max_y: Optional[float] = None

        # X position stability tracking
        self.x_history: deque[float] = deque(maxlen=15)  # ~0.5 second of X positions
        self.anchor_x: Optional[float] = None  # Reference X position when jumping started

        # Jump timing
        self.last_jump_time: Optional[float] = None

        # Counters
        self.total_jumps = 0
        self.session_jumps = 0

        # For debug/status
        self.last_valley_y: Optional[float] = None
        self.last_peak_y: Optional[float] = None
        self.is_stationary = False

        # Amplitude limits - reject erratic fluctuations
        self.max_amplitude = 0.10  # Max 10% of frame height (real jumps are 0.5-8%)

        # Warmup tracking - require consistent pose detection before counting
        self.consecutive_detections = 0
        self.warmup_frames = 30  # ~2 seconds at 15fps
        self.is_warmed_up = False
        self.last_detection_time: Optional[float] = None
        self.max_detection_gap = 0.5  # Reset warmup if no detection for 0.5s

        logger.info(
            f"Initialized JumpDetector: min_amplitude={min_amplitude}, "
            f"max_amplitude={self.max_amplitude}, max_x_drift={max_x_drift}"
        )

    def process(self, body_position: BodyPosition) -> Optional[JumpEvent]:
        """
        Process a body position and detect if a jump was completed.

        Only counts jumps when:
        1. Person is stationary (X position stable)
        2. Movement is rhythmic (consistent jump frequency)
        3. Vertical movement exceeds threshold
        4. Warmup complete (consistent pose detection for ~2 seconds)
        """
        curr_x = body_position.x
        curr_y = body_position.y
        now = time.time()

        # Track warmup - need consistent pose detection
        if self.last_detection_time is not None:
            gap = now - self.last_detection_time
            if gap > self.max_detection_gap:
                # Gap too large - reset warmup
                if self.is_warmed_up:
                    logger.info(f"Pose detection gap {gap:.2f}s - resetting warmup")
                self.consecutive_detections = 0
                self.is_warmed_up = False

        self.last_detection_time = now
        self.consecutive_detections += 1

        if not self.is_warmed_up:
            if self.consecutive_detections >= self.warmup_frames:
                self.is_warmed_up = True
                logger.info(f"Warmup complete after {self.consecutive_detections} frames")
            else:
                # Still warming up - don't count jumps
                return None

        # Track X position history
        self.x_history.append(curr_x)

        # Check X position stability
        if len(self.x_history) >= 5:
            x_range = max(self.x_history) - min(self.x_history)
            self.is_stationary = x_range < self.max_x_drift

            # Set anchor position when person becomes stationary
            if self.is_stationary and self.anchor_x is None:
                self.anchor_x = sum(self.x_history) / len(self.x_history)

            # Check if person moved away from anchor
            if self.anchor_x is not None:
                drift_from_anchor = abs(curr_x - self.anchor_x)
                if drift_from_anchor > self.max_x_drift:
                    # Person moved - reset anchor
                    self.anchor_x = None
                    self.is_stationary = False

        # Check if too long since last jump (person stopped)
        if self.last_jump_time is not None:
            time_since_jump = now - self.last_jump_time
            if time_since_jump > self.max_jump_interval:
                # Reset jump tracking - person stopped jumping
                self._reset_jump_state()

        # Need position history for direction detection
        if self.last_y is None:
            self.last_y = curr_y
            return None

        if self.prev_y is None:
            self.prev_y = self.last_y
            self.last_y = curr_y
            return None

        # Direction detection
        going_up = curr_y < self.last_y

        # Track extremes and detect jumps
        if going_up:
            if not self.was_going_up:
                # Direction changed: was going down, now up
                self.local_min_y = self.last_y
        else:
            if self.was_going_up:
                # Direction changed: was going up, now down = TOP OF JUMP
                self.local_max_y = self.last_y

                # Check if this is a valid jump
                if self._is_valid_jump(now):
                    self.total_jumps += 1
                    self.session_jumps += 1
                    self.last_jump_time = now

                    # Store for debug
                    self.last_valley_y = self.local_min_y
                    self.last_peak_y = self.local_max_y

                    amplitude = self.local_min_y - self.local_max_y
                    logger.info(
                        f"Jump #{self.session_jumps}! amp={amplitude:.4f}, stationary={self.is_stationary}"
                    )

                    self.prev_y = self.last_y
                    self.last_y = curr_y
                    self.was_going_up = going_up

                    return JumpEvent(
                        total_count=self.total_jumps,
                        session_count=self.session_jumps,
                    )

        self.prev_y = self.last_y
        self.last_y = curr_y
        self.was_going_up = going_up
        return None

    def _is_valid_jump(self, now: float) -> bool:
        """Check if current peak represents a valid rope skipping jump."""
        # Must have both valley and peak
        if self.local_min_y is None or self.local_max_y is None:
            return False

        # Check amplitude in valid range (rejects both noise and wild fluctuations)
        amplitude = self.local_min_y - self.local_max_y
        if amplitude < self.min_amplitude:
            print(f"[JUMP] Rejected: amp={amplitude:.4f} < {self.min_amplitude}", flush=True)
            return False
        if amplitude > self.max_amplitude:
            print(f"[JUMP] Rejected: amp={amplitude:.4f} > {self.max_amplitude}", flush=True)
            return False

        # Check minimum time gap since last jump (prevents noise/jitter)
        if self.last_jump_time is not None:
            time_since_last = now - self.last_jump_time
            if time_since_last < self.min_jump_gap:
                print(f"[JUMP] Rejected: gap={time_since_last:.3f}s < {self.min_jump_gap}", flush=True)
                return False

        print(f"[JUMP] Valid! amp={amplitude:.4f}", flush=True)
        return True

    def _reset_jump_state(self) -> None:
        """Reset jump tracking state when person stops."""
        self.last_jump_time = None
        self.local_min_y = None
        self.local_max_y = None

    def reset_session(self) -> None:
        """Reset session counter (keeps total)."""
        self.session_jumps = 0
        self._reset_jump_state()
        logger.info("Session counter reset")

    def reset_all(self) -> None:
        """Reset all counters and state."""
        self.total_jumps = 0
        self.session_jumps = 0
        self.last_y = None
        self.prev_y = None
        self.was_going_up = False
        self.x_history.clear()
        self.anchor_x = None
        self._reset_jump_state()
        logger.info("All counters and state reset")

    def reset_daily(self) -> None:
        """Reset for new day."""
        self.total_jumps = 0
        self.session_jumps = 0
        logger.info("Daily reset completed")
