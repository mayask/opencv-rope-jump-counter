import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .pose import HipPosition

logger = logging.getLogger(__name__)


@dataclass
class JumpEvent:
    """Represents a completed jump."""

    total_count: int
    session_count: int


class JumpDetector:
    """Detects and counts jump cycles using direction change detection."""

    def __init__(self, min_amplitude: float = 0.008):
        """
        Initialize jump detector.

        Args:
            min_amplitude: Minimum Y movement to count as a jump (fraction of frame height)
        """
        self.min_amplitude = min_amplitude

        # Track last few positions for direction detection (no smoothing)
        self.last_y: Optional[float] = None
        self.prev_y: Optional[float] = None

        # Direction tracking
        self.was_going_up = False
        self.local_min_y: Optional[float] = None  # Valley (highest Y value = lowest position)
        self.local_max_y: Optional[float] = None  # Peak (lowest Y value = highest position)

        # Counters
        self.total_jumps = 0
        self.session_jumps = 0

        # For debug
        self.last_valley_y: Optional[float] = None
        self.last_peak_y: Optional[float] = None

        logger.info(f"Initialized JumpDetector: min_amplitude={min_amplitude}")

    def process(self, hip_position: HipPosition) -> Optional[JumpEvent]:
        """
        Process a hip position and detect if a jump was completed.

        Detects jumps by finding direction reversals (up->down = peak = jump counted).
        """
        curr_y = hip_position.y

        # Need at least 2 previous values
        if self.last_y is None:
            self.last_y = curr_y
            return None

        if self.prev_y is None:
            self.prev_y = self.last_y
            self.last_y = curr_y
            return None

        # Determine current direction (lower Y = going up in frame)
        going_up = curr_y < self.last_y

        # Track local extremes
        if going_up:
            # Going up - track the valley we just left (if we were going down)
            if not self.was_going_up:
                # Direction changed: was down, now up
                # The last_y was a local minimum (valley)
                self.local_min_y = self.last_y
        else:
            # Going down - track the peak we just left (if we were going up)
            if self.was_going_up:
                # Direction changed: was up, now down
                # The last_y was a local maximum (peak) = TOP OF JUMP
                self.local_max_y = self.last_y

                # Check if this completes a valid jump cycle
                if self.local_min_y is not None:
                    amplitude = self.local_min_y - self.local_max_y

                    if amplitude >= self.min_amplitude:
                        self.total_jumps += 1
                        self.session_jumps += 1

                        # Store for debug
                        self.last_valley_y = self.local_min_y
                        self.last_peak_y = self.local_max_y

                        logger.info(
                            f"Jump #{self.session_jumps}! amplitude={amplitude:.4f}, "
                            f"peak_y={self.local_max_y:.3f}, valley_y={self.local_min_y:.3f}"
                        )

                        # Update state
                        self.prev_y = self.last_y
                        self.last_y = curr_y
                        self.was_going_up = going_up

                        return JumpEvent(
                            total_count=self.total_jumps,
                            session_count=self.session_jumps,
                        )

        # Update state
        self.prev_y = self.last_y
        self.last_y = curr_y
        self.was_going_up = going_up

        return None

    def reset_session(self) -> None:
        """Reset session counter (keeps total)."""
        self.session_jumps = 0
        logger.info("Session counter reset")

    def reset_all(self) -> None:
        """Reset all counters and state."""
        self.total_jumps = 0
        self.session_jumps = 0
        self.last_y = None
        self.prev_y = None
        self.was_going_up = False
        self.local_min_y = None
        self.local_max_y = None
        logger.info("All counters and state reset")

    def reset_daily(self) -> None:
        """Reset for new day (clears total but keeps calibration)."""
        self.total_jumps = 0
        self.session_jumps = 0
        logger.info("Daily reset completed")
