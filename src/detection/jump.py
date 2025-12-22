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
    """Detects and counts completed jump cycles using hip position tracking."""

    def __init__(
        self,
        jump_threshold_ratio: float = 0.12,
        baseline_window: int = 60,
        min_jump_frames: int = 3,
    ):
        """
        Initialize jump detector.

        Args:
            jump_threshold_ratio: How much hip must rise relative to baseline (0.12 = 12%)
            baseline_window: Number of frames to use for baseline calculation
            min_jump_frames: Minimum frames in air to count as valid jump (debouncing)
        """
        self.jump_threshold_ratio = jump_threshold_ratio
        self.min_jump_frames = min_jump_frames

        # Baseline tracking (stores lowest/grounded positions)
        self.baseline_samples: deque[float] = deque(maxlen=baseline_window)
        self.baseline_y: Optional[float] = None

        # Jump state machine
        self.is_in_air = False
        self.air_frame_count = 0

        # Counters
        self.total_jumps = 0
        self.session_jumps = 0

        logger.info(
            f"Initialized JumpDetector: threshold={jump_threshold_ratio}, "
            f"min_frames={min_jump_frames}"
        )

    def process(self, hip_position: HipPosition) -> Optional[JumpEvent]:
        """
        Process a hip position and detect if a jump was completed.

        Args:
            hip_position: Current hip position from pose detection

        Returns:
            JumpEvent if a jump was just completed, None otherwise
        """
        hip_y = hip_position.y

        # Initialize baseline if not set
        if self.baseline_y is None:
            self.baseline_y = hip_y
            self.baseline_samples.append(hip_y)
            return None

        # Calculate jump threshold (lower Y = higher position in frame)
        threshold = self.baseline_y - self.jump_threshold_ratio

        if not self.is_in_air:
            # Currently on ground - update baseline
            self.baseline_samples.append(hip_y)
            # Baseline is the maximum Y (lowest point) in recent samples
            self.baseline_y = max(self.baseline_samples)

            # Check if jumped (Y decreased significantly = moved up)
            if hip_y < threshold:
                self.is_in_air = True
                self.air_frame_count = 1
                logger.debug(f"Jump started: y={hip_y:.3f}, threshold={threshold:.3f}")
        else:
            # Currently in air
            self.air_frame_count += 1

            # Check if landed (Y increased back to near baseline)
            if hip_y >= threshold:
                self.is_in_air = False

                # Only count if was in air long enough (debouncing)
                if self.air_frame_count >= self.min_jump_frames:
                    self.total_jumps += 1
                    self.session_jumps += 1
                    logger.debug(
                        f"Jump completed! Total={self.total_jumps}, "
                        f"Session={self.session_jumps}, air_frames={self.air_frame_count}"
                    )
                    return JumpEvent(
                        total_count=self.total_jumps,
                        session_count=self.session_jumps,
                    )
                else:
                    logger.debug(
                        f"Jump rejected (too short): {self.air_frame_count} frames"
                    )

        return None

    def reset_session(self) -> None:
        """Reset session counter (keeps total)."""
        self.session_jumps = 0
        logger.info("Session counter reset")

    def reset_all(self) -> None:
        """Reset all counters and state."""
        self.total_jumps = 0
        self.session_jumps = 0
        self.baseline_y = None
        self.baseline_samples.clear()
        self.is_in_air = False
        self.air_frame_count = 0
        logger.info("All counters and state reset")

    def reset_daily(self) -> None:
        """Reset for new day (clears total but keeps calibration)."""
        self.total_jumps = 0
        self.session_jumps = 0
        logger.info("Daily reset completed")
