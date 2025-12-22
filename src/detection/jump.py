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
    Detects rope skipping jumps with strict filters to reject other movements.

    Key requirements:
    1. Person must be stationary (not walking/running)
    2. Vertical oscillation must be rhythmic (consistent frequency)
    3. Amplitude must be significant (not just standing sway)
    4. Must see consistent pattern before counting starts
    """

    def __init__(
        self,
        min_amplitude: float = 0.04,  # Minimum 4% of person's body height
        max_x_drift: float = 0.15,  # Max 15% horizontal movement (allow moving around)
        max_jump_interval: float = 1.5,  # Max 1.5 seconds between jumps (slower pace OK)
        min_jump_gap: float = 0.15,  # Minimum 0.15s between jumps (max ~6/sec)
        rhythm_tolerance: float = 0.6,  # 60% tolerance on jump interval consistency
        confirmation_jumps: int = 3,  # Need 3 consistent oscillations to confirm rhythm
    ):
        self.min_amplitude = min_amplitude
        self.max_x_drift = max_x_drift
        self.max_jump_interval = max_jump_interval
        self.min_jump_gap = min_jump_gap
        self.rhythm_tolerance = rhythm_tolerance
        self.confirmation_jumps = confirmation_jumps

        # Position tracking
        self.last_y: Optional[float] = None
        self.prev_y: Optional[float] = None
        self.was_going_up = False
        self.local_min_y: Optional[float] = None
        self.local_max_y: Optional[float] = None

        # X position stability tracking
        self.x_history: deque[float] = deque(maxlen=30)  # ~2 seconds of X positions
        self.anchor_x: Optional[float] = None

        # Jump timing and rhythm detection
        self.oscillation_times: deque[float] = deque(maxlen=10)  # Recent oscillation timestamps
        self.last_jump_time: Optional[float] = None
        self.rhythm_confirmed = False
        self.pending_jumps = 0  # Jumps detected but not yet counted (during confirmation)

        # Counters
        self.total_jumps = 0
        self.session_jumps = 0

        # For debug/status
        self.last_valley_y: Optional[float] = None
        self.last_peak_y: Optional[float] = None
        self.is_stationary = False

        # Amplitude limits (relative to person's bbox height)
        self.max_amplitude = 0.15  # Max 15% of person height

        # Track person's bbox height for amplitude normalization
        self.last_bbox_height: float = 0.0

        # Detection gap tracking (no warmup - pose detector handles that)
        self.last_detection_time: Optional[float] = None
        self.max_detection_gap = 0.5

        logger.info(
            f"Initialized JumpDetector: min_amplitude={min_amplitude}, "
            f"max_amplitude={self.max_amplitude}, max_x_drift={max_x_drift}, "
            f"confirmation_jumps={confirmation_jumps}"
        )

    def process(self, body_position: BodyPosition) -> Optional[JumpEvent]:
        """
        Process a body position and detect if a jump was completed.

        Only counts jumps when:
        1. Person is stationary (X position stable)
        2. Rhythm is confirmed (consistent oscillation pattern)
        3. Vertical movement exceeds threshold
        """
        curr_x = body_position.x
        curr_y = body_position.y
        now = time.time()

        # Store bbox height for amplitude normalization
        if body_position.bbox_height > 0:
            self.last_bbox_height = body_position.bbox_height

        # Check for detection gaps - reset state if camera lost track
        if self.last_detection_time is not None:
            gap = now - self.last_detection_time
            if gap > self.max_detection_gap:
                logger.info(f"Pose detection gap {gap:.2f}s - resetting")
                self._full_reset()

        self.last_detection_time = now

        # Track X position history
        self.x_history.append(curr_x)

        # Check X position stability (must be stationary to count jumps)
        if len(self.x_history) >= 5:
            x_range = max(self.x_history) - min(self.x_history)
            was_stationary = self.is_stationary
            self.is_stationary = x_range < self.max_x_drift

            # If person starts moving, reset rhythm detection
            if was_stationary and not self.is_stationary:
                self._reset_rhythm()
                print(f"[JUMP] Person moving (x_range={x_range:.3f}) - reset rhythm", flush=True)

            # Set anchor when becoming stationary
            if self.is_stationary and self.anchor_x is None:
                self.anchor_x = sum(self.x_history) / len(self.x_history)

            # Check drift from anchor
            if self.anchor_x is not None:
                drift = abs(curr_x - self.anchor_x)
                if drift > self.max_x_drift:
                    self.anchor_x = None
                    self.is_stationary = False
                    self._reset_rhythm()

        # Must be stationary to detect jumps
        if not self.is_stationary:
            self.last_y = curr_y
            return None

        # Check if too long since last oscillation
        if self.oscillation_times and (now - self.oscillation_times[-1]) > self.max_jump_interval:
            self._reset_rhythm()

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

        # Track extremes and detect oscillations
        if going_up:
            if not self.was_going_up:
                # Direction changed: was going down, now up = BOTTOM of oscillation
                self.local_min_y = self.last_y
        else:
            if self.was_going_up:
                # Direction changed: was going up, now down = TOP of oscillation
                self.local_max_y = self.last_y

                # Check if this is a valid oscillation
                result = self._process_oscillation(now)
                if result:
                    self.prev_y = self.last_y
                    self.last_y = curr_y
                    self.was_going_up = going_up
                    return result

        self.prev_y = self.last_y
        self.last_y = curr_y
        self.was_going_up = going_up
        return None

    def _process_oscillation(self, now: float) -> Optional[JumpEvent]:
        """Process a detected oscillation peak and determine if it's a valid jump."""
        # Must have both valley and peak
        if self.local_min_y is None or self.local_max_y is None:
            return None

        # Check amplitude (normalized to person's bbox height for distance-independence)
        raw_amplitude = self.local_min_y - self.local_max_y
        if self.last_bbox_height > 0:
            amplitude = raw_amplitude / self.last_bbox_height
        else:
            amplitude = raw_amplitude  # Fallback to frame-relative if no bbox
        if amplitude < self.min_amplitude:
            print(f"[JUMP] Rejected: amp={amplitude:.4f} < {self.min_amplitude}", flush=True)
            return None
        if amplitude > self.max_amplitude:
            print(f"[JUMP] Rejected: amp={amplitude:.4f} > {self.max_amplitude}", flush=True)
            return None

        # Check minimum gap since last oscillation
        if self.oscillation_times:
            gap = now - self.oscillation_times[-1]
            if gap < self.min_jump_gap:
                print(f"[JUMP] Rejected: gap={gap:.3f}s < {self.min_jump_gap}", flush=True)
                return None

        # Record this oscillation
        self.oscillation_times.append(now)

        # Check rhythm consistency
        if not self.rhythm_confirmed:
            if len(self.oscillation_times) >= self.confirmation_jumps:
                if self._check_rhythm():
                    self.rhythm_confirmed = True
                    # Count the pending jumps now that rhythm is confirmed
                    self.pending_jumps = len(self.oscillation_times)
                    self.total_jumps += self.pending_jumps
                    self.session_jumps += self.pending_jumps
                    self.last_jump_time = now
                    logger.info(f"Rhythm confirmed! Counted {self.pending_jumps} jumps")
                    print(f"[JUMP] Rhythm confirmed! Counted {self.pending_jumps} jumps", flush=True)
                    return JumpEvent(
                        total_count=self.total_jumps,
                        session_count=self.session_jumps,
                    )
            # Not enough oscillations yet or rhythm not consistent
            print(f"[JUMP] Pending: {len(self.oscillation_times)}/{self.confirmation_jumps} oscillations", flush=True)
            return None

        # Rhythm already confirmed - count this jump
        self.total_jumps += 1
        self.session_jumps += 1
        self.last_jump_time = now
        self.last_valley_y = self.local_min_y
        self.last_peak_y = self.local_max_y

        print(f"[JUMP] Valid! #{self.session_jumps} amp={amplitude:.4f}", flush=True)
        logger.info(f"Jump #{self.session_jumps}! amp={amplitude:.4f}")

        return JumpEvent(
            total_count=self.total_jumps,
            session_count=self.session_jumps,
        )

    def _check_rhythm(self) -> bool:
        """Check if oscillations have a consistent rhythm (like rope skipping)."""
        if len(self.oscillation_times) < 3:
            return False

        # Calculate intervals between oscillations
        intervals = []
        times = list(self.oscillation_times)
        for i in range(1, len(times)):
            intervals.append(times[i] - times[i - 1])

        if not intervals:
            return False

        # Check if intervals are consistent
        avg_interval = sum(intervals) / len(intervals)

        # Rope skipping typically 1-5 jumps per second (0.2s - 1.0s interval)
        if avg_interval < 0.15 or avg_interval > 1.0:
            print(f"[JUMP] Rhythm check failed: avg_interval={avg_interval:.3f}s outside 0.15-1.0s range", flush=True)
            return False

        # Check consistency - all intervals should be within tolerance of average
        for interval in intervals:
            deviation = abs(interval - avg_interval) / avg_interval
            if deviation > self.rhythm_tolerance:
                print(f"[JUMP] Rhythm check failed: interval={interval:.3f}s deviates {deviation:.1%} from avg={avg_interval:.3f}s", flush=True)
                return False

        print(f"[JUMP] Rhythm check passed: avg_interval={avg_interval:.3f}s, {len(intervals)} consistent intervals", flush=True)
        return True

    def _reset_rhythm(self) -> None:
        """Reset rhythm detection state."""
        self.oscillation_times.clear()
        self.rhythm_confirmed = False
        self.pending_jumps = 0
        self.last_jump_time = None
        self.local_min_y = None
        self.local_max_y = None

    def _full_reset(self) -> None:
        """Full reset of all tracking state."""
        self.last_y = None
        self.prev_y = None
        self.was_going_up = False
        self.x_history.clear()
        self.anchor_x = None
        self.is_stationary = False
        self._reset_rhythm()

    def reset_session(self) -> None:
        """Reset session counter (keeps total)."""
        self.session_jumps = 0
        self._reset_rhythm()
        logger.info("Session counter reset")

    def reset_all(self) -> None:
        """Reset all counters and state."""
        self.total_jumps = 0
        self.session_jumps = 0
        self._full_reset()
        logger.info("All counters and state reset")

    def reset_daily(self) -> None:
        """Reset for new day."""
        self.total_jumps = 0
        self.session_jumps = 0
        logger.info("Daily reset completed")
