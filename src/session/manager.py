import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from .events import SessionEvent, SessionEventType

logger = logging.getLogger(__name__)


@dataclass
class SessionStatus:
    """Current session status."""

    active: bool
    session_jumps: int
    daily_total: int
    start_time: Optional[datetime]
    duration_seconds: Optional[float]
    last_milestone: int


class SessionManager:
    """Manages jumping sessions with auto-detection and daily reset."""

    def __init__(
        self,
        start_threshold: int = 5,
        stop_timeout: float = 30.0,
        milestone_interval: int = 100,
    ):
        """
        Initialize session manager.

        Args:
            start_threshold: Number of jumps to trigger session start
            stop_timeout: Seconds of inactivity to end session
            milestone_interval: Send notification every N jumps
        """
        self.start_threshold = start_threshold
        self.stop_timeout = stop_timeout
        self.milestone_interval = milestone_interval

        # Session state
        self.session_active = False
        self.session_start_time: Optional[datetime] = None
        self.session_jumps = 0
        self.pending_jumps = 0  # Jumps before session officially starts

        # Timing
        self.last_jump_time: Optional[datetime] = None

        # Daily tracking
        self.daily_total = 0
        self.last_reset_date: Optional[date] = None

        # Milestone tracking
        self.last_milestone = 0

        logger.info(
            f"Initialized SessionManager: start_threshold={start_threshold}, "
            f"stop_timeout={stop_timeout}s, milestone={milestone_interval}"
        )

    def check_daily_reset(self) -> Optional[SessionEvent]:
        """Check and perform daily reset if needed."""
        today = date.today()
        if self.last_reset_date != today:
            old_total = self.daily_total
            self.daily_total = 0
            self.last_reset_date = today
            self.last_milestone = 0

            if old_total > 0:
                logger.info(f"Daily reset: cleared {old_total} jumps")
                return SessionEvent(
                    event_type=SessionEventType.DAILY_RESET,
                    timestamp=datetime.now(),
                    session_jumps=0,
                    daily_total=0,
                )
        return None

    def record_jump(self, count: int = 1) -> Optional[SessionEvent]:
        """
        Record jump(s) and return event if milestone reached.

        Args:
            count: Number of jumps to record (usually 1, but can be more
                   when jump detector confirms rhythm with batch)

        Returns:
            SessionEvent if milestone reached or session started, None otherwise
        """
        self.check_daily_reset()
        now = datetime.now()
        self.last_jump_time = now

        if not self.session_active:
            self.pending_jumps += count

            if self.pending_jumps >= self.start_threshold:
                # Start session
                self.session_active = True
                self.session_start_time = now
                self.session_jumps = self.pending_jumps
                self.daily_total += self.pending_jumps
                self.pending_jumps = 0

                logger.info(f"Session started with {self.session_jumps} jumps")
                return SessionEvent(
                    event_type=SessionEventType.SESSION_STARTED,
                    timestamp=now,
                    session_jumps=self.session_jumps,
                    daily_total=self.daily_total,
                )
        else:
            self.session_jumps += count
            self.daily_total += count

            # Check milestone
            current_milestone = (
                self.session_jumps // self.milestone_interval
            ) * self.milestone_interval

            if current_milestone > self.last_milestone and current_milestone > 0:
                self.last_milestone = current_milestone
                logger.info(f"Milestone reached: {current_milestone}")
                return SessionEvent(
                    event_type=SessionEventType.MILESTONE_REACHED,
                    timestamp=now,
                    session_jumps=self.session_jumps,
                    daily_total=self.daily_total,
                    milestone=current_milestone,
                )

        return None

    def check_timeout(self) -> Optional[SessionEvent]:
        """
        Check if session should end due to inactivity.

        Returns:
            SessionEvent if session ended, None otherwise
        """
        if not self.session_active:
            return None

        now = datetime.now()
        if self.last_jump_time:
            elapsed = (now - self.last_jump_time).total_seconds()
            if elapsed > self.stop_timeout:
                return self._end_session("timeout")

        return None

    def force_start(self) -> SessionEvent:
        """Manually start a session."""
        now = datetime.now()
        self.session_active = True
        self.session_start_time = now
        self.session_jumps = self.pending_jumps
        self.daily_total += self.pending_jumps
        self.pending_jumps = 0
        self.last_jump_time = now

        logger.info("Session manually started")
        return SessionEvent(
            event_type=SessionEventType.SESSION_STARTED,
            timestamp=now,
            session_jumps=self.session_jumps,
            daily_total=self.daily_total,
        )

    def force_stop(self) -> Optional[SessionEvent]:
        """Manually stop the session."""
        if not self.session_active:
            return None
        return self._end_session("manual")

    def _end_session(self, reason: str) -> SessionEvent:
        """End the current session."""
        now = datetime.now()
        duration = None
        if self.session_start_time:
            duration = (now - self.session_start_time).total_seconds()

        event = SessionEvent(
            event_type=SessionEventType.SESSION_ENDED,
            timestamp=now,
            session_jumps=self.session_jumps,
            daily_total=self.daily_total,
            session_duration_seconds=duration,
        )

        logger.info(
            f"Session ended ({reason}): {self.session_jumps} jumps, "
            f"duration={duration:.1f}s" if duration else ""
        )

        # Reset session state
        self.session_active = False
        self.session_start_time = None
        self.session_jumps = 0
        self.pending_jumps = 0
        self.last_milestone = 0

        return event

    def reset_all(self) -> None:
        """Reset all state."""
        self.session_active = False
        self.session_start_time = None
        self.session_jumps = 0
        self.pending_jumps = 0
        self.daily_total = 0
        self.last_milestone = 0
        self.last_jump_time = None
        logger.info("Session manager reset")

    def get_status(self) -> SessionStatus:
        """Get current session status."""
        duration = None
        if self.session_active and self.session_start_time:
            duration = (datetime.now() - self.session_start_time).total_seconds()

        return SessionStatus(
            active=self.session_active,
            session_jumps=self.session_jumps,
            daily_total=self.daily_total,
            start_time=self.session_start_time,
            duration_seconds=duration,
            last_milestone=self.last_milestone,
        )
