import asyncio
import logging
from typing import Optional

import aiohttp

from ..session.events import SessionEvent, SessionEventType

logger = logging.getLogger(__name__)


class WebhookSender:
    """Sends webhooks to Home Assistant on milestones."""

    def __init__(
        self,
        webhook_url: str,
        max_retries: int = 3,
        timeout: float = 5.0,
    ):
        """
        Initialize webhook sender.

        Args:
            webhook_url: Home Assistant webhook URL
            max_retries: Maximum retry attempts for failed requests
            timeout: Request timeout in seconds
        """
        self.webhook_url = webhook_url
        self.max_retries = max_retries
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(f"Initialized WebhookSender: url={webhook_url}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def send_event(self, event: SessionEvent) -> bool:
        """
        Send a session event to Home Assistant.

        Only sends MILESTONE_REACHED events.

        Args:
            event: Session event to send

        Returns:
            True if sent successfully, False otherwise
        """
        # Only send milestone events
        if event.event_type != SessionEventType.MILESTONE_REACHED:
            return True

        if not self.webhook_url:
            logger.warning("No webhook URL configured, skipping notification")
            return False

        payload = {
            "event": "rope_skip_milestone",
            "count": event.session_jumps,
            "milestone": event.milestone,
            "daily_total": event.daily_total,
            "timestamp": event.timestamp.isoformat(),
        }

        return await self._send_with_retry(payload)

    async def _send_with_retry(self, payload: dict) -> bool:
        """Send payload with retry logic."""
        session = await self._get_session()

        for attempt in range(self.max_retries):
            try:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status < 300:
                        logger.info(
                            f"Webhook sent successfully: milestone={payload.get('milestone')}"
                        )
                        return True
                    else:
                        logger.warning(
                            f"Webhook failed with status {response.status} "
                            f"(attempt {attempt + 1}/{self.max_retries})"
                        )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Webhook timeout (attempt {attempt + 1}/{self.max_retries})"
                )
            except aiohttp.ClientError as e:
                logger.warning(
                    f"Webhook error: {e} (attempt {attempt + 1}/{self.max_retries})"
                )
            except Exception as e:
                logger.error(f"Unexpected webhook error: {e}")
                return False

            # Exponential backoff
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2**attempt)

        logger.error(f"Webhook failed after {self.max_retries} attempts")
        return False

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
