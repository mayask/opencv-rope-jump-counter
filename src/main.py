import logging
import sys

import uvicorn

from .config.settings import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    config = get_config()

    # Validate required config
    if not config.stream.rtsp_url:
        logger.error("RTSP_URL environment variable is required")
        sys.exit(1)

    if not config.webhook.url:
        logger.warning("WEBHOOK_URL not set - no notifications will be sent")

    logger.info(f"Starting Rope Skipping Counter on {config.api.host}:{config.api.port}")
    logger.info(f"RTSP URL: {config.stream.rtsp_url}")
    logger.info(f"Webhook URL: {config.webhook.url or 'Not configured'}")
    logger.info(f"Milestone interval: {config.webhook.milestone_interval}")

    uvicorn.run(
        "src.api.app:app",
        host=config.api.host,
        port=config.api.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
