import logging
import os
import sys

import uvicorn

from .config.settings import get_config

# Configure logging with both console and file output
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Add file handler if LOG_FILE is set or default to /app/logs/app.log
log_file = os.environ.get("LOG_FILE", "/app/logs/app.log")
log_dir = os.path.dirname(log_file)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(log_format))

file_handler = None
try:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))
except Exception:
    pass

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(console_handler)
if file_handler:
    root_logger.addHandler(file_handler)

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
        log_config=None,  # Use our pre-configured loggers
    )


if __name__ == "__main__":
    main()
