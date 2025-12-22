import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class StreamConfig(BaseModel):
    rtsp_url: str = Field(default="")
    target_fps: int = Field(default=15)
    reconnect_delay: int = Field(default=5)


class DetectionConfig(BaseModel):
    model_complexity: int = Field(default=0)
    min_detection_confidence: float = Field(default=0.5)
    min_tracking_confidence: float = Field(default=0.5)
    jump_threshold_ratio: float = Field(default=0.12)
    min_jump_frames: int = Field(default=3)


class SessionConfig(BaseModel):
    start_threshold: int = Field(default=5)
    stop_timeout: int = Field(default=30)


class WebhookConfig(BaseModel):
    url: str = Field(default="")
    milestone_interval: int = Field(default=100)
    max_retries: int = Field(default=3)


class APIConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)


class AppConfig(BaseSettings):
    stream: StreamConfig = Field(default_factory=StreamConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "AppConfig":
        """Load config from YAML file and override with environment variables."""
        config_data = {}

        # Load from YAML if exists
        if config_path is None:
            config_path = os.environ.get("CONFIG_PATH", "config.yaml")

        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                config_data = yaml.safe_load(f) or {}

        # Create config instance
        config = cls(**config_data)

        # Override with environment variables
        if rtsp_url := os.environ.get("RTSP_URL"):
            config.stream.rtsp_url = rtsp_url

        if webhook_url := os.environ.get("WEBHOOK_URL"):
            config.webhook.url = webhook_url

        if milestone := os.environ.get("MILESTONE_INTERVAL"):
            config.webhook.milestone_interval = int(milestone)

        return config


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = AppConfig.load()
    return _config
