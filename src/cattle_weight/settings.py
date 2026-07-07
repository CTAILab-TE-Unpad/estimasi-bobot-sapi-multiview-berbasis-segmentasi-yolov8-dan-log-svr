"""Runtime settings sourced from environment variables or a ``.env`` file.

All deployment / path / server configuration lives here. Domain algorithm
constants belong in :mod:`cattle_weight.config` instead.

Environment variable prefix: ``CATTLE_``

Examples::

    CATTLE_LOG_LEVEL=DEBUG
    CATTLE_MODELS_DIR=/opt/models
    CATTLE_YOLO_SEG_MODEL=yolov8n-seg.pt   # use lighter model in staging

Usage::

    from cattle_weight.settings import get_settings

    settings = get_settings()
    model_path = settings.yolo_seg_model_path
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    All fields can be overridden by environment variables prefixed with
    ``CATTLE_`` (case-insensitive), or by values in a ``.env`` file located
    in the working directory.
    """

    # --- Model filenames (not full paths — resolved via models_dir) ---
    models_dir: Path = Field(
        default=Path("models"),
        description="Directory that contains all model artifact files.",
    )
    yolo_seg_model: str = Field(
        default="yolov8l-seg.pt",
        description="Filename of the YOLOv8 cattle segmentation model.",
    )
    yolo_sticker_model: str = Field(
        default="best_sticker.pt",
        description="Filename of the YOLOv8 sticker detection model.",
    )
    svr_model: str = Field(
        default="svr_log_pipeline.joblib",
        description="Filename of the fitted SVR pipeline (joblib).",
    )
    model_metadata: str = Field(
        default="model_metadata.joblib",
        description="Filename of the model metadata file (joblib).",
    )

    # --- Server ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CATTLE_",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Resolved absolute paths (computed properties) ---

    @property
    def yolo_seg_model_path(self) -> Path:
        """Absolute path to the YOLOv8 segmentation model."""
        return self.models_dir / self.yolo_seg_model

    @property
    def yolo_sticker_model_path(self) -> Path:
        """Absolute path to the sticker detection model."""
        return self.models_dir / self.yolo_sticker_model

    @property
    def svr_model_path(self) -> Path:
        """Absolute path to the SVR pipeline file."""
        return self.models_dir / self.svr_model

    @property
    def model_metadata_path(self) -> Path:
        """Absolute path to the model metadata file."""
        return self.models_dir / self.model_metadata


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Using ``lru_cache`` ensures the ``.env`` file is read only once per
    process lifetime, and the same object is shared everywhere.
    """
    return Settings()
