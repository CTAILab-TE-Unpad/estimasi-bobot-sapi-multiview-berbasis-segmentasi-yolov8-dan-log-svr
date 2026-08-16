from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    models_dir: Path = Field(default=Path("models"))
    yolo_seg_model: str = Field(default="yolov8l-seg.pt")
    yolo_sticker_model: str = Field(default="best_sticker.pt")
    yolo_sticker_circle_model: str = Field(default="best_sticker_circle.pt")
    svr_model: str = Field(default="svr_log_pipeline.joblib")
    model_metadata: str = Field(default="model_metadata.joblib")

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_prefix="CATTLE_",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def yolo_seg_model_path(self) -> Path:
        return self.models_dir / self.yolo_seg_model

    @property
    def yolo_sticker_model_path(self) -> Path:
        return self.models_dir / self.yolo_sticker_model

    @property
    def yolo_sticker_circle_model_path(self) -> Path:
        return self.models_dir / self.yolo_sticker_circle_model

    @property
    def svr_model_path(self) -> Path:
        return self.models_dir / self.svr_model

    @property
    def model_metadata_path(self) -> Path:
        return self.models_dir / self.model_metadata


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Domain constants
AREA_RATIO_MIN: float = 0.02
AREA_RATIO_MAX: float = 0.90
STICKER_TARGET_CM: float = 2.5
STICKER_CONF_THRESHOLD: float = 0.25
CHEST_X_RATIO: float = 0.25
CHEST_DEPTH_TOP_FRACTION: float = 0.55
