from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import joblib

if TYPE_CHECKING:
    from src.utils.settings import Settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self) -> None:
        self._seg_model: Any = None
        self._sticker_model: Any = None
        self._svr_pipe: Any = None
        self._meta: dict[str, Any] | None = None
        self._loaded: bool = False

    def load(self, settings: Settings) -> None:
        from ultralytics import YOLO  # noqa: PLC0415

        logger.info("Loading ML models...")
        try:
            self._seg_model = YOLO(str(settings.yolo_seg_model_path))
            self._sticker_model = YOLO(str(settings.yolo_sticker_model_path))
            self._svr_pipe = joblib.load(settings.svr_model_path)
            self._meta = joblib.load(settings.model_metadata_path)
            self._loaded = True
            logger.info("All models loaded.")
        except Exception as exc:
            logger.error("Model loading failed: %s", exc)
            raise RuntimeError(f"ModelRegistry.load() failed: {exc}") from exc

    def _assert_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call registry.load(settings) first.")

    @property
    def seg_model(self) -> Any:
        self._assert_loaded()
        return self._seg_model

    @property
    def sticker_model(self) -> Any:
        self._assert_loaded()
        return self._sticker_model

    @property
    def svr_pipe(self) -> Any:
        self._assert_loaded()
        return self._svr_pipe

    @property
    def meta(self) -> dict[str, Any]:
        self._assert_loaded()
        return self._meta  # type: ignore[return-value]

    @property
    def is_loaded(self) -> bool:
        return self._loaded
