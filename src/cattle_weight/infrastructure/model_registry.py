"""Model registry — owns the full lifecycle of all ML model instances.

Replaces the previous approach of module-level global variables
(``model_seg = None``, etc.) with an encapsulated class that is:

- **Mockable**: tests can inject a :class:`MockModelRegistry` via ``app.state``
- **Fail-fast**: accessing any model before :meth:`load` raises ``RuntimeError``
- **Type-safe**: properties return typed values, not ``Optional[Any]``
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import joblib

if TYPE_CHECKING:
    from cattle_weight.settings import Settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Container for all ML models used by the API.

    Designed to be stored in ``app.state`` during the FastAPI lifespan and
    retrieved in endpoints via a ``Depends`` provider.

    Example::

        registry = ModelRegistry()
        registry.load(settings)          # called once at startup
        seg_model = registry.seg_model   # raises RuntimeError if not loaded
    """

    def __init__(self) -> None:
        self._seg_model: Any = None
        self._sticker_model: Any = None
        self._svr_pipe: Any = None
        self._meta: dict[str, Any] | None = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, settings: Settings) -> None:
        """Load all model artifacts into memory.

        Args:
            settings: Application settings providing resolved model paths.

        Raises:
            RuntimeError: If any model file cannot be loaded.
        """
        # Deferred import: keeps startup fast and avoids import-time side effects
        # (ultralytics prints version info on import)
        from ultralytics import YOLO  # noqa: PLC0415

        logger.info("Loading ML models into memory...")
        try:
            self._seg_model = YOLO(str(settings.yolo_seg_model_path))
            logger.info("  ✓ Segmentation model : %s", settings.yolo_seg_model)

            self._sticker_model = YOLO(str(settings.yolo_sticker_model_path))
            logger.info("  ✓ Sticker model      : %s", settings.yolo_sticker_model)

            self._svr_pipe = joblib.load(settings.svr_model_path)
            logger.info("  ✓ SVR pipeline       : %s", settings.svr_model)

            self._meta = joblib.load(settings.model_metadata_path)
            logger.info("  ✓ Model metadata     : %s", settings.model_metadata)

            self._loaded = True
            logger.info("All models loaded successfully.")

        except Exception as exc:
            logger.error("Model loading failed: %s", exc)
            raise RuntimeError(f"ModelRegistry.load() failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Properties (fail-fast if not loaded)
    # ------------------------------------------------------------------

    def _assert_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "ModelRegistry: models have not been loaded yet. "
                "Ensure registry.load(settings) is called during app lifespan startup."
            )

    @property
    def seg_model(self) -> Any:
        """YOLOv8 cattle segmentation model."""
        self._assert_loaded()
        return self._seg_model

    @property
    def sticker_model(self) -> Any:
        """YOLOv8 sticker detection model."""
        self._assert_loaded()
        return self._sticker_model

    @property
    def svr_pipe(self) -> Any:
        """Fitted scikit-learn SVR pipeline."""
        self._assert_loaded()
        return self._svr_pipe

    @property
    def meta(self) -> dict[str, Any]:
        """Model metadata dictionary (feature column order, etc.)."""
        self._assert_loaded()
        return self._meta  # type: ignore[return-value]

    @property
    def is_loaded(self) -> bool:
        """``True`` if :meth:`load` has completed successfully."""
        return self._loaded
