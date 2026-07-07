"""FastAPI application factory.

Using ``create_app()`` instead of a module-level global ``app = FastAPI()``
provides several advantages:

- **Testability**: tests can call ``create_app(test_settings)`` with custom
  config without triggering production side-effects on import.
- **Configurability**: different environments (dev, staging, prod) can pass
  different ``Settings`` objects.
- **Clean lifespan**: model loading is tied to the specific ``app`` instance,
  not to module import time.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from cattle_weight.api.routes.predict import router as predict_router
from cattle_weight.exceptions import CalibrationError, PredictionError, SegmentationError
from cattle_weight.infrastructure.logging import get_logger, setup_logging
from cattle_weight.infrastructure.model_registry import ModelRegistry
from cattle_weight.settings import Settings, get_settings

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Args:
        settings: Optional custom :class:`~cattle_weight.settings.Settings`.
            Defaults to environment-based settings from :func:`~cattle_weight.settings.get_settings`.

    Returns:
        A fully configured :class:`fastapi.FastAPI` instance ready to serve.
    """
    cfg = settings or get_settings()
    setup_logging(cfg.log_level)

    registry = ModelRegistry()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Load models on startup; log on shutdown."""
        registry.load(cfg)
        app.state.registry = registry
        logger.info("Application startup complete. Serving on %s:%d", cfg.api_host, cfg.api_port)
        yield
        logger.info("Application shutting down.")

    app = FastAPI(
        title="Cattle Weight Estimation API",
        description=(
            "Microservice for estimating cattle live weight from **side** and **back** images "
            "using YOLOv8 instance segmentation and a Log-space SVR regression model.\n\n"
            "**Requirements**: Both uploaded images must contain a 2.5 cm reference sticker "
            "for pixel-to-centimetre scale calibration."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # -----------------------------------------------------------------------
    # Global exception handlers
    # Domain exceptions raised by core logic are caught here and mapped to
    # consistent JSON error responses with machine-readable error codes.
    # -----------------------------------------------------------------------

    @app.exception_handler(SegmentationError)
    async def _handle_segmentation_error(request: Request, exc: SegmentationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "SEGMENTATION_FAILED", "message": str(exc)},
        )

    @app.exception_handler(CalibrationError)
    async def _handle_calibration_error(request: Request, exc: CalibrationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "CALIBRATION_FAILED", "message": str(exc)},
        )

    @app.exception_handler(PredictionError)
    async def _handle_prediction_error(request: Request, exc: PredictionError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": "PREDICTION_FAILED", "message": str(exc)},
        )

    # -----------------------------------------------------------------------
    # Routers
    # -----------------------------------------------------------------------
    app.include_router(predict_router, prefix="/api/v1", tags=["Inference"])

    # -----------------------------------------------------------------------
    # System endpoints
    # -----------------------------------------------------------------------

    @app.get("/health", tags=["System"], summary="Health check")
    async def health() -> dict[str, str]:
        """Returns ``{"status": "ok"}`` when the service is running."""
        return {"status": "ok"}

    @app.get("/ready", tags=["System"], summary="Readiness check")
    async def ready() -> dict[str, object]:
        """Returns model loading status. Use for Kubernetes readiness probes."""
        return {
            "status": "ready" if registry.is_loaded else "not_ready",
            "models_loaded": registry.is_loaded,
        }

    return app
