from __future__ import annotations

import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import cv2
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from src.extractor.morphometry import (
    bridge_scale,
    process_back_view,
    process_side_view,
    ramanujan_girth,
)
from src.extractor.predictor import build_features, predict_weight
from src.utils.exceptions import CalibrationError, PredictionError, SegmentationError
from src.utils.model_registry import ModelRegistry
from src.utils.schema import CalibrationInfo, PhysicalMeasurements, PredictionResponse
from src.utils.settings import get_settings

cfg = get_settings()

logger = logging.getLogger(__name__)

# Thread pool shared across all requests. CPU-bound YOLO/SVR work runs here so
# the asyncio event loop is never blocked by heavy computation.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="inference")


def _setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    root = logging.getLogger()
    root.setLevel(numeric_level)
    if not root.handlers:
        root.addHandler(handler)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)


_setup_logging(cfg.log_level)

_registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _registry.load(cfg)
    app.state.registry = _registry
    logger.info("Startup complete. Serving on %s:%d", cfg.api_host, cfg.api_port)
    yield
    _executor.shutdown(wait=False)
    logger.info("Shutting down.")


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
)


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


@app.exception_handler(SegmentationError)
async def _handle_segmentation_error(request: Request, exc: SegmentationError) -> JSONResponse:
    return JSONResponse(
        status_code=422, content={"error": "SEGMENTATION_FAILED", "message": str(exc)}
    )


@app.exception_handler(CalibrationError)
async def _handle_calibration_error(request: Request, exc: CalibrationError) -> JSONResponse:
    return JSONResponse(
        status_code=422, content={"error": "CALIBRATION_FAILED", "message": str(exc)}
    )


@app.exception_handler(PredictionError)
async def _handle_prediction_error(request: Request, exc: PredictionError) -> JSONResponse:
    return JSONResponse(
        status_code=500, content={"error": "PREDICTION_FAILED", "message": str(exc)}
    )


@app.get("/api/health", tags=["System"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/ready", tags=["System"])
async def ready() -> dict[str, object]:
    return {
        "status": "ready" if _registry.is_loaded else "not_ready",
        "models_loaded": _registry.is_loaded,
    }


def _decode_image(raw_bytes: bytes, field_name: str) -> np.ndarray:
    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            f"Could not decode '{field_name}'. Please upload a valid JPEG or PNG file."
        )
    return img


def _run_prediction_pipeline(
    side_bytes: bytes,
    back_bytes: bytes,
    registry: ModelRegistry,
    side_filename: str,
    back_filename: str,
    side_sticker_cm: float = 10.16,
    back_sticker_cm: float = 10.16,
    sticker_shape: str = "square",
) -> PredictionResponse:
    """
    Synchronous prediction pipeline — executed inside the thread pool so the
    asyncio event loop stays unblocked.

    Side and back inference run in parallel using a nested thread pool so that
    both YOLO calls (segmentation + sticker) for each view happen concurrently.
    """
    # --- Decode images ---
    side_img = _decode_image(side_bytes, "side_image")
    back_img = _decode_image(back_bytes, "back_image")

    logger.info(
        "Predict request — side: '%s' (%dx%d), back: '%s' (%dx%d)",
        side_filename,
        side_img.shape[1],
        side_img.shape[0],
        back_filename,
        back_img.shape[1],
        back_img.shape[0],
    )

    shape_mode = "square" if sticker_shape.lower().startswith("sq") else "circle"

    # --- Parallel segmentation: side + back processed concurrently ---
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="seg") as seg_pool:
        future_side = seg_pool.submit(
            process_side_view,
            side_img,
            registry.seg_model,
            registry.sticker_model,
            target_cm=side_sticker_cm,
            shape=shape_mode,
        )
        future_back = seg_pool.submit(
            process_back_view,
            back_img,
            registry.seg_model,
            registry.sticker_model,
            target_cm=back_sticker_cm,
            shape=shape_mode,
        )
        side_res = future_side.result()
        back_res = future_back.result()

    # --- Scale calibration ---
    scale_side, scale_back, calib_source = bridge_scale(
        side_res["scale"],
        back_res["scale"],
        side_res["withers_height_px"],
        back_res["withers_height_px"],
    )

    # --- Physical measurements ---
    body_length_cm = round(scale_side * side_res["body_length_px"], 2)
    withers_height_cm = round(scale_side * side_res["withers_height_px"], 2)
    b_cm = round(scale_side * side_res["b_px"], 2)
    a_cm = round(scale_back * back_res["a_px"], 2)
    chest_girth_cm = round(ramanujan_girth(a_cm, b_cm), 2)

    logger.info(
        "Measurements — BL: %.1f cm, WH: %.1f cm, CG: %.1f cm",
        body_length_cm,
        withers_height_cm,
        chest_girth_cm,
    )

    # --- SVR prediction ---
    features = build_features(body_length_cm, withers_height_cm, chest_girth_cm)
    weight_kg = predict_weight(features, registry.svr_pipe, registry.meta["feature_columns"])

    return PredictionResponse(
        predicted_weight_kg=round(weight_kg, 2),
        calibration=CalibrationInfo(
            scale_source=calib_source,
            side_sticker_size_cm=side_sticker_cm,
            back_sticker_size_cm=back_sticker_cm,
            scale_side_cm_per_px=round(scale_side, 6),
            scale_back_cm_per_px=round(scale_back, 6),
        ),
        measurements=PhysicalMeasurements(
            body_length_cm=body_length_cm,
            withers_height_cm=withers_height_cm,
            chest_girth_cm=chest_girth_cm,
            chest_width_2a_cm=round(2 * a_cm, 2),
            chest_depth_2b_cm=round(2 * b_cm, 2),
        ),
        model_features={k: float(v) for k, v in features.items()},
    )


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Estimate cattle live weight",
)
async def predict(
    side_image: UploadFile = File(..., description="Side (lateral) view — JPEG or PNG"),
    back_image: UploadFile = File(..., description="Back (posterior) view — JPEG or PNG"),
    side_sticker_cm: float = Form(10.16, description="Real size of calibration sticker on side image in cm"),
    back_sticker_cm: float = Form(10.16, description="Real size of calibration sticker on back image in cm"),
    sticker_shape: str = Form("square", description="Sticker shape model: 'square' or 'circle'"),
    registry: ModelRegistry = Depends(get_registry),
) -> PredictionResponse:
    # Read image bytes in the async context (non-blocking I/O)
    side_bytes, back_bytes = await asyncio.gather(
        side_image.read(),
        back_image.read(),
    )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            _run_prediction_pipeline,
            side_bytes,
            back_bytes,
            registry,
            side_image.filename or "side_image",
            back_image.filename or "back_image",
            side_sticker_cm,
            back_sticker_cm,
            sticker_shape,
        )
    except ValueError as exc:
        # Image decode errors raised inside the thread
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SegmentationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except CalibrationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PredictionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result
