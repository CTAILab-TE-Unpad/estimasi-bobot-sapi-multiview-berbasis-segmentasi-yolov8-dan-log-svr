from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import cv2
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from extractor.morphometry import bridge_scale, process_back_view, process_side_view, ramanujan_girth
from extractor.predictor import build_features, predict_weight
from extractor.visualization import generate_result_visualization
from utils.exceptions import CalibrationError, PredictionError, SegmentationError
from utils.model_registry import ModelRegistry
from utils.schema import CalibrationInfo, PhysicalMeasurements, PredictionResponse
from utils.settings import get_settings

cfg = get_settings()

logger = logging.getLogger(__name__)


def _setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
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
    return JSONResponse(status_code=422, content={"error": "SEGMENTATION_FAILED", "message": str(exc)})


@app.exception_handler(CalibrationError)
async def _handle_calibration_error(request: Request, exc: CalibrationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "CALIBRATION_FAILED", "message": str(exc)})


@app.exception_handler(PredictionError)
async def _handle_prediction_error(request: Request, exc: PredictionError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": "PREDICTION_FAILED", "message": str(exc)})


@app.get("/health", tags=["System"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["System"])
async def ready() -> dict[str, object]:
    return {"status": "ready" if _registry.is_loaded else "not_ready", "models_loaded": _registry.is_loaded}


def _decode_image(raw_bytes: bytes, field_name: str) -> np.ndarray:
    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode '{field_name}'. Please upload a valid JPEG or PNG file.",
        )
    return img


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"], summary="Estimate cattle live weight")
async def predict(
    side_image: UploadFile = File(..., description="Side (lateral) view — JPEG or PNG"),
    back_image: UploadFile = File(..., description="Back (posterior) view — JPEG or PNG"),
    registry: ModelRegistry = Depends(get_registry),
) -> PredictionResponse:
    side_img = _decode_image(await side_image.read(), "side_image")
    back_img = _decode_image(await back_image.read(), "back_image")
    logger.info(
        "Predict request — side: '%s' (%dx%d), back: '%s' (%dx%d)",
        side_image.filename, side_img.shape[1], side_img.shape[0],
        back_image.filename, back_img.shape[1], back_img.shape[0],
    )

    try:
        side_res = process_side_view(side_img, registry.seg_model, registry.sticker_model)
        back_res = process_back_view(back_img, registry.seg_model, registry.sticker_model)
    except SegmentationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        scale_side, scale_back, calib_source = bridge_scale(
            side_res["scale"], back_res["scale"],
            side_res["withers_height_px"], back_res["withers_height_px"],
        )
    except CalibrationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    body_length_cm = round(scale_side * side_res["body_length_px"], 2)
    withers_height_cm = round(scale_side * side_res["withers_height_px"], 2)
    b_cm = round(scale_side * side_res["b_px"], 2)
    a_cm = round(scale_back * back_res["a_px"], 2)
    chest_girth_cm = round(ramanujan_girth(a_cm, b_cm), 2)

    logger.info("Measurements — BL: %.1f cm, WH: %.1f cm, CG: %.1f cm", body_length_cm, withers_height_cm, chest_girth_cm)

    try:
        features = build_features(body_length_cm, withers_height_cm, chest_girth_cm)
        weight_kg = predict_weight(features, registry.svr_pipe, registry.meta["feature_columns"])
    except PredictionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    viz_b64 = generate_result_visualization(
        side_img, side_res, back_img, back_res,
        scale_side, scale_back,
        body_length_cm, withers_height_cm, b_cm, a_cm, chest_girth_cm,
    )

    return PredictionResponse(
        predicted_weight_kg=round(weight_kg, 2),
        calibration=CalibrationInfo(
            scale_source=calib_source,
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
        visualization_png_b64=viz_b64,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host=cfg.api_host, port=cfg.api_port, reload=False, workers=1, log_level=cfg.log_level.lower())
