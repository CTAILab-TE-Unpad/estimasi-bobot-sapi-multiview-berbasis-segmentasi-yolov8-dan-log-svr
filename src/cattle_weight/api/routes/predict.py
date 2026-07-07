"""POST /api/v1/predict — cattle weight estimation endpoint.

This module is the outermost layer of the application. Its only jobs are:

1. Validate and decode the uploaded images.
2. Orchestrate calls to the core pipeline functions.
3. Map domain exceptions to appropriate HTTP error responses.
4. Assemble and return the typed :class:`~cattle_weight.api.schemas.prediction.PredictionResponse`.

No business logic should live here — keep it in the ``core`` sub-package.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from cattle_weight.api.dependencies import get_registry
from cattle_weight.api.schemas.prediction import (
    CalibrationInfo,
    PhysicalMeasurements,
    PredictionResponse,
)
from cattle_weight.core.morphometry import (
    bridge_scale,
    process_back_view,
    process_side_view,
    ramanujan_girth,
)
from cattle_weight.core.predictor import build_features, predict_weight
from cattle_weight.core.visualization import generate_result_visualization
from cattle_weight.exceptions import CalibrationError, PredictionError, SegmentationError
from cattle_weight.infrastructure.model_registry import ModelRegistry

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_image(raw_bytes: bytes, field_name: str) -> np.ndarray:
    """Decode image bytes to a BGR numpy array.

    Args:
        raw_bytes: Raw bytes from the uploaded file.
        field_name: Field name used in the error message for clarity.

    Returns:
        BGR image array (H×W×3, uint8).

    Raises:
        HTTPException 400: If the bytes cannot be decoded as an image.
    """
    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not decode '{field_name}' as an image. "
                "Please upload a valid JPEG or PNG file."
            ),
        )
    return img


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Estimate cattle live weight",
    description=(
        "Upload a **side (lateral)** view and a **back (posterior)** view image "
        "of a cattle. Both images must contain a visible calibration sticker "
        "(2.5 cm diameter). Returns predicted weight, physical measurements, "
        "and an annotated visualization."
    ),
    responses={
        400: {"description": "Image decoding failed"},
        422: {"description": "Segmentation or calibration failed"},
        500: {"description": "Internal prediction error"},
    },
)
async def predict(
    side_image: UploadFile = File(..., description="Side (lateral) view — JPEG or PNG"),
    back_image: UploadFile = File(..., description="Back (posterior) view — JPEG or PNG"),
    registry: ModelRegistry = Depends(get_registry),
) -> PredictionResponse:
    """Inference pipeline: segment → calibrate → measure → predict → visualize."""

    # 1. Decode uploaded images
    side_img = _decode_image(await side_image.read(), "side_image")
    back_img = _decode_image(await back_image.read(), "back_image")
    logger.info(
        "Predict request received — side: '%s' (%dx%d), back: '%s' (%dx%d)",
        side_image.filename, side_img.shape[1], side_img.shape[0],
        back_image.filename, back_img.shape[1], back_img.shape[0],
    )

    # 2. Segmentation + landmark extraction
    try:
        side_res = process_side_view(side_img, registry.seg_model, registry.sticker_model)
        back_res = process_back_view(back_img, registry.seg_model, registry.sticker_model)
    except SegmentationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # 3. Scale calibration (handles missing sticker via bridge)
    try:
        scale_side, scale_back, calib_source = bridge_scale(
            side_res["scale"],
            back_res["scale"],
            side_res["withers_height_px"],
            back_res["withers_height_px"],
        )
    except CalibrationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # 4. Convert pixel landmarks → real-world measurements (cm)
    body_length_cm = round(scale_side * side_res["body_length_px"], 2)
    withers_height_cm = round(scale_side * side_res["withers_height_px"], 2)
    b_cm = round(scale_side * side_res["b_px"], 2)
    a_cm = round(scale_back * back_res["a_px"], 2)
    chest_girth_cm = round(ramanujan_girth(a_cm, b_cm), 2)

    logger.info(
        "Measurements — BL: %.1f cm, WH: %.1f cm, CG: %.1f cm, 2a: %.1f cm, 2b: %.1f cm",
        body_length_cm, withers_height_cm, chest_girth_cm, 2 * a_cm, 2 * b_cm,
    )

    # 5. Feature engineering + SVR prediction
    try:
        features = build_features(body_length_cm, withers_height_cm, chest_girth_cm)
        weight_kg = predict_weight(features, registry.svr_pipe, registry.meta["feature_columns"])
    except PredictionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # 6. Generate annotated visualization
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
