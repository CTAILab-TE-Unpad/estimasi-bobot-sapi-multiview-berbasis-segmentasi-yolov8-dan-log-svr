from __future__ import annotations

import io
import base64
import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any

import cv2
import numpy as np
import uvicorn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
        "Supports both square and circular calibration stickers with sub-pixel invariant scale extraction."
    ),
    version="1.1.0",
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


def _draw_sticker_on_axes(ax: plt.Axes, img: np.ndarray, res: dict, scale_val: float, title_prefix: str) -> None:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb)
    ax.axis("off")

    geom = res.get("sticker_geom")
    if geom is not None:
        shape_type = geom.get("shape", "square")
        if shape_type == "circle":
            xc, yc = geom["center"]
            major = geom["major_axis"]
            minor = geom["minor_axis"]
            angle = geom["angle"]

            ellipse = patches.Ellipse(
                (xc, yc), major, minor, angle=angle,
                fill=False, edgecolor="cyan", lw=3, label=f"Fit (D={major:.1f}px)"
            )
            ax.add_patch(ellipse)
            ax.plot(xc, yc, "r+", ms=10, mew=2)
            ax.set_title(f"{title_prefix} Stiker Circle\nDiameter={major:.1f}px | S={scale_val:.5f} cm/px", fontsize=11, fontweight="bold")
        else:
            x, y, w, h = geom["bbox"]
            side = geom.get("side_px", max(w, h))
            rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor="lime", lw=3, label=f"Fit (L={side:.1f}px)")
            ax.add_patch(rect)
            ax.set_title(f"{title_prefix} Stiker Square\nSide={side:.1f}px | S={scale_val:.5f} cm/px", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
    else:
        ax.set_title(f"{title_prefix} Stiker\n[Stiker Tidak Terdeteksi / Bridged]", fontsize=11, color="orange", fontweight="bold")


def _generate_visualizations(
    side_img: np.ndarray, side_res: dict,
    back_img: np.ndarray, back_res: dict,
    S_side: float, S_back: float,
    BL_cm: float, WH_cm: float, b_cm: float, a_cm: float, CG_cm: float,
) -> Dict[str, str]:
    # 1. Morfometri Sapi
    fig_morph, axes_m = plt.subplots(1, 2, figsize=(16, 7))
    side_rgb = cv2.cvtColor(side_img, cv2.COLOR_BGR2RGB)
    ov_side = side_rgb.copy()
    ov_side[side_res["cow_mask"] == 255] = [0, 255, 0]
    side_blend = cv2.addWeighted(side_rgb, 0.75, ov_side, 0.25, 0)

    x_min, x_max = side_res["x_min"], side_res["x_max"]
    y_min, y_max = side_res["y_min"], side_res["y_max"]
    chest_x = side_res["chest_x"]

    axes_m[0].imshow(side_blend)
    axes_m[0].plot([x_min + (x_max - x_min) / 2]*2, [y_min, y_max], "c-", lw=4, label=f"WH = {WH_cm:.1f} cm")
    axes_m[0].plot([x_min, x_max], [y_min + int((y_max - y_min)*0.15)]*2, "y-", lw=4, label=f"BL = {BL_cm:.1f} cm")
    axes_m[0].plot([chest_x]*2, [side_res["wh_y_start"], side_res["wh_y_end"]], "m-", lw=4, label=f"2b = {2*b_cm:.1f} cm")
    axes_m[0].set_title(f"Morfometri Tampak Samping (Side View)\nS_factor = {S_side:.5f} cm/px", fontsize=12, fontweight="bold")
    axes_m[0].legend(loc="upper right", fontsize=10)
    axes_m[0].axis("off")

    back_rgb = cv2.cvtColor(back_img, cv2.COLOR_BGR2RGB)
    ov_back = back_rgb.copy()
    ov_back[back_res["cow_mask"] == 255] = [0, 255, 0]
    back_blend = cv2.addWeighted(back_rgb, 0.75, ov_back, 0.25, 0)

    xc = back_res["x_center"]
    yc = back_res["chest_y"]
    a_px = back_res["a_px"]
    b_px = side_res["b_px"]

    axes_m[1].imshow(back_blend)
    axes_m[1].plot([xc - a_px, xc + a_px], [yc]*2, "m-", lw=4, label=f"2a = {2*a_cm:.1f} cm")
    axes_m[1].add_patch(patches.Ellipse((xc, yc), 2 * a_px, 2 * b_px, fill=False, edgecolor="deepskyblue", lw=3, label=f"CG = {CG_cm:.1f} cm"))
    axes_m[1].set_title(f"Morfometri Tampak Belakang (Back View)\nS_factor = {S_back:.5f} cm/px", fontsize=12, fontweight="bold")
    axes_m[1].legend(loc="upper right", fontsize=10)
    axes_m[1].axis("off")

    plt.tight_layout()
    buf_m = io.BytesIO()
    fig_morph.savefig(buf_m, format="png", bbox_inches="tight", dpi=120)
    buf_m.seek(0)
    morph_b64 = base64.b64encode(buf_m.read()).decode("utf-8")
    plt.close(fig_morph)

    # 2. Deteksi Stiker Sesuai Bentuk
    fig_stick, axes_s = plt.subplots(1, 2, figsize=(14, 6))
    _draw_sticker_on_axes(axes_s[0], side_img, side_res, S_side, "Tampak Samping")
    _draw_sticker_on_axes(axes_s[1], back_img, back_res, S_back, "Tampak Belakang")

    plt.tight_layout()
    buf_s = io.BytesIO()
    fig_stick.savefig(buf_s, format="png", bbox_inches="tight", dpi=120)
    buf_s.seek(0)
    sticker_b64 = base64.b64encode(buf_s.read()).decode("utf-8")
    plt.close(fig_stick)

    # 3. Dashboard Lengkap (4 Panel Grid)
    fig_all, axes_all = plt.subplots(2, 2, figsize=(16, 12))
    axes_all[0, 0].imshow(side_blend)
    axes_all[0, 0].plot([x_min + (x_max - x_min) / 2]*2, [y_min, y_max], "c-", lw=3, label=f"WH = {WH_cm:.1f} cm")
    axes_all[0, 0].plot([x_min, x_max], [y_min + int((y_max - y_min)*0.15)]*2, "y-", lw=3, label=f"BL = {BL_cm:.1f} cm")
    axes_all[0, 0].plot([chest_x]*2, [side_res["wh_y_start"], side_res["wh_y_end"]], "m-", lw=3, label=f"2b = {2*b_cm:.1f} cm")
    axes_all[0, 0].set_title("Morfometri Samping (BL, WH, 2b)", fontsize=11, fontweight="bold")
    axes_all[0, 0].legend(loc="upper right", fontsize=8)
    axes_all[0, 0].axis("off")

    axes_all[0, 1].imshow(back_blend)
    axes_all[0, 1].plot([xc - a_px, xc + a_px], [yc]*2, "m-", lw=3, label=f"2a = {2*a_cm:.1f} cm")
    axes_all[0, 1].add_patch(patches.Ellipse((xc, yc), 2 * a_px, 2 * b_px, fill=False, edgecolor="deepskyblue", lw=3, label=f"CG = {CG_cm:.1f} cm"))
    axes_all[0, 1].set_title("Morfometri Belakang (2a, CG)", fontsize=11, fontweight="bold")
    axes_all[0, 1].legend(loc="upper right", fontsize=8)
    axes_all[0, 1].axis("off")

    _draw_sticker_on_axes(axes_all[1, 0], side_img, side_res, S_side, "Deteksi Stiker Samping")
    _draw_sticker_on_axes(axes_all[1, 1], back_img, back_res, S_back, "Deteksi Stiker Belakang")

    plt.tight_layout()
    buf_all = io.BytesIO()
    fig_all.savefig(buf_all, format="png", bbox_inches="tight", dpi=120)
    buf_all.seek(0)
    full_b64 = base64.b64encode(buf_all.read()).decode("utf-8")
    plt.close(fig_all)

    return {
        "morphometry_b64": morph_b64,
        "sticker_detection_b64": sticker_b64,
        "full_dashboard_b64": full_b64,
    }


def _run_prediction_pipeline(
    side_bytes: bytes,
    back_bytes: bytes,
    registry: ModelRegistry,
    side_filename: str,
    back_filename: str,
    side_sticker_cm: float = 10.16,
    back_sticker_cm: float = 10.16,
    sticker_shape: str = "square",
    include_visualizations: bool = True,
) -> PredictionResponse:
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

    shape_mode = "circle" if str(sticker_shape).lower().startswith("cir") else "square"
    sticker_model = registry.get_sticker_model(shape_mode)

    # --- Parallel segmentation: side + back processed concurrently ---
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="seg") as seg_pool:
        future_side = seg_pool.submit(
            process_side_view,
            side_img,
            registry.seg_model,
            sticker_model,
            target_cm=side_sticker_cm,
            shape=shape_mode,
        )
        future_back = seg_pool.submit(
            process_back_view,
            back_img,
            registry.seg_model,
            sticker_model,
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

    # --- Visualizations ---
    vis_dict = None
    full_b64 = None
    morph_b64 = None
    sticker_b64 = None
    if include_visualizations:
        vis_dict = _generate_visualizations(
            side_img, side_res, back_img, back_res,
            scale_side, scale_back,
            body_length_cm, withers_height_cm, b_cm, a_cm, chest_girth_cm
        )
        full_b64 = vis_dict["full_dashboard_b64"]
        morph_b64 = vis_dict["morphometry_b64"]
        sticker_b64 = vis_dict["sticker_detection_b64"]

    return PredictionResponse(
        predicted_weight_kg=round(weight_kg, 2),
        s_factor={
            "side": round(scale_side, 6),
            "back": round(scale_back, 6),
            "unit": "cm/px",
        },
        calibration=CalibrationInfo(
            sticker_shape=shape_mode,
            scale_source=calib_source,
            side_sticker_size_cm=side_sticker_cm,
            back_sticker_size_cm=back_sticker_cm,
            scale_side_cm_per_px=round(scale_side, 6),
            scale_back_cm_per_px=round(scale_back, 6),
            s_factor_side=round(scale_side, 6),
            s_factor_back=round(scale_back, 6),
        ),
        measurements=PhysicalMeasurements(
            body_length_cm=body_length_cm,
            withers_height_cm=withers_height_cm,
            chest_girth_cm=chest_girth_cm,
            chest_width_2a_cm=round(2 * a_cm, 2),
            chest_depth_2b_cm=round(2 * b_cm, 2),
        ),
        model_features={k: float(v) for k, v in features.items()},
        visualizations=vis_dict,
        visualization_png_b64=full_b64,
        visualization_morphometry_b64=morph_b64,
        visualization_sticker_b64=sticker_b64,
    )


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Estimate cattle live weight with scale factor calibration",
)
async def predict(
    side_image: UploadFile = File(..., description="Side (lateral) view — JPEG or PNG"),
    back_image: UploadFile = File(..., description="Back (posterior) view — JPEG or PNG"),
    side_sticker_cm: Optional[float] = Form(None, description="Real size / diameter of calibration sticker on side image in cm"),
    back_sticker_cm: Optional[float] = Form(None, description="Real size / diameter of calibration sticker on back image in cm"),
    diameter_cm: Optional[float] = Form(None, description="Alias for circle sticker diameter in cm"),
    side_diameter_cm: Optional[float] = Form(None, description="Side view circle sticker diameter in cm"),
    back_diameter_cm: Optional[float] = Form(None, description="Back view circle sticker diameter in cm"),
    side_size_cm: Optional[float] = Form(None, description="Side sticker size in cm"),
    back_size_cm: Optional[float] = Form(None, description="Back sticker size in cm"),
    sticker_shape: str = Form("square", description="Sticker shape model: 'square' or 'circle'"),
    include_visualizations: bool = Form(True, description="Whether to include Base64 PNG visualization overlay images"),
    registry: ModelRegistry = Depends(get_registry),
) -> PredictionResponse:
    # Resolve shape mode
    shape_mode = "circle" if str(sticker_shape).lower().startswith("cir") else "square"

    # Resolving side sticker dimension (cm)
    if side_diameter_cm is not None:
        side_dim = float(side_diameter_cm)
    elif diameter_cm is not None and shape_mode == "circle":
        side_dim = float(diameter_cm)
    elif side_size_cm is not None:
        side_dim = float(side_size_cm)
    elif side_sticker_cm is not None:
        side_dim = float(side_sticker_cm)
    else:
        side_dim = 14.0 if shape_mode == "circle" else 10.16

    # Resolving back sticker dimension (cm)
    if back_diameter_cm is not None:
        back_dim = float(back_diameter_cm)
    elif diameter_cm is not None and shape_mode == "circle":
        back_dim = float(diameter_cm)
    elif back_size_cm is not None:
        back_dim = float(back_size_cm)
    elif back_sticker_cm is not None:
        back_dim = float(back_sticker_cm)
    else:
        back_dim = 14.0 if shape_mode == "circle" else 10.16

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
            side_dim,
            back_dim,
            shape_mode,
            include_visualizations,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=cfg.api_reload,
        log_level=cfg.log_level.lower(),
    )
