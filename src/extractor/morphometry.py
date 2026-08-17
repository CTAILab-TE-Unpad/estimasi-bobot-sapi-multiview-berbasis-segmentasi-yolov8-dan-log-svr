from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.extractor.segmentation import BBox, BinaryMask, get_sticker_scale, run_segmentation
from src.utils.exceptions import CalibrationError, SegmentationError
from src.utils.settings import CHEST_DEPTH_TOP_FRACTION, CHEST_X_RATIO

logger = logging.getLogger(__name__)


def ramanujan_girth(a: float, b: float) -> float:
    return float(np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b))))


def process_side_view(
    img: NDArray,
    seg_model: Any,
    sticker_model: Any,
    target_cm: float = 10.16,
    shape: str = "square",
) -> dict[str, Any]:
    scale, sticker_bbox, _, sticker_geom = get_sticker_scale(img, sticker_model, target_cm=target_cm, shape=shape)
    cow_mask: BinaryMask = run_segmentation(img, seg_model)

    y_idx, x_idx = np.where(cow_mask == 255)
    x_min, x_max = int(x_idx.min()), int(x_idx.max())
    y_min, y_max = int(y_idx.min()), int(y_idx.max())

    body_length_px = float(x_max - x_min)
    chest_x = x_min + int(CHEST_X_RATIO * body_length_px)

    col_pixels = np.where(cow_mask[:, chest_x] == 255)[0]
    if len(col_pixels) > 0:
        wh_y_start, wh_y_end = int(col_pixels.min()), int(col_pixels.max())
    else:
        wh_y_start, wh_y_end = y_min, y_max

    withers_height_px = float(wh_y_end - wh_y_start)

    top_cutoff = y_min + int(CHEST_DEPTH_TOP_FRACTION * (y_max - y_min))
    dorsal_mask = np.zeros_like(cow_mask)
    dorsal_mask[y_min:top_cutoff, :] = cow_mask[y_min:top_cutoff, :]
    dorsal_y, _ = np.where(dorsal_mask == 255)
    b_px = (float(dorsal_y.max()) - float(dorsal_y.min())) / 2.0 if len(dorsal_y) > 0 else withers_height_px / 2.0

    logger.debug("Side: BL=%.1f px, WH=%.1f px, b=%.1f px", body_length_px, withers_height_px, b_px)

    return {
        "scale": scale,
        "sticker_bbox": sticker_bbox,
        "sticker_geom": sticker_geom,
        "cow_mask": cow_mask,
        "body_length_px": body_length_px,
        "withers_height_px": withers_height_px,
        "b_px": b_px,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "chest_x": chest_x,
        "wh_y_start": wh_y_start, "wh_y_end": wh_y_end,
    }


def process_back_view(
    img: NDArray,
    seg_model: Any,
    sticker_model: Any,
    target_cm: float = 10.16,
    shape: str = "square",
) -> dict[str, Any]:
    scale, sticker_bbox, _, sticker_geom = get_sticker_scale(img, sticker_model, target_cm=target_cm, shape=shape)
    cow_mask: BinaryMask = run_segmentation(img, seg_model)

    y_idx, x_idx = np.where(cow_mask == 255)
    y_min, y_max = int(y_idx.min()), int(y_idx.max())

    chest_y = y_min + int(CHEST_X_RATIO * (y_max - y_min))
    row_pixels = np.where(cow_mask[chest_y, :] == 255)[0]

    if len(row_pixels) > 0:
        x_start, x_end = int(row_pixels.min()), int(row_pixels.max())
    else:
        x_start, x_end = int(x_idx.min()), int(x_idx.max())

    a_px = (x_end - x_start) / 2.0
    x_center = (x_start + x_end) / 2.0

    logger.debug("Back: a=%.1f px, x_center=%.1f px", a_px, x_center)

    return {
        "scale": scale,
        "sticker_bbox": sticker_bbox,
        "sticker_geom": sticker_geom,
        "cow_mask": cow_mask,
        "a_px": a_px,
        "x_center": x_center,
        "chest_y": float(chest_y),
        "withers_height_px": float(y_max - y_min),
    }


def bridge_scale(
    scale_side: float | None,
    scale_back: float | None,
    wh_side_px: float,
    wh_back_px: float,
) -> tuple[float, float, str]:
    if scale_side is not None and scale_back is not None:
        return scale_side, scale_back, "dual_sticker"

    if scale_side is None and scale_back is None:
        raise CalibrationError(
            "Sticker not detected in either view. At least one must be visible."
        )

    if scale_side is None:
        if wh_side_px <= 0:
            raise CalibrationError("Back sticker detected but side withers height is zero.")
        bridged = scale_back * (wh_back_px / wh_side_px)  # type: ignore[operator]
        logger.info("Scale bridged from back sticker: %.5f cm/px", bridged)
        return bridged, scale_back, "bridged_from_back_sticker"  # type: ignore[return-value]

    if wh_back_px <= 0:
        raise CalibrationError("Side sticker detected but back withers height is zero.")
    bridged = scale_side * (wh_side_px / wh_back_px)
    logger.info("Scale bridged from side sticker: %.5f cm/px", bridged)
    return scale_side, bridged, "bridged_from_side_sticker"
