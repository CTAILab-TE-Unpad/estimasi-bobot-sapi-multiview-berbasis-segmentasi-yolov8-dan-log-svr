"""Morphometric landmark extraction from segmentation masks.

Provides three main entry points:

- :func:`process_side_view` — extract body length, withers height, chest depth
  from the lateral (side) image.
- :func:`process_back_view` — extract chest width semi-axis from the posterior
  (back) image.
- :func:`bridge_scale` — resolve a missing scale factor using the withers-height
  ratio between views.
- :func:`ramanujan_girth` — compute ellipse perimeter approximation.

All functions accept BGR :class:`numpy.ndarray` images and pre-loaded YOLO
models from :class:`~cattle_weight.infrastructure.model_registry.ModelRegistry`.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cattle_weight.config import CHEST_DEPTH_TOP_FRACTION, CHEST_X_RATIO
from cattle_weight.core.segmentation import BBox, BinaryMask, get_sticker_scale, run_segmentation
from cattle_weight.exceptions import CalibrationError, SegmentationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def ramanujan_girth(a: float, b: float) -> float:
    """Approximate ellipse perimeter using Ramanujan's second formula.

    Used to estimate Chest Girth (CG) from the chest cross-section modelled
    as an ellipse with semi-axes *a* (width, from back view) and *b* (depth,
    from side view).

    Args:
        a: Horizontal semi-axis in centimetres (half chest width).
        b: Vertical semi-axis in centimetres (half chest depth).

    Returns:
        Estimated perimeter in centimetres.
    """
    return float(np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b))))


# ---------------------------------------------------------------------------
# View processing
# ---------------------------------------------------------------------------


def process_side_view(
    img: NDArray,
    seg_model: Any,
    sticker_model: Any,
) -> dict[str, Any]:
    """Extract morphometric landmarks from the side (lateral) view image.

    Pipeline:
    1. Detect calibration sticker → pixel-to-cm scale factor.
    2. Segment cattle silhouette → binary mask.
    3. Measure body length, withers height, chest depth from mask geometry.

    Args:
        img: BGR image array of the side view.
        seg_model: Loaded YOLOv8 cattle segmentation model.
        sticker_model: Loaded YOLOv8 sticker detection model.

    Returns:
        Dictionary with keys:

        - ``scale`` (float | None): Pixel-to-cm scale; None if sticker missing.
        - ``sticker_bbox`` (BBox | None): Sticker bounding box in pixels.
        - ``cow_mask`` (BinaryMask): uint8 binary silhouette mask.
        - ``body_length_px`` (float): Horizontal body extent in pixels.
        - ``withers_height_px`` (float): Vertical chest column height in pixels.
        - ``b_px`` (float): Chest depth semi-axis in pixels.
        - ``x_min``, ``x_max``, ``y_min``, ``y_max`` (int): Bounding box corners.
        - ``chest_x`` (int): x-coordinate of the chest measurement column.
        - ``wh_y_start``, ``wh_y_end`` (int): y-extent of the chest column.

    Raises:
        SegmentationError: If the cattle silhouette cannot be segmented.
    """
    scale, sticker_bbox, _ = get_sticker_scale(img, sticker_model)
    cow_mask: BinaryMask = run_segmentation(img, seg_model)

    y_idx, x_idx = np.where(cow_mask == 255)
    x_min, x_max = int(x_idx.min()), int(x_idx.max())
    y_min, y_max = int(y_idx.min()), int(y_idx.max())

    body_length_px = float(x_max - x_min)
    chest_x = x_min + int(CHEST_X_RATIO * body_length_px)

    # Withers height: vertical extent of the mask at the chest column
    col_pixels = np.where(cow_mask[:, chest_x] == 255)[0]
    if len(col_pixels) > 0:
        wh_y_start, wh_y_end = int(col_pixels.min()), int(col_pixels.max())
    else:
        wh_y_start, wh_y_end = y_min, y_max

    withers_height_px = float(wh_y_end - wh_y_start)

    # Chest depth semi-axis (b): half the height of the upper dorsal region
    top_cutoff = y_min + int(CHEST_DEPTH_TOP_FRACTION * (y_max - y_min))
    dorsal_mask = np.zeros_like(cow_mask)
    dorsal_mask[y_min:top_cutoff, :] = cow_mask[y_min:top_cutoff, :]
    dorsal_y, _ = np.where(dorsal_mask == 255)

    b_px = (float(dorsal_y.max()) - float(dorsal_y.min())) / 2.0 if len(dorsal_y) > 0 else withers_height_px / 2.0

    logger.debug(
        "Side view: BL=%.1f px, WH=%.1f px, b=%.1f px, scale=%s",
        body_length_px, withers_height_px, b_px,
        f"{scale:.5f}" if scale is not None else "None",
    )

    return {
        "scale": scale,
        "sticker_bbox": sticker_bbox,
        "cow_mask": cow_mask,
        "body_length_px": body_length_px,
        "withers_height_px": withers_height_px,
        "b_px": b_px,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "chest_x": chest_x,
        "wh_y_start": wh_y_start,
        "wh_y_end": wh_y_end,
    }


def process_back_view(
    img: NDArray,
    seg_model: Any,
    sticker_model: Any,
) -> dict[str, Any]:
    """Extract morphometric landmarks from the back (posterior) view image.

    Pipeline:
    1. Detect calibration sticker → pixel-to-cm scale factor.
    2. Segment cattle silhouette → binary mask.
    3. Measure chest width semi-axis from mask geometry.

    Args:
        img: BGR image array of the back view.
        seg_model: Loaded YOLOv8 cattle segmentation model.
        sticker_model: Loaded YOLOv8 sticker detection model.

    Returns:
        Dictionary with keys:

        - ``scale`` (float | None): Pixel-to-cm scale; None if sticker missing.
        - ``sticker_bbox`` (BBox | None): Sticker bounding box in pixels.
        - ``cow_mask`` (BinaryMask): uint8 binary silhouette mask.
        - ``a_px`` (float): Chest width semi-axis in pixels.
        - ``x_center`` (float): x-centroid of the chest row in pixels.
        - ``chest_y`` (float): y-coordinate of the chest measurement row.
        - ``withers_height_px`` (float): Full vertical extent in pixels.

    Raises:
        SegmentationError: If the cattle silhouette cannot be segmented.
    """
    scale, sticker_bbox, _ = get_sticker_scale(img, sticker_model)
    cow_mask: BinaryMask = run_segmentation(img, seg_model)

    y_idx, x_idx = np.where(cow_mask == 255)
    y_min, y_max = int(y_idx.min()), int(y_idx.max())

    # Chest width: measured at CHEST_X_RATIO down from the top
    chest_y = y_min + int(CHEST_X_RATIO * (y_max - y_min))
    row_pixels = np.where(cow_mask[chest_y, :] == 255)[0]

    if len(row_pixels) > 0:
        x_start, x_end = int(row_pixels.min()), int(row_pixels.max())
    else:
        x_start, x_end = int(x_idx.min()), int(x_idx.max())

    a_px = (x_end - x_start) / 2.0
    x_center = (x_start + x_end) / 2.0

    logger.debug(
        "Back view: a=%.1f px, x_center=%.1f px, scale=%s",
        a_px, x_center,
        f"{scale:.5f}" if scale is not None else "None",
    )

    return {
        "scale": scale,
        "sticker_bbox": sticker_bbox,
        "cow_mask": cow_mask,
        "a_px": a_px,
        "x_center": x_center,
        "chest_y": float(chest_y),
        "withers_height_px": float(y_max - y_min),
    }


# ---------------------------------------------------------------------------
# Scale bridging
# ---------------------------------------------------------------------------


def bridge_scale(
    scale_side: float | None,
    scale_back: float | None,
    wh_side_px: float,
    wh_back_px: float,
) -> tuple[float, float, str]:
    """Resolve missing scale factors using the withers-height bridge method.

    When a sticker is undetected in one view, the known scale from the other
    view is transferred using the withers-height pixel ratio as a proxy for
    the physical distance to the camera.

    Args:
        scale_side: Scale from side view (``None`` if sticker not found).
        scale_back: Scale from back view (``None`` if sticker not found).
        wh_side_px: Withers height in pixels from side view.
        wh_back_px: Withers height in pixels from back view.

    Returns:
        ``(scale_side, scale_back, source_label)`` where ``source_label`` is
        one of: ``"dual_sticker"``, ``"bridged_from_back_sticker"``,
        ``"bridged_from_side_sticker"``.

    Raises:
        CalibrationError: If both stickers are missing, or if the bridge view
            has a zero/invalid withers height pixel value.
    """
    if scale_side is not None and scale_back is not None:
        return scale_side, scale_back, "dual_sticker"

    if scale_side is None and scale_back is None:
        raise CalibrationError(
            "Calibration sticker was not detected in either the side or back view. "
            "At least one reference sticker must be visible to compute real-world measurements."
        )

    if scale_side is None:
        # Bridge: derive side scale from back scale + WH ratio
        if wh_side_px <= 0:
            raise CalibrationError(
                "Back sticker detected, but side withers height is zero — "
                "cannot bridge scale to side view."
            )
        bridged = scale_back * (wh_back_px / wh_side_px)  # type: ignore[operator]
        logger.info("Scale bridged: side scale derived from back sticker (%.5f cm/px)", bridged)
        return bridged, scale_back, "bridged_from_back_sticker"  # type: ignore[return-value]

    # scale_back is None → bridge from side
    if wh_back_px <= 0:
        raise CalibrationError(
            "Side sticker detected, but back withers height is zero — "
            "cannot bridge scale to back view."
        )
    bridged = scale_side * (wh_side_px / wh_back_px)
    logger.info("Scale bridged: back scale derived from side sticker (%.5f cm/px)", bridged)
    return scale_side, bridged, "bridged_from_side_sticker"
