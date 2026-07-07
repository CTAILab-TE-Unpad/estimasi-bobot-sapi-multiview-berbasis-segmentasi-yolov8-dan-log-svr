"""YOLO-based segmentation utilities for cattle silhouette and sticker detection.

Two responsibilities:

1. **Sticker detection** (:func:`get_sticker_scale`) — detects the calibration
   sticker and returns the pixel-to-cm scale factor.
2. **Cattle segmentation** (:func:`run_segmentation`) — runs the YOLOv8-seg model
   and returns the largest detected binary mask.

Both functions accept a pre-loaded YOLO model instance from :class:`ModelRegistry`
and operate on a BGR :class:`numpy.ndarray` (OpenCV convention).
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from cattle_weight.config import STICKER_CONF_THRESHOLD, STICKER_TARGET_CM
from cattle_weight.exceptions import SegmentationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: uint8 binary mask (0 or 255), shape (H, W)
BinaryMask = NDArray[np.uint8]

#: Bounding box as (x_min, y_min, width, height) in pixel coordinates
BBox = tuple[int, int, int, int]

#: Return type of :func:`get_sticker_scale`
StickerResult = tuple[float | None, BBox | None, BinaryMask | None]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_sticker_scale(
    img: NDArray,
    model: Any,
    target_cm: float = STICKER_TARGET_CM,
    conf: float = STICKER_CONF_THRESHOLD,
) -> StickerResult:
    """Detect the calibration sticker and compute pixel-to-cm scale factor.

    Iterates over all YOLO detections and returns the first mask whose bounding
    box has a positive diameter. Returns ``(None, None, None)`` if no sticker
    is found — callers should handle this case via the bridge calibration method.

    Args:
        img: BGR image array (H×W×3, uint8).
        model: Loaded YOLO sticker detection model instance.
        target_cm: Real-world diameter of the sticker in centimetres.
        conf: YOLO confidence threshold for sticker detection.

    Returns:
        ``(scale_cm_per_px, bbox, binary_mask)`` — all ``None`` if not found.
    """
    results = model(img, conf=conf, verbose=False)

    for r in results:
        if r.masks is None:
            continue
        for mask_data in r.masks.data:
            mask_np = mask_data.cpu().numpy()
            mask_np = cv2.resize(
                mask_np,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            y_idx, x_idx = np.where(mask_np > 0.5)
            if len(y_idx) == 0:
                continue

            x_min, x_max = int(x_idx.min()), int(x_idx.max())
            y_min, y_max = int(y_idx.min()), int(y_idx.max())
            width_px = x_max - x_min
            height_px = y_max - y_min
            diameter_px = (width_px + height_px) / 2.0

            if diameter_px <= 0:
                logger.debug("Sticker mask found but diameter is zero — skipping.")
                continue

            scale = target_cm / diameter_px
            bbox: BBox = (x_min, y_min, width_px, height_px)
            logger.debug("Sticker detected: diameter=%.1f px, scale=%.5f cm/px", diameter_px, scale)
            return scale, bbox, (mask_np * 255).astype(np.uint8)

    logger.debug("No sticker detected in image.")
    return None, None, None


def run_segmentation(img: NDArray, model: Any) -> BinaryMask:
    """Run YOLOv8 instance segmentation and return the primary cattle mask.

    Selects the first (highest-confidence) detection mask and resizes it to
    the input image dimensions.

    Args:
        img: BGR image array (H×W×3, uint8).
        model: Loaded YOLOv8-seg cattle segmentation model instance.

    Returns:
        Binary mask array (uint8, values 0 or 255), shape (H, W).

    Raises:
        SegmentationError: If no detections are found or the mask is empty.
    """
    results = model(img, verbose=False)

    for r in results:
        if r.masks is not None and len(r.masks.data) > 0:
            mask_np = r.masks.data[0].cpu().numpy()
            mask_resized = cv2.resize(
                mask_np,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            binary: BinaryMask = (mask_resized * 255).astype(np.uint8)

            if binary.max() == 0:
                raise SegmentationError(
                    "YOLO returned a mask, but it is entirely empty after resizing."
                )
            logger.debug("Segmentation succeeded: mask coverage = %.1f%%", binary.mean() / 255 * 100)
            return binary

    raise SegmentationError(
        "No objects detected in the image. "
        "Ensure the image clearly shows the cattle from the correct angle."
    )
