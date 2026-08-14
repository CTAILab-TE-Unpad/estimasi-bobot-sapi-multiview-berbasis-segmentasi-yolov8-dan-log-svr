from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.utils.exceptions import SegmentationError
from src.utils.settings import STICKER_CONF_THRESHOLD, STICKER_TARGET_CM

logger = logging.getLogger(__name__)

BinaryMask = NDArray[np.uint8]
BBox = tuple[int, int, int, int]
StickerResult = tuple[float | None, BBox | None, BinaryMask | None]


def get_sticker_scale(
    img: NDArray,
    model: Any,
    target_cm: float = STICKER_TARGET_CM,
    shape: str = "square",
    conf: float = STICKER_CONF_THRESHOLD,
) -> StickerResult:
    results = model(img, conf=conf, verbose=False)

    for r in results:
        if r.masks is None or len(r.boxes) == 0:
            continue
        for mask_data in r.masks.data:
            mask_np = mask_data.cpu().numpy()
            mask_np = cv2.resize(mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            y_idx, x_idx = np.where(mask_np > 0.5)
            if len(y_idx) == 0:
                continue

            area = np.count_nonzero(mask_np > 0.5)
            x_min, x_max = int(x_idx.min()), int(x_idx.max())
            y_min, y_max = int(y_idx.min()), int(y_idx.max())
            width_px = x_max - x_min
            height_px = y_max - y_min

            if shape.lower().startswith("sq"):
                if area <= 0:
                    continue
                scale = target_cm / np.sqrt(float(area))
            else:
                diameter_px = (width_px + height_px) / 2.0
                if diameter_px <= 0:
                    continue
                scale = target_cm / diameter_px

            bbox: BBox = (x_min, y_min, width_px, height_px)
            logger.debug("Sticker: scale=%.5f cm/px (shape=%s)", scale, shape)
            return scale, bbox, (mask_np * 255).astype(np.uint8)

    logger.debug("No sticker detected.")
    return None, None, None


def run_segmentation(img: NDArray, model: Any) -> BinaryMask:
    results = model(img, verbose=False)

    for r in results:
        if r.masks is not None and len(r.masks.data) > 0:
            mask_np = r.masks.data[0].cpu().numpy()
            mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            binary: BinaryMask = (mask_resized * 255).astype(np.uint8)

            if binary.max() == 0:
                raise SegmentationError("YOLO returned an empty mask after resizing.")
            logger.debug("Segmentation: mask coverage = %.1f%%", binary.mean() / 255 * 100)
            return binary

    raise SegmentationError(
        "No objects detected. Ensure the image clearly shows the cattle from the correct angle."
    )
