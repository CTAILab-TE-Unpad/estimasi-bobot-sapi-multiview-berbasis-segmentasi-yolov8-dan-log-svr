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
StickerResult = tuple[float | None, BBox | None, BinaryMask | None, dict[str, Any] | None]


def get_sticker_scale(
    img: NDArray,
    model: Any,
    target_cm: float = STICKER_TARGET_CM,
    shape: str = "square",
    conf: float = STICKER_CONF_THRESHOLD,
) -> StickerResult:
    results = model(img, conf=conf, verbose=False)

    for r in results:
        if len(r.boxes) == 0:
            continue

        box = r.boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        width_px = max(1, x2 - x1)
        height_px = max(1, y2 - y1)

        # Prepare mask if available
        mask_uint8: BinaryMask | None = None
        area: int = 0
        if r.masks is not None and len(r.masks.data) > 0:
            mask_np = r.masks.data[0].cpu().numpy()
            mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_uint8 = (mask_resized > 0.5).astype(np.uint8) * 255
            area = int(np.count_nonzero(mask_uint8 > 0))

        if shape.lower().startswith("sq"):
            if area > 0:
                side_px = np.sqrt(float(area))
            else:
                side_px = float((width_px + height_px) / 2.0)
            scale = target_cm / side_px
            sticker_geom = {
                "shape": "square",
                "side_px": float(side_px),
                "bbox": (x1, y1, width_px, height_px),
                "crop_bbox": (max(0, x1 - 10), max(0, y1 - 10), min(img.shape[1], x2 + 10), min(img.shape[0], y2 + 10))
            }
        else:
            # --- Circular Sticker: High-Res Sub-Pixel Ellipse Fitting ---
            pad_x = int(width_px * 0.20)
            pad_y = int(height_px * 0.20)
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(img.shape[1], x2 + pad_x)
            crop_y2 = min(img.shape[0], y2 + pad_y)

            crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            major_axis_px = float((width_px + height_px) / 2.0)
            minor_axis_px = float(min(width_px, height_px))
            center_x = float(x1 + width_px / 2.0)
            center_y = float(y1 + height_px / 2.0)
            ellipse_angle = 0.0

            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 30, 100)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

                cnts, _ = cv2.findContours(edges_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                crop_cx = crop.shape[1] / 2.0
                crop_cy = crop.shape[0] / 2.0
                expected_r = (width_px + height_px) / 4.0
                best_score = float("inf")

                for c in cnts:
                    if len(c) >= 15 and cv2.contourArea(c) > 200:
                        (xc, yc), (d1, d2), angle = cv2.fitEllipse(c)
                        major = float(max(d1, d2))
                        minor = float(min(d1, d2))
                        dist_to_center = float(np.hypot(xc - crop_cx, yc - crop_cy))
                        size_diff = abs(major - 2 * expected_r) / (2 * expected_r)

                        if size_diff < 0.25 and dist_to_center < expected_r * 0.5:
                            score = dist_to_center + size_diff * 50
                            if score < best_score:
                                best_score = score
                                major_axis_px = major
                                minor_axis_px = minor
                                center_x = float(crop_x1 + xc)
                                center_y = float(crop_y1 + yc)
                                ellipse_angle = float(angle)

            if major_axis_px <= 0:
                continue
            scale = target_cm / major_axis_px
            sticker_geom = {
                "shape": "circle",
                "center": (center_x, center_y),
                "major_axis": float(major_axis_px),
                "minor_axis": float(minor_axis_px),
                "angle": float(ellipse_angle),
                "bbox": (x1, y1, width_px, height_px),
                "crop_bbox": (crop_x1, crop_y1, crop_x2, crop_y2)
            }

        bbox: BBox = (x1, y1, width_px, height_px)
        logger.debug("Sticker: scale=%.5f cm/px (shape=%s)", scale, shape)
        return scale, bbox, mask_uint8, sticker_geom

    logger.debug("No sticker detected.")
    return None, None, None, None


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
