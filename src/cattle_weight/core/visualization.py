"""Visualization of inference results as an annotated side-by-side image.

Generates a matplotlib figure with both views annotated with measurement
overlays (body length line, withers height line, ellipse girth approximation,
sticker bounding box) and returns it as a base64-encoded PNG string suitable
for embedding in a JSON API response.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for threads and headless servers
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def generate_result_visualization(
    side_img: NDArray,
    side_res: dict[str, Any],
    back_img: NDArray,
    back_res: dict[str, Any],
    scale_side: float,
    scale_back: float,
    body_length_cm: float,
    withers_height_cm: float,
    b_cm: float,
    a_cm: float,
    chest_girth_cm: float,
    dpi: int = 100,
) -> str:
    """Generate an annotated side-by-side visualization and return as base64 PNG.

    Args:
        side_img: BGR array of the side view image.
        side_res: Landmark dict from :func:`~cattle_weight.core.morphometry.process_side_view`.
        back_img: BGR array of the back view image.
        back_res: Landmark dict from :func:`~cattle_weight.core.morphometry.process_back_view`.
        scale_side: Pixel-to-cm conversion factor for the side view.
        scale_back: Pixel-to-cm conversion factor for the back view.
        body_length_cm: Body length (BL) in cm.
        withers_height_cm: Withers height (WH) in cm.
        b_cm: Chest depth semi-axis in cm.
        a_cm: Chest width semi-axis in cm.
        chest_girth_cm: Estimated chest girth (CG) in cm.
        dpi: Output PNG resolution in dots per inch.

    Returns:
        Base64-encoded UTF-8 string of the PNG image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # -----------------------------------------------------------------------
    # Side view panel
    # -----------------------------------------------------------------------
    side_rgb = cv2.cvtColor(side_img, cv2.COLOR_BGR2RGB)
    overlay = side_rgb.copy()
    overlay[side_res["cow_mask"] == 255] = [0, 255, 0]
    blended = cv2.addWeighted(side_rgb, 0.7, overlay, 0.3, 0)

    x_min, x_max = side_res["x_min"], side_res["x_max"]
    y_min = side_res["y_min"]
    chest_x = side_res["chest_x"]
    wh_y_start, wh_y_end = side_res["wh_y_start"], side_res["wh_y_end"]

    axes[0].imshow(blended)
    # Withers height line
    axes[0].plot(
        [x_min + (x_max - x_min) / 2] * 2, [wh_y_start, wh_y_end],
        "c-", lw=3, label=f"WH = {withers_height_cm:.1f} cm",
    )
    # Body length line
    axes[0].plot(
        [x_min, x_max], [y_min + 100] * 2,
        "y-", lw=3, label=f"BL = {body_length_cm:.1f} cm",
    )
    # Chest depth (2b) line
    axes[0].plot(
        [chest_x] * 2, [y_min, y_min + 2 * side_res["b_px"]],
        "m-", lw=3, label=f"2b = {2 * b_cm:.1f} cm",
    )
    # Sticker bounding box
    if side_res.get("sticker_bbox") is not None:
        sbx, sby, sbw, sbh = side_res["sticker_bbox"]
        axes[0].add_patch(
            patches.Rectangle(
                (sbx, sby), sbw, sbh,
                linewidth=2, edgecolor="red", facecolor="none", label="Sticker",
            )
        )
    axes[0].set_title(f"Side View  ·  scale = {scale_side:.4f} cm/px", fontsize=11)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].axis("off")

    # -----------------------------------------------------------------------
    # Back view panel
    # -----------------------------------------------------------------------
    back_rgb = cv2.cvtColor(back_img, cv2.COLOR_BGR2RGB)
    overlay_back = back_rgb.copy()
    overlay_back[back_res["cow_mask"] == 255] = [0, 255, 0]
    blended_back = cv2.addWeighted(back_rgb, 0.7, overlay_back, 0.3, 0)

    xc = back_res["x_center"]
    yc = back_res["chest_y"]
    a_px = back_res["a_px"]
    b_px = side_res["b_px"]

    axes[1].imshow(blended_back)
    # Chest width (2a) line
    axes[1].plot(
        [xc - a_px, xc + a_px], [yc] * 2,
        "m-", lw=3, label=f"2a = {2 * a_cm:.1f} cm",
    )
    # Sticker bounding box
    if back_res.get("sticker_bbox") is not None:
        bbx, bby, bbw, bbh = back_res["sticker_bbox"]
        axes[1].add_patch(
            patches.Rectangle(
                (bbx, bby), bbw, bbh,
                linewidth=2, edgecolor="red", facecolor="none", label="Sticker",
            )
        )
    # Chest girth ellipse
    axes[1].add_patch(
        patches.Ellipse(
            (xc, yc), 2 * a_px, 2 * b_px,
            fill=False, edgecolor="dodgerblue", lw=2.5,
            label=f"CG ≈ {chest_girth_cm:.1f} cm",
        )
    )
    axes[1].set_title(f"Back View  ·  scale = {scale_back:.4f} cm/px", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].axis("off")

    # -----------------------------------------------------------------------
    # Encode to base64
    # -----------------------------------------------------------------------
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    logger.debug("Visualization generated: %d bytes (base64).", len(encoded))
    return encoded
