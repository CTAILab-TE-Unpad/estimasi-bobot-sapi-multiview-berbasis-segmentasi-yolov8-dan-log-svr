"""Domain constants for the Cattle Weight Estimation pipeline.

These are **algorithm-level constants** derived from domain knowledge and
empirical calibration. They are intentionally NOT configurable at runtime via
environment variables — changing them changes the model's behaviour and should
be accompanied by a code review and re-validation.

For deployment / path / server configuration, see :mod:`cattle_weight.settings`.
"""

# ---------------------------------------------------------------------------
# Segmentation Quality Gates
# ---------------------------------------------------------------------------

#: Minimum contour area as a fraction of total image area.
#: Objects smaller than this are rejected as noise.
AREA_RATIO_MIN: float = 0.02

#: Maximum contour area as a fraction of total image area.
#: Objects larger than this likely indicate a mis-segmentation.
AREA_RATIO_MAX: float = 0.90

# ---------------------------------------------------------------------------
# Sticker-Based Calibration
# ---------------------------------------------------------------------------

#: Known real-world diameter of the calibration sticker (cm).
#: Must match the physical sticker used during image capture.
STICKER_TARGET_CM: float = 2.5

#: YOLO confidence threshold for sticker detection.
STICKER_CONF_THRESHOLD: float = 0.25

# ---------------------------------------------------------------------------
# Morphometry
# ---------------------------------------------------------------------------

#: Position of the chest cross-section along Body Length (ratio from front).
#: 0.25 = 25% from the front of the body bounding box.
CHEST_X_RATIO: float = 0.25

#: Top fraction of withers height used to estimate chest depth semi-axis (b).
#: Only the upper 55% of the silhouette height is considered.
CHEST_DEPTH_TOP_FRACTION: float = 0.55
