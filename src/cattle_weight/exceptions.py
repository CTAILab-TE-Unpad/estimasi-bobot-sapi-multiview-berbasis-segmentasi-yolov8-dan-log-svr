"""Custom domain exception hierarchy for the Cattle Weight Estimation API.

Raising typed exceptions (instead of bare ``Exception`` or ``HTTPException``)
keeps the core logic decoupled from the HTTP transport layer. FastAPI exception
handlers in ``api/app.py`` map these to the appropriate HTTP responses.
"""


class CattleWeightError(Exception):
    """Base exception for all cattle weight domain errors."""


class SegmentationError(CattleWeightError):
    """Raised when YOLO segmentation fails or produces an invalid/empty result."""


class CalibrationError(CattleWeightError):
    """Raised when sticker-based pixel-to-cm scale calibration fails.

    This includes cases where the sticker is undetected in both views and
    no bridge calibration is possible.
    """


class PredictionError(CattleWeightError):
    """Raised when the SVR inference pipeline fails unexpectedly."""
