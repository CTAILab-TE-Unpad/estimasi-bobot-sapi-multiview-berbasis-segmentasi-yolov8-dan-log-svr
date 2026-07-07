"""Pydantic schemas for the /predict endpoint.

Defining explicit response models provides:
- Auto-generated OpenAPI documentation with field descriptions
- Runtime output validation (catches accidental ``None`` values, wrong types)
- Type-safe serialization without ``JSONResponse(content=dict(...))`` boilerplate
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CalibrationInfo(BaseModel):
    """Pixel-to-centimetre scale calibration metadata."""

    scale_source: str = Field(
        description=(
            "How the scale factor was determined. "
            "One of: 'dual_sticker', 'bridged_from_back_sticker', 'bridged_from_side_sticker'."
        )
    )
    scale_side_cm_per_px: float = Field(
        description="Side view pixel-to-cm conversion factor (cm/px)."
    )
    scale_back_cm_per_px: float = Field(
        description="Back view pixel-to-cm conversion factor (cm/px)."
    )


class PhysicalMeasurements(BaseModel):
    """Estimated physical dimensions extracted from the cattle images."""

    body_length_cm: float = Field(description="Body Length (BL) in centimetres.")
    withers_height_cm: float = Field(description="Withers Height (WH) in centimetres.")
    chest_girth_cm: float = Field(
        description="Estimated Chest Girth (CG) via Ramanujan ellipse formula, in centimetres."
    )
    chest_width_2a_cm: float = Field(description="Full chest width (2a) from back view, in centimetres.")
    chest_depth_2b_cm: float = Field(description="Full chest depth (2b) from side view, in centimetres.")


class PredictionResponse(BaseModel):
    """Full response payload returned by POST /api/v1/predict."""

    predicted_weight_kg: float = Field(description="Estimated cattle live weight in kilograms.")
    calibration: CalibrationInfo = Field(description="Scale calibration details.")
    measurements: PhysicalMeasurements = Field(description="Extracted physical measurements.")
    model_features: dict[str, float] = Field(
        description="All engineered features passed to the SVR model."
    )
    visualization_png_b64: str = Field(
        description="Base64-encoded PNG of the annotated side and back view images."
    )
