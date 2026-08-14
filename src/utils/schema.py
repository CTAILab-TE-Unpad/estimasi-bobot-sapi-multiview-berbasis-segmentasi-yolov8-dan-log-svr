from __future__ import annotations

from pydantic import BaseModel, Field


class CalibrationInfo(BaseModel):
    scale_source: str = Field(description="One of: 'dual_sticker', 'bridged_from_back_sticker', 'bridged_from_side_sticker'.")
    side_sticker_size_cm: float = Field(default=10.16, description="Side sticker calibration size in cm.")
    back_sticker_size_cm: float = Field(default=10.16, description="Back sticker calibration size in cm.")
    scale_side_cm_per_px: float = Field(description="Side view pixel-to-cm conversion factor.")
    scale_back_cm_per_px: float = Field(description="Back view pixel-to-cm conversion factor.")


class PhysicalMeasurements(BaseModel):
    body_length_cm: float
    withers_height_cm: float
    chest_girth_cm: float
    chest_width_2a_cm: float
    chest_depth_2b_cm: float


class PredictionResponse(BaseModel):
    predicted_weight_kg: float
    calibration: CalibrationInfo
    measurements: PhysicalMeasurements
    model_features: dict[str, float]
