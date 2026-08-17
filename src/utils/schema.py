from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class CalibrationInfo(BaseModel):
    sticker_shape: str = Field(default="square", description="Sticker shape: 'square' or 'circle'.")
    scale_source: str = Field(description="One of: 'dual_sticker', 'bridged_from_back_sticker', 'bridged_from_side_sticker'.")
    side_sticker_size_cm: float = Field(default=10.16, description="Side sticker calibration size / diameter in cm.")
    back_sticker_size_cm: float = Field(default=10.16, description="Back sticker calibration size / diameter in cm.")
    scale_side_cm_per_px: float = Field(description="Side view pixel-to-cm conversion factor.")
    scale_back_cm_per_px: float = Field(description="Back view pixel-to-cm conversion factor.")
    s_factor_side: float = Field(description="Alias for scale_side_cm_per_px.")
    s_factor_back: float = Field(description="Alias for scale_back_cm_per_px.")


class PhysicalMeasurements(BaseModel):
    body_length_cm: float
    withers_height_cm: float
    chest_girth_cm: float
    chest_width_2a_cm: float
    chest_depth_2b_cm: float


class PredictionResponse(BaseModel):
    predicted_weight_kg: float
    s_factor: dict[str, float | str] = Field(description="Scale factors (cm/px) for side and back views.")
    calibration: CalibrationInfo
    measurements: PhysicalMeasurements
    model_features: dict[str, float]
    visualizations: Optional[dict[str, str]] = Field(default=None, description="Base64 PNG visualizations for morphometry and sticker detection.")
    visualization_png_b64: Optional[str] = Field(default=None, description="Combined dashboard visualization.")
    visualization_morphometry_b64: Optional[str] = Field(default=None, description="Cattle morphometry visualization.")
    visualization_sticker_b64: Optional[str] = Field(default=None, description="Sticker detection visualization according to shape.")
