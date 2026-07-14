from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils.exceptions import PredictionError

logger = logging.getLogger(__name__)


def build_features(body_length_cm: float, withers_height_cm: float, chest_girth_cm: float) -> dict[str, float]:
    bl, wh, cg = body_length_cm, withers_height_cm, chest_girth_cm
    vol_proxy = bl * (cg ** 2)
    log_vol = float(np.log(vol_proxy)) if vol_proxy > 0.0 else 0.0

    features: dict[str, float] = {
        "BL_yolov8l": bl,
        "WH_yolov8l": wh,
        "CG_yolov8l": cg,
        "BL_WH": bl * wh,
        "BL_sq": bl ** 2,
        "WH_BL_ratio": wh / bl if bl > 0.0 else 0.0,
        "CG_sq": cg ** 2,
        "vol_proxy": vol_proxy,
        "log_vol": log_vol,
    }
    logger.debug("Features: %s", {k: f"{v:.3f}" for k, v in features.items()})
    return features


def predict_weight(features: dict[str, float], svr_pipe: object, feature_columns: list[str]) -> float:
    try:
        df = pd.DataFrame([features])[feature_columns]
        log_pred: float = svr_pipe.predict(df)[0]  # type: ignore[union-attr]
        weight_kg = float(np.exp(log_pred))
    except Exception as exc:
        raise PredictionError(f"SVR prediction failed: {exc}") from exc

    logger.info("Predicted: %.2f kg (log: %.4f)", weight_kg, log_pred)
    return weight_kg
