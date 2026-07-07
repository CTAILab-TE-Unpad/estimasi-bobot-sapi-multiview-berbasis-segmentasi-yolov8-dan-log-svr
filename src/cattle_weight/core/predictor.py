"""Feature engineering and SVR weight prediction.

The SVR model was trained in **log-space**: it predicts ``log(weight_kg)``,
so the output must be exponentiated to recover the weight in kilograms.

The engineered features capture non-linear interactions between the three
primary physical measurements (Body Length, Withers Height, Chest Girth)
that are most predictive of live weight.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cattle_weight.exceptions import PredictionError

logger = logging.getLogger(__name__)


def build_features(
    body_length_cm: float,
    withers_height_cm: float,
    chest_girth_cm: float,
) -> dict[str, float]:
    """Engineer regression features from the three primary morphometric measurements.

    The feature set mirrors what the SVR was trained on. The column order is
    enforced at prediction time by the metadata's ``feature_columns`` list.

    Args:
        body_length_cm: Body length (BL) in centimetres.
        withers_height_cm: Withers height (WH) in centimetres.
        chest_girth_cm: Estimated chest girth (CG) in centimetres.

    Returns:
        Ordered dictionary mapping feature names to float values.
    """
    bl = body_length_cm
    wh = withers_height_cm
    cg = chest_girth_cm

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

    logger.debug("Built features: %s", {k: f"{v:.3f}" for k, v in features.items()})
    return features


def predict_weight(
    features: dict[str, float],
    svr_pipe: object,
    feature_columns: list[str],
) -> float:
    """Predict cattle live weight using the trained Log-SVR pipeline.

    Args:
        features: Engineered feature dict from :func:`build_features`.
        svr_pipe: Fitted scikit-learn pipeline loaded from joblib.
        feature_columns: Ordered list of expected feature names from model
            metadata. Ensures column order matches the training-time order.

    Returns:
        Predicted live weight in kilograms (positive float).

    Raises:
        PredictionError: If inference fails for any reason.
    """
    try:
        df = pd.DataFrame([features])[feature_columns]
        log_pred: float = svr_pipe.predict(df)[0]  # type: ignore[union-attr]
        weight_kg = float(np.exp(log_pred))
    except Exception as exc:
        raise PredictionError(f"SVR prediction failed: {exc}") from exc

    logger.info("Predicted weight: %.2f kg  (log-space: %.4f)", weight_kg, log_pred)
    return weight_kg
