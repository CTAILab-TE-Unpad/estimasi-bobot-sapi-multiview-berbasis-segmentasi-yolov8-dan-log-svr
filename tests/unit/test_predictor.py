"""Unit tests for cattle_weight.core.predictor."""
from __future__ import annotations

import numpy as np
import pytest

from cattle_weight.core.predictor import build_features, predict_weight
from cattle_weight.exceptions import PredictionError

EXPECTED_KEYS = frozenset({
    "BL_yolov8l", "WH_yolov8l", "CG_yolov8l",
    "BL_WH", "BL_sq", "WH_BL_ratio",
    "CG_sq", "vol_proxy", "log_vol",
})


class TestBuildFeatures:
    """Tests for feature engineering logic."""

    def test_all_expected_keys_present(self) -> None:
        features = build_features(120.0, 110.0, 160.0)
        assert set(features.keys()) == EXPECTED_KEYS

    def test_all_values_are_float(self) -> None:
        features = build_features(120.0, 110.0, 160.0)
        for key, val in features.items():
            assert isinstance(val, float), f"Feature '{key}' is not float: {type(val)}"

    def test_derived_bl_wh(self) -> None:
        bl, wh, cg = 120.0, 110.0, 160.0
        f = build_features(bl, wh, cg)
        assert f["BL_WH"] == pytest.approx(bl * wh)

    def test_derived_bl_sq(self) -> None:
        bl = 120.0
        f = build_features(bl, 110.0, 160.0)
        assert f["BL_sq"] == pytest.approx(bl ** 2)

    def test_derived_wh_bl_ratio(self) -> None:
        bl, wh = 120.0, 110.0
        f = build_features(bl, wh, 160.0)
        assert f["WH_BL_ratio"] == pytest.approx(wh / bl)

    def test_derived_vol_proxy(self) -> None:
        bl, cg = 120.0, 160.0
        f = build_features(bl, 110.0, cg)
        assert f["vol_proxy"] == pytest.approx(bl * cg ** 2)

    def test_derived_log_vol(self) -> None:
        bl, cg = 120.0, 160.0
        f = build_features(bl, 110.0, cg)
        assert f["log_vol"] == pytest.approx(np.log(bl * cg ** 2))

    def test_zero_body_length_no_crash(self) -> None:
        """bl=0 should not raise — WH_BL_ratio and vol_proxy default to 0."""
        f = build_features(0.0, 110.0, 160.0)
        assert f["WH_BL_ratio"] == 0.0
        assert f["vol_proxy"] == 0.0
        assert f["log_vol"] == 0.0

    def test_primary_features_passthrough(self) -> None:
        bl, wh, cg = 130.5, 115.2, 170.8
        f = build_features(bl, wh, cg)
        assert f["BL_yolov8l"] == pytest.approx(bl)
        assert f["WH_yolov8l"] == pytest.approx(wh)
        assert f["CG_yolov8l"] == pytest.approx(cg)


class TestPredictWeight:
    """Tests for SVR inference wrapper."""

    def test_raises_prediction_error_on_none_pipe(self) -> None:
        features = build_features(120.0, 110.0, 160.0)
        cols = list(EXPECTED_KEYS)
        with pytest.raises(PredictionError):
            predict_weight(features, None, cols)  # type: ignore[arg-type]

    def test_returns_positive_float(self) -> None:
        """With a mock pipe that returns log(200), weight should be ~200 kg."""
        from unittest.mock import MagicMock
        mock_pipe = MagicMock()
        mock_pipe.predict.return_value = np.array([np.log(200.0)])

        features = build_features(120.0, 110.0, 160.0)
        cols = list(features.keys())
        weight = predict_weight(features, mock_pipe, cols)

        assert isinstance(weight, float)
        assert weight == pytest.approx(200.0, rel=1e-4)

    def test_column_order_enforced(self) -> None:
        """predict_weight must select columns in the order given by feature_columns."""
        from unittest.mock import MagicMock, call
        mock_pipe = MagicMock()
        mock_pipe.predict.return_value = np.array([5.0])

        features = build_features(120.0, 110.0, 160.0)
        cols = list(features.keys())
        predict_weight(features, mock_pipe, cols)

        # Ensure predict was actually called once
        assert mock_pipe.predict.call_count == 1
