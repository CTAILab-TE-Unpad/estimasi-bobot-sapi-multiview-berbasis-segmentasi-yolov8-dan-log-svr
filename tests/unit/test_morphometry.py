"""Unit tests for cattle_weight.core.morphometry."""
from __future__ import annotations

import numpy as np
import pytest

from cattle_weight.core.morphometry import bridge_scale, ramanujan_girth
from cattle_weight.exceptions import CalibrationError


class TestRamanujanGirth:
    """Tests for the Ramanujan ellipse perimeter approximation."""

    def test_circle_approaches_circumference(self) -> None:
        """For a = b (circle), result should equal 2πr within float tolerance."""
        r = 10.0
        result = ramanujan_girth(r, r)
        assert abs(result - 2 * np.pi * r) < 0.01, (
            f"Expected ~{2 * np.pi * r:.4f}, got {result:.4f}"
        )

    def test_positive_for_positive_inputs(self) -> None:
        assert ramanujan_girth(15.0, 12.0) > 0

    def test_symmetric(self) -> None:
        """Ellipse perimeter is symmetric: girth(a, b) == girth(b, a)."""
        assert abs(ramanujan_girth(15.0, 10.0) - ramanujan_girth(10.0, 15.0)) < 1e-9

    def test_larger_axes_means_larger_girth(self) -> None:
        assert ramanujan_girth(20.0, 15.0) > ramanujan_girth(10.0, 8.0)

    @pytest.mark.parametrize("a,b", [(1.0, 1.0), (50.0, 30.0), (100.0, 80.0)])
    def test_parametrized_positive(self, a: float, b: float) -> None:
        assert ramanujan_girth(a, b) > 0


class TestBridgeScale:
    """Tests for the withers-height bridge calibration method."""

    def test_both_scales_present_returns_dual_sticker(self) -> None:
        s, b, src = bridge_scale(0.1, 0.2, 100.0, 90.0)
        assert s == 0.1
        assert b == 0.2
        assert src == "dual_sticker"

    def test_bridge_from_back_when_side_missing(self) -> None:
        # s_back=0.2, wh ratio = 1.0 → bridged side should equal 0.2
        s_side, s_back, src = bridge_scale(None, 0.2, 100.0, 100.0)
        assert s_side == pytest.approx(0.2)
        assert s_back == 0.2
        assert src == "bridged_from_back_sticker"

    def test_bridge_from_side_when_back_missing(self) -> None:
        s_side, s_back, src = bridge_scale(0.1, None, 100.0, 100.0)
        assert s_back == pytest.approx(0.1)
        assert src == "bridged_from_side_sticker"

    def test_wh_ratio_applied_correctly(self) -> None:
        # back WH = 200 px, side WH = 100 px → bridged side = 0.2 * (200/100) = 0.4
        s_side, _, _ = bridge_scale(None, 0.2, 100.0, 200.0)
        assert s_side == pytest.approx(0.4)

    def test_both_missing_raises_calibration_error(self) -> None:
        with pytest.raises(CalibrationError, match="not detected in either"):
            bridge_scale(None, None, 100.0, 100.0)

    def test_zero_side_wh_raises_when_side_missing(self) -> None:
        with pytest.raises(CalibrationError, match="withers height is zero"):
            bridge_scale(None, 0.2, 0.0, 100.0)

    def test_zero_back_wh_raises_when_back_missing(self) -> None:
        with pytest.raises(CalibrationError, match="withers height is zero"):
            bridge_scale(0.1, None, 100.0, 0.0)
