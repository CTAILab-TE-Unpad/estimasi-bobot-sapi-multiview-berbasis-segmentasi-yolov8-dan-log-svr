"""Shared pytest fixtures and test utilities.

Fixtures defined here are automatically available to all tests without
explicit imports (pytest conftest.py convention).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cattle_weight.api.app import create_app
from cattle_weight.infrastructure.model_registry import ModelRegistry
from cattle_weight.settings import Settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "BL_yolov8l", "WH_yolov8l", "CG_yolov8l",
    "BL_WH", "BL_sq", "WH_BL_ratio",
    "CG_sq", "vol_proxy", "log_vol",
]


# ---------------------------------------------------------------------------
# Mock Model Registry
# ---------------------------------------------------------------------------


class MockModelRegistry(ModelRegistry):
    """ModelRegistry that bypasses real model files — for unit and integration tests.

    - ``seg_model``: Returns a mask dynamically sized to the input image (covers 80%).
    - ``sticker_model``: Returns a 50×50 px sticker mask → scale ≈ 0.05 cm/px.
    - ``svr_pipe``: Returns log(200) → ~200 kg prediction.
    """

    def __init__(self) -> None:
        super().__init__()
        self._loaded = True
        self._meta = {"feature_columns": FEATURE_COLUMNS}

        # SVR mock: log(200) → ~200 kg
        self._svr_pipe = MagicMock()
        self._svr_pipe.predict.return_value = np.array([np.log(200.0)])

        # Segmentation model: dynamically-sized mask to match actual decoded image
        def _seg(img_arr: np.ndarray, verbose: bool = False) -> list:
            h, w = img_arr.shape[:2]
            mask_np = np.zeros((h, w), dtype=np.float32)
            mask_np[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)] = 1.0
            m = MagicMock()
            m.data = [MagicMock()]
            m.data[0].cpu.return_value.numpy.return_value = mask_np
            r = MagicMock()
            r.masks = m
            return [r]

        self._seg_model = MagicMock(side_effect=_seg)

        # Sticker model: 50×50 sticker in top-right corner
        # diameter = (50 + 50) / 2 = 50 px → scale = 2.5 / 50 = 0.05 cm/px
        def _sticker(img_arr: np.ndarray, conf: float = 0.25, verbose: bool = False) -> list:
            h, w = img_arr.shape[:2]
            s_np = np.zeros((h, w), dtype=np.float32)
            x0 = max(0, w - 60)
            x1 = max(0, w - 10)
            s_np[10:60, x0:x1] = 1.0
            m = MagicMock()
            m.data = [MagicMock()]
            m.data[0].cpu.return_value.numpy.return_value = s_np
            r = MagicMock()
            r.masks = m
            return [r]

        self._sticker_model = MagicMock(side_effect=_sticker)

    def load(self, settings: Any) -> None:
        """No-op: mock registry is pre-loaded."""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_settings() -> Settings:
    """Minimal Settings for testing — no real .env file needed."""
    return Settings(models_dir="models", log_level="WARNING")


@pytest.fixture()
def mock_registry() -> MockModelRegistry:
    """A fully pre-loaded MockModelRegistry."""
    return MockModelRegistry()


@pytest.fixture()
def sample_bgr_image() -> np.ndarray:
    """480x640 BGR image with a white rectangle simulating a cattle silhouette."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (80, 60), (560, 420), (200, 200, 200), -1)
    return img


@pytest.fixture()
def sample_jpeg_bytes(sample_bgr_image: np.ndarray) -> bytes:
    """JPEG bytes of the sample BGR image — for multipart upload tests."""
    _, buf = cv2.imencode(".jpg", sample_bgr_image)
    return buf.tobytes()


@pytest.fixture()
def client(test_settings: Settings, mock_registry: MockModelRegistry) -> TestClient:
    """FastAPI TestClient with mocked model registry injected via dependency override."""
    from cattle_weight.api.dependencies import get_registry

    app = create_app(test_settings)
    app.state.registry = mock_registry
    app.dependency_overrides[get_registry] = lambda: mock_registry

    return TestClient(app)
