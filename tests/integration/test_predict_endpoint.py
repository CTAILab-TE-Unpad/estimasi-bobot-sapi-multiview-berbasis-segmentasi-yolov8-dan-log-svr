"""Integration tests for POST /api/v1/predict and system endpoints.

These tests use a FastAPI TestClient with a MockModelRegistry that bypasses
real model loading. Image decoding and request routing are fully exercised,
but YOLO inference and SVR prediction are mocked.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestSystemEndpoints:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_ready_returns_status(self, client: TestClient) -> None:
        response = client.get("/ready")
        assert response.status_code == 200
        body = response.json()
        assert "status" in body
        assert "models_loaded" in body

    def test_docs_accessible(self, client: TestClient) -> None:
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json_accessible(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/api/v1/predict" in schema["paths"]


class TestPredictEndpoint:
    def test_missing_both_files_returns_422(self, client: TestClient) -> None:
        response = client.post("/api/v1/predict")
        assert response.status_code == 422

    def test_missing_back_image_returns_422(self, client: TestClient, sample_jpeg_bytes: bytes) -> None:
        response = client.post(
            "/api/v1/predict",
            files={"side_image": ("side.jpg", sample_jpeg_bytes, "image/jpeg")},
        )
        assert response.status_code == 422

    def test_invalid_image_bytes_returns_400(self, client: TestClient) -> None:
        garbage = b"this-is-not-an-image"
        response = client.post(
            "/api/v1/predict",
            files={
                "side_image": ("side.jpg", garbage, "image/jpeg"),
                "back_image": ("back.jpg", garbage, "image/jpeg"),
            },
        )
        assert response.status_code == 400

    def test_valid_images_return_200_with_schema(
        self, client: TestClient, sample_jpeg_bytes: bytes
    ) -> None:
        """Full happy-path test: two valid images → structured JSON response."""
        response = client.post(
            "/api/v1/predict",
            files={
                "side_image": ("side.jpg", sample_jpeg_bytes, "image/jpeg"),
                "back_image": ("back.jpg", sample_jpeg_bytes, "image/jpeg"),
            },
        )
        assert response.status_code == 200
        body = response.json()

        # Top-level keys
        assert "predicted_weight_kg" in body
        assert "calibration" in body
        assert "measurements" in body
        assert "model_features" in body
        assert "visualization_png_b64" in body

        # Calibration sub-fields
        calib = body["calibration"]
        assert "scale_source" in calib
        assert "scale_side_cm_per_px" in calib
        assert "scale_back_cm_per_px" in calib

        # Measurements sub-fields
        meas = body["measurements"]
        assert "body_length_cm" in meas
        assert "withers_height_cm" in meas
        assert "chest_girth_cm" in meas

        # Weight is a positive number
        assert body["predicted_weight_kg"] > 0

        # Visualization is non-empty base64
        assert len(body["visualization_png_b64"]) > 100

    def test_model_features_have_all_columns(
        self, client: TestClient, sample_jpeg_bytes: bytes
    ) -> None:
        response = client.post(
            "/api/v1/predict",
            files={
                "side_image": ("side.jpg", sample_jpeg_bytes, "image/jpeg"),
                "back_image": ("back.jpg", sample_jpeg_bytes, "image/jpeg"),
            },
        )
        assert response.status_code == 200
        features = response.json()["model_features"]
        expected = {
            "BL_yolov8l", "WH_yolov8l", "CG_yolov8l",
            "BL_WH", "BL_sq", "WH_BL_ratio",
            "CG_sq", "vol_proxy", "log_vol",
        }
        assert set(features.keys()) == expected
