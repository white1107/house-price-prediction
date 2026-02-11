"""Tests for FastAPI endpoints."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.main import app, load_model


@pytest.fixture(scope="module")
def client():
    # Ensure model is loaded before tests
    load_model()
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_fields(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_name" in data
        assert "features_count" in data

    def test_health_status_healthy(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "healthy"


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 200

    def test_predict_response_fields(self, client):
        response = client.post("/predict", json={})
        data = response.json()
        assert "predicted_price" in data
        assert "model_name" in data
        assert "model_version" in data

    def test_predict_price_is_positive(self, client):
        response = client.post("/predict", json={})
        assert response.json()["predicted_price"] > 0

    def test_predict_with_custom_features(self, client):
        response = client.post("/predict", json={
            "OverallQual": 10,
            "GrLivArea": 3000,
            "YearBuilt": 2020,
        })
        assert response.status_code == 200
        assert response.json()["predicted_price"] > 0

    def test_high_quality_more_expensive(self, client):
        low = client.post("/predict", json={"OverallQual": 3}).json()["predicted_price"]
        high = client.post("/predict", json={"OverallQual": 9}).json()["predicted_price"]
        assert high > low

    def test_larger_area_more_expensive(self, client):
        small = client.post("/predict", json={"GrLivArea": 800}).json()["predicted_price"]
        large = client.post("/predict", json={"GrLivArea": 3000}).json()["predicted_price"]
        assert large > small


class TestBatchPredictEndpoint:
    def test_batch_returns_200(self, client):
        response = client.post("/predict/batch", json={
            "houses": [{}, {}]
        })
        assert response.status_code == 200

    def test_batch_response_count(self, client):
        response = client.post("/predict/batch", json={
            "houses": [{}, {}, {}]
        })
        data = response.json()
        assert data["count"] == 3
        assert len(data["predictions"]) == 3

    def test_batch_max_limit(self, client):
        houses = [{}] * 101
        response = client.post("/predict/batch", json={"houses": houses})
        assert response.status_code == 400


class TestModelReloadEndpoint:
    def test_reload_returns_200(self, client):
        response = client.post("/model/reload")
        assert response.status_code == 200

    def test_reload_response_fields(self, client):
        response = client.post("/model/reload")
        data = response.json()
        assert data["status"] == "reloaded"
        assert "model_name" in data
        assert "features_count" in data
