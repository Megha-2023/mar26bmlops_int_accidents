import importlib
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------
# Test doubles used by API tests
# ---------------------------------
class DummyModel:
    """Simple fake model returning a fixed probability distribution."""

    def predict_proba(self, data):
        return np.array([[0.05, 0.10, 0.82, 0.03]])


class FailingModel:
    """Fake model that raises an error during prediction."""

    def predict_proba(self, data):
        raise RuntimeError("model prediction failed")


# ---------------------------------
# Shared app/client factory
# ---------------------------------
def build_test_client(monkeypatch, fake_model):
    """
    Patch model loading before importing the FastAPI app, then return
    a TestClient bound to the patched app instance.
    """
    import src.api.model_loader as model_loader

    monkeypatch.setattr(model_loader, "load_model", lambda: fake_model)

    # Force a fresh import of src.api.main so the patched loader is used
    if "src.api.main" in sys.modules:
        del sys.modules["src.api.main"]

    import src.api.main as main
    importlib.reload(main)

    return TestClient(main.app), main


# ---------------------------------
# Root endpoint
# ---------------------------------
def test_root_endpoint_returns_expected_metadata(monkeypatch):
    client, _ = build_test_client(monkeypatch, DummyModel())

    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Welcome to the Accident Severity Prediction API"
    assert body["docs_url"] == "/docs"
    assert body["health_endpoint"] == "/health"
    assert body["model_info_endpoint"] == "/model-info"
    assert body["predict_endpoint"] == "/predict"


# ---------------------------------
# Health endpoint
# ---------------------------------
def test_health_endpoint_returns_ok(monkeypatch):
    client, _ = build_test_client(monkeypatch, DummyModel())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------
# Model info endpoint
# ---------------------------------
def test_model_info_returns_expected_structure(monkeypatch):
    client, main = build_test_client(monkeypatch, DummyModel())

    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()

    assert body["expected_number_of_features"] == 24
    assert body["feature_columns"] == main.MODEL_COLUMNS
    assert len(body["feature_columns"]) == 24


# ---------------------------------
# Predict endpoint: success path
# ---------------------------------
def test_predict_returns_expected_prediction(monkeypatch, valid_prediction_payload):
    client, _ = build_test_client(monkeypatch, DummyModel())

    response = client.post("/predict", json=valid_prediction_payload)

    assert response.status_code == 200
    body = response.json()

    assert body["prediction"] == 2
    assert body["severity"] == "Serious injury"
    assert body["description"] == "Predicted as an accident with serious injuries."
    assert body["confidence"] == 0.82
    assert body["probabilities"] == {
        "no_injury_minor": 0.05,
        "slight_injury": 0.1,
        "serious_injury": 0.82,
        "fatal": 0.03,
    }


# ---------------------------------
# Predict endpoint: schema rejection
# ---------------------------------
def test_predict_rejects_invalid_payload(monkeypatch, valid_prediction_payload):
    client, _ = build_test_client(monkeypatch, DummyModel())

    bad_payload = valid_prediction_payload.copy()
    bad_payload["mois"] = 13

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422


# ---------------------------------
# Predict endpoint: internal failure
# ---------------------------------
def test_predict_returns_500_when_model_fails(monkeypatch, valid_prediction_payload):
    client, _ = build_test_client(monkeypatch, FailingModel())

    response = client.post("/predict", json=valid_prediction_payload)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error during prediction"}