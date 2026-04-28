import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api.main as api_module
from src.api.main import app


# -----------------------------
# FIX: mock heavy dependencies
# -----------------------------
@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    """
    Prevent real model loading + monitoring logic in CI/Docker.
    """

    # -------------------------
    # Fake model
    # -------------------------
    class FakeModel:
        def predict_proba(self, X):
            # deterministic probability output (4 classes)
            return np.array([[0.1, 0.2, 0.3, 0.4]])

    monkeypatch.setattr(api_module, "load_local_model", lambda: FakeModel())

    # monitoring model
    monkeypatch.setattr(api_module, "load_monitoring_model", lambda: FakeModel())

    # fake predictions
    monkeypatch.setattr(
        api_module,
        "generate_predictions",
        lambda model, data: [1] * len(data)
    )

    # fake dataset loaders (avoid /opt/airflow etc.)
    monkeypatch.setattr(api_module, "load_reference_data", lambda: pd.DataFrame({
        "f1": [1, 2],
        "f2": [3, 4]
    }))

    monkeypatch.setattr(api_module, "load_current_data", lambda: pd.DataFrame({
        "f1": [5, 6],
        "f2": [7, 8]
    }))


client = TestClient(app)


# -----------------------------
# Root endpoint
# -----------------------------
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "predict_endpoint" in response.json()


# -----------------------------
# Health check
# -----------------------------
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# -----------------------------
# Model info
# -----------------------------
def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "feature_columns" in data
    assert data["expected_number_of_features"] == 24


# -----------------------------
# Predict endpoint
# -----------------------------
def test_predict():
    payload = {
        "mois": 1, "jour": 2, "hour": 12, "lum": 1, "int": 1,
        "atm": 1, "col": 1, "catr": 1, "circ": 1, "nbv": 2,
        "vosp": 0, "surf": 1, "infra": 0, "situ": 1, "lat": 45.0,
        "long": 5.0, "place": 1, "catu": 1, "sexe": 1, "locp": 0,
        "actp": 0, "etatp": 0, "catv": 1, "victim_age": 30
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert isinstance(data["prediction"], int)


# -----------------------------
# Invalid report type
# -----------------------------
def test_invalid_monitoring_report():
    response = client.get("/monitor/report/invalid")
    assert response.status_code == 400


# -----------------------------
# Monitoring report (mocked)
# -----------------------------
def test_monitoring_data_report(tmp_path, monkeypatch):
    fake_report = tmp_path / "report.html"
    fake_report.write_text("<html>ok</html>")

    monkeypatch.setattr(
        "src.monitoring.evidently_report.generate_data_drift_report",
        lambda **kwargs: str(fake_report)
    )

    response = client.get("/monitor/report/data")
    assert response.status_code == 200


def test_monitoring_prediction_report(tmp_path, monkeypatch):
    fake_report = tmp_path / "report.html"
    fake_report.write_text("<html>ok</html>")

    monkeypatch.setattr(
        "src.monitoring.evidently_report.generate_prediction_drift_report",
        lambda **kwargs: str(fake_report)
    )

    response = client.get("/monitor/report/prediction")
    assert response.status_code == 200