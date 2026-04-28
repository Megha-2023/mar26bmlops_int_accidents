import pandas as pd
import json
from pathlib import Path

import src.models.train_model as train_module  # adjust if your file name differs


# -----------------------------
# Fixture: tiny dataset
# -----------------------------
def create_fake_data(data_path: Path):
    data_path.mkdir(parents=True, exist_ok=True)

    X_train = pd.DataFrame({
        "f1": [1, 2, 3, 4],
        "f2": [10, 20, 30, 40]
    })

    y_train = pd.Series([0, 1, 0, 1])

    X_train.to_csv(data_path / "X_train.csv", index=False)
    y_train.to_csv(data_path / "y_train.csv", index=False)


# -----------------------------
# Test: training runs end-to-end
# -----------------------------
def test_train_model_runs(tmp_path, monkeypatch):
    data_path = tmp_path / "data"
    model_path = tmp_path / "model" / "xgb.pkl"
    params_path = tmp_path / "params.json"

    create_fake_data(data_path)

    result = train_module.train_model(
        data_path=str(data_path),
        model_path=str(model_path),
        params_path=str(params_path),
        force=True  # IMPORTANT: avoid caching logic
    )

    # -----------------------------
    # Assertions
    # -----------------------------
    assert model_path.exists()
    assert params_path.exists()

    assert result["skipped"] is False
    assert result["features"] == 2
    assert result["train_rows"] == 4


# -----------------------------
# Test: skip logic works
# -----------------------------
def test_model_skips_if_up_to_date(tmp_path):
    data_path = tmp_path / "data"
    model_path = tmp_path / "model" / "xgb.pkl"
    params_path = tmp_path / "params.json"

    create_fake_data(data_path)

    # First run → train model
    train_module.train_model(
        data_path=str(data_path),
        model_path=str(model_path),
        params_path=str(params_path),
        force=True
    )

    # Second run → should skip
    result = train_module.train_model(
        data_path=str(data_path),
        model_path=str(model_path),
        params_path=str(params_path),
        force=False
    )

    assert result["skipped"] is True
    assert "model_path" in result


# -----------------------------
# Test: params override JSON
# -----------------------------
def test_params_override(tmp_path):
    data_path = tmp_path / "data"
    model_path = tmp_path / "model" / "xgb.pkl"
    params_path = tmp_path / "params.json"

    create_fake_data(data_path)

    override_params = {
        "n_estimators": 5,
        "max_depth": 2
    }

    params_path.write_text(json.dumps(override_params))

    result = train_module.train_model(
        data_path=str(data_path),
        model_path=str(model_path),
        params_path=str(params_path),
        force=True
    )

    assert model_path.exists()
    assert result["skipped"] is False
