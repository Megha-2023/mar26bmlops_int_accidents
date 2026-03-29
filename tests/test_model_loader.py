import pytest
from pathlib import Path
from src.api import model_loader


# -----------------------------
# Model loading failure behavior
# -----------------------------
# This test verifies that the loader fails clearly when the model file
# does not exist. A FileNotFoundError is the expected behavior here.
def test_load_model_raises_if_file_missing(monkeypatch):
    # Replace the real model path with a fake path that does not exist.
    fake_path = Path("fake/path/model.pkl")
    monkeypatch.setattr(model_loader, "MODEL_PATH", fake_path)

    # The loader should raise FileNotFoundError instead of failing silently.
    with pytest.raises(FileNotFoundError):
        model_loader.load_model()


# -----------------------------
# Model loading success behavior
# -----------------------------
# This test verifies that when the model file exists, the loader returns
# the object produced by joblib.load().
def test_load_model_success(monkeypatch):
    fake_path = Path("models/xgb_model.pkl")

    # Replace the real model path with a controlled test path.
    monkeypatch.setattr(model_loader, "MODEL_PATH", fake_path)

    # Simulate that the file exists, without requiring a real file on disk.
    monkeypatch.setattr(Path, "exists", lambda self: True)

    # Create a fake model object to represent the loaded model.
    fake_model = object()

    # Simulate joblib.load returning the fake model.
    monkeypatch.setattr(model_loader.joblib, "load", lambda path: fake_model)

    # Run the function under test.
    model = model_loader.load_model()

    # The returned object should be exactly the object from joblib.load.
    assert model is fake_model