from pathlib import Path
import joblib

# Go from src/api/model_loader.py up to the project root
BASE_DIR = Path(__file__).resolve().parents[2]

# Path to the trained model
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"


def load_model():
    """
    Load the trained model from disk.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    return joblib.load(MODEL_PATH)