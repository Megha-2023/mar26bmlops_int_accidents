from pathlib import Path
import pickle

import pandas as pd


MODEL_PATH = Path("models") / "xgb_model.pkl"


def load_model():
    """Load the trained model artifact."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def generate_predictions(model, features: pd.DataFrame) -> pd.Series:
    """Generate prediction labels for a feature DataFrame."""
    predictions = model.predict(features)
    return pd.Series(predictions, name="prediction")