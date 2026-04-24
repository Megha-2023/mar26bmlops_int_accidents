from pathlib import Path
import pickle

import pandas as pd


MODEL_PATH = Path("models") / "xgb_model.pkl"


def load_model():
    """Load the trained model artifact."""
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def generate_predictions(model, features: pd.DataFrame) -> pd.Series:
    """
    Generate prediction labels for a feature DataFrame.

    Args:
        model: Loaded trained model.
        features: Feature matrix.

    Returns:
        Pandas Series named 'prediction'.
    """
    predictions = model.predict(features)
    return pd.Series(predictions, name="prediction", index=features.index)