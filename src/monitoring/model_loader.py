import pandas as pd

from src.model_registry import load_local_model


def load_model():
    """Load the trained model from MLflow Registry."""
    return load_local_model()


def generate_predictions(model, features: pd.DataFrame) -> pd.Series:
    """
    Generate prediction labels for a feature DataFrame.
    """
    predictions = model.predict(features)
    return pd.Series(predictions, name="prediction", index=features.index)
