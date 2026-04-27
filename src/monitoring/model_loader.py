import pandas as pd

from src.model_registry import DEFAULT_MLFLOW_MODEL_URI, load_registered_model


def load_model(model_uri: str = DEFAULT_MLFLOW_MODEL_URI):
    """Load the trained model from MLflow Registry."""
    return load_registered_model(model_uri)


def generate_predictions(model, features: pd.DataFrame) -> pd.Series:
    """
    Generate prediction labels for a feature DataFrame.
    """
    predictions = model.predict(features)
    return pd.Series(predictions, name="prediction", index=features.index)
