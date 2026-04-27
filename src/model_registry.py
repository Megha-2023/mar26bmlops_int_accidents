import os

import mlflow
import mlflow.xgboost


DEFAULT_MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)
DEFAULT_MLFLOW_MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI",
    "models:/accident_prediction@challenger",
)


def load_registered_model(model_uri: str = DEFAULT_MLFLOW_MODEL_URI):
    """Load the model from MLflow using one explicit registry URI."""
    mlflow.set_tracking_uri(DEFAULT_MLFLOW_TRACKING_URI)

    try:
        return mlflow.xgboost.load_model(model_uri)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from MLflow Registry at '{model_uri}'."
        ) from exc
