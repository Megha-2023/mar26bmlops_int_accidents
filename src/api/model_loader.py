from src.model_registry import DEFAULT_MLFLOW_MODEL_URI, load_registered_model


def load_model(model_uri: str = DEFAULT_MLFLOW_MODEL_URI):
    """Load the trained model from MLflow Registry."""
    return load_registered_model(model_uri)
