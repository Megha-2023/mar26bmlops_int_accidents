import logging

from fastapi import FastAPI, HTTPException
import pandas as pd

from src.api.schemas import PredictionRequest, PredictionResponse
from src.api.model_loader import load_model

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Accident Severity Prediction API",
    description="FastAPI service for serving the trained XGBoost accident model.",
    version="1.1.0",
    openapi_tags=[
        {
            "name": "System",
            "description": "Service health and model metadata endpoints."
        },
        {
            "name": "Inference",
            "description": "Endpoints used for accident severity prediction."
        }
    ]
)

# Load the model once when the API starts
model = load_model()

# Exact feature order expected by the model
MODEL_COLUMNS = [
    "mois", "jour", "hour", "lum", "int", "atm", "col", "catr",
    "circ", "nbv", "vosp", "surf", "infra", "situ", "lat", "long",
    "place", "catu", "sexe", "locp", "actp", "etatp", "catv", "victim_age"
]


@app.get(
    "/",
    tags=["System"],
    summary="Root endpoint",
    description="Welcome endpoint that provides a quick overview of the API."
)
def root():
    return {
        "message": "Welcome to the Accident Severity Prediction API",
        "docs_url": "/docs",
        "health_endpoint": "/health",
        "model_info_endpoint": "/model-info",
        "predict_endpoint": "/predict"
    }


@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    description="Returns the running status of the API service."
)
def health():
    return {"status": "ok"}


@app.get(
    "/model-info",
    tags=["System"],
    summary="Model metadata",
    description="Returns the expected input features and metadata for the loaded model."
)
def model_info():
    return {
        "model_type": str(type(model)),
        "expected_number_of_features": len(MODEL_COLUMNS),
        "feature_columns": MODEL_COLUMNS,
    }


@app.post(
    "/predict",
    tags=["Inference"],
    summary="Predict accident severity",
    description="Takes accident-related features as input and returns the predicted severity class, confidence score, and class probabilities.",
    response_model=PredictionResponse
)
def predict(payload: PredictionRequest):
    try:
        # Export request body using aliases so that
        # 'intersection_type' becomes 'int'
        payload_dict = payload.model_dump(by_alias=True)

        logger.info(f"Received prediction request: {payload_dict}")

        # Build a one-row DataFrame with the exact feature order expected by the model
        data = pd.DataFrame(
            [[payload_dict[col] for col in MODEL_COLUMNS]],
            columns=MODEL_COLUMNS
        )

        # Predict probabilities for all classes
        proba = model.predict_proba(data)[0]

        # Predicted class = index of max probability
        prediction = int(proba.argmax())
        confidence = float(max(proba))

        severity_map = {
            0: {
                "label": "No injury / minor",
                "description": "Predicted as the least severe accident class."
            },
            1: {
                "label": "Slight injury",
                "description": "Predicted as an accident with slight injuries."
            },
            2: {
                "label": "Serious injury",
                "description": "Predicted as an accident with serious injuries."
            },
            3: {
                "label": "Fatal",
                "description": "Predicted as the most severe accident class."
            }
        }

        result = severity_map.get(
            prediction,
            {
                "label": "Unknown",
                "description": "Unknown prediction class."
            }
        )

        probabilities = {
            "no_injury_minor": float(proba[0]),
            "slight_injury": float(proba[1]),
            "serious_injury": float(proba[2]),
            "fatal": float(proba[3]),
        }

        logger.info(
            f"Prediction successful: class={prediction}, confidence={confidence:.4f}"
        )

        return {
            "prediction": prediction,
            "severity": result["label"],
            "description": result["description"],
            "confidence": confidence,
            "probabilities": probabilities
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )