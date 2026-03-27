from fastapi import FastAPI, HTTPException
import pandas as pd

from src.api.schemas import PredictionRequest
from src.api.model_loader import load_model
from src.api.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Accident Severity Prediction API",
    description="FastAPI service for serving the trained XGBoost accident model",
    version="1.0.0",
)

# Load the model once when the API starts
model = load_model()

# Exact feature order expected by the model
MODEL_COLUMNS = [
    "mois", "jour", "hour", "lum", "int", "atm", "col", "catr",
    "circ", "nbv", "vosp", "surf", "infra", "situ", "lat", "long",
    "place", "catu", "sexe", "locp", "actp", "etatp", "catv", "victim_age"
]


@app.get("/health")
def health():
    """
    Simple endpoint to verify that the API is running.
    """
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    """
    Return basic information about the loaded model and expected features.
    """
    return {
        "expected_number_of_features": len(MODEL_COLUMNS),
        "feature_columns": MODEL_COLUMNS,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(payload: PredictionRequest):
    """
    Make a prediction from a JSON body containing the 24 named features.
    """
    try:
        payload_dict = payload.model_dump(by_alias=True)

        data = pd.DataFrame(
            [[payload_dict[col] for col in MODEL_COLUMNS]],
            columns=MODEL_COLUMNS
        )

        # Get probabilities for all classes
        proba = model.predict_proba(data)[0]

        # Predicted class = class with highest probability
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

        return {
            "prediction": prediction,
            "severity": result["label"],
            "description": result["description"],
            "confidence": confidence,
            "probabilities": probabilities
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))