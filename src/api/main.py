import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import pandas as pd

from src.api.model_loader import load_model as load_api_model
from src.api.schemas import PredictionRequest, PredictionResponse
from src.monitoring.data_loader import load_current_data, load_reference_data
from src.monitoring.evidently_report import (
    generate_data_drift_report,
    generate_prediction_drift_report,
)
from src.monitoring.model_loader import (
    generate_predictions,
    load_model as load_monitoring_model,
)

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Accident Severity Prediction API",
    description="FastAPI service for serving the trained XGBoost accident model.",
    version="1.2.0",
    openapi_tags=[
        {
            "name": "System",
            "description": "Service health and model metadata endpoints."
        },
        {
            "name": "Inference",
            "description": "Endpoints used for accident severity prediction."
        },
        {
            "name": "Monitoring",
            "description": "Endpoints for Evidently monitoring reports."
        }
    ]
)

# Load the API model once when the API starts
model = load_api_model()

# Exact feature order expected by the model
MODEL_COLUMNS = [
    "mois", "jour", "hour", "lum", "int", "atm", "col", "catr",
    "circ", "nbv", "vosp", "surf", "infra", "situ", "lat", "long",
    "place", "catu", "sexe", "locp", "actp", "etatp", "catv", "victim_age"
]


def attach_prediction_column(model_obj, data: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the input data with a prediction column added.
    """
    data_with_predictions = data.copy()
    data_with_predictions["prediction"] = generate_predictions(model_obj, data)
    return data_with_predictions


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
        "predict_endpoint": "/predict",
        "data_drift_report_endpoint": "/monitor/report/data",
        "prediction_drift_report_endpoint": "/monitor/report/prediction",
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
        payload_dict = payload.model_dump(by_alias=True)

        logger.info("Received prediction request: %s", payload_dict)

        data = pd.DataFrame(
            [[payload_dict[col] for col in MODEL_COLUMNS]],
            columns=MODEL_COLUMNS
        )

        proba = model.predict_proba(data)[0]

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
            "Prediction successful: class=%s, confidence=%.4f",
            prediction,
            confidence
        )

        return {
            "prediction": prediction,
            "severity": result["label"],
            "description": result["description"],
            "confidence": confidence,
            "probabilities": probabilities
        }

    except Exception as exc:
        logger.error("Prediction failed: %s", str(exc))
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        ) from exc


@app.get(
    "/monitor/report/{report_type}",
    tags=["Monitoring"],
    summary="Generate and serve an Evidently monitoring report",
    description=(
        "Generates an Evidently HTML report using the reference and current "
        "datasets, then serves the HTML file directly. "
        "Supported report types: 'data' and 'prediction'."
    )
)
def get_monitoring_report(report_type: str):
    try:
        logger.info("Generating monitoring report of type: %s", report_type)

        reference_data = load_reference_data()
        current_data = load_current_data()

        if report_type == "data":
            report_path = generate_data_drift_report(
                reference_data=reference_data,
                current_data=current_data,
                report_name="xtrain_vs_xtest_drift_report.html",
            )

        elif report_type == "prediction":
            monitoring_model = load_monitoring_model()

            reference_with_predictions = attach_prediction_column(
                monitoring_model,
                reference_data
            )
            current_with_predictions = attach_prediction_column(
                monitoring_model,
                current_data
            )

            report_path = generate_prediction_drift_report(
                reference_data=reference_with_predictions,
                current_data=current_with_predictions,
                prediction_column="prediction",
                report_name="xtrain_vs_xtest_prediction_drift_report.html",
            )

        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid report type. Use 'data' or 'prediction'."
            )

        report_path = Path(report_path)

        if not report_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Monitoring report was not generated successfully."
            )

        logger.info("Monitoring report ready: %s", report_path)

        return FileResponse(
            path=report_path,
            media_type="text/html",
            filename=report_path.name,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Monitoring report generation failed: %s", str(exc))
        raise HTTPException(
            status_code=500,
            detail="Internal server error during monitoring report generation"
        ) from exc