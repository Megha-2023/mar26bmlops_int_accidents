import pytest
from pydantic import ValidationError

from src.api.schemas import PredictionRequest, PredictionResponse


# -----------------------------
# Basic pytest smoke check
# -----------------------------
# This confirms that pytest is running correctly in the project.
# It is not a business-logic test, just an environment sanity check.
def test_pytest_is_working():
    assert 1 + 1 == 2


# -----------------------------
# PredictionRequest: valid input
# -----------------------------
# This test verifies that a fully valid payload is accepted by the schema
# and that key fields are correctly stored on the model instance.
def test_prediction_request_accepts_valid_payload(valid_prediction_payload):
    payload = PredictionRequest(**valid_prediction_payload)

    assert payload.mois == 5
    assert payload.jour == 12
    assert payload.hour == 14
    assert payload.lat == 48.8566
    assert payload.long == 2.3522
    assert payload.victim_age == 35


# -----------------------------
# PredictionRequest: invalid month
# -----------------------------
# The schema should reject months outside the valid calendar range.
def test_prediction_request_rejects_invalid_month(valid_prediction_payload):
    bad_payload = valid_prediction_payload.copy()
    bad_payload["mois"] = 13

    with pytest.raises(ValidationError):
        PredictionRequest(**bad_payload)


# -----------------------------
# PredictionRequest: invalid hour
# -----------------------------
# The schema should reject hours outside the 0-23 range.
def test_prediction_request_rejects_invalid_hour(valid_prediction_payload):
    bad_payload = valid_prediction_payload.copy()
    bad_payload["hour"] = 24

    with pytest.raises(ValidationError):
        PredictionRequest(**bad_payload)


# -----------------------------
# PredictionRequest: invalid latitude
# -----------------------------
# Latitude must remain within the valid geographic range [-90, 90].
def test_prediction_request_rejects_invalid_latitude(valid_prediction_payload):
    bad_payload = valid_prediction_payload.copy()
    bad_payload["lat"] = 120.0

    with pytest.raises(ValidationError):
        PredictionRequest(**bad_payload)


# -----------------------------
# PredictionRequest: invalid longitude
# -----------------------------
# Longitude must remain within the valid geographic range [-180, 180].
def test_prediction_request_rejects_invalid_longitude(valid_prediction_payload):
    bad_payload = valid_prediction_payload.copy()
    bad_payload["long"] = 250.0

    with pytest.raises(ValidationError):
        PredictionRequest(**bad_payload)


# -----------------------------
# PredictionRequest: invalid victim age
# -----------------------------
# The schema should reject ages outside the expected human age range.
def test_prediction_request_rejects_invalid_victim_age(valid_prediction_payload):
    bad_payload = valid_prediction_payload.copy()
    bad_payload["victim_age"] = 140

    with pytest.raises(ValidationError):
        PredictionRequest(**bad_payload)


# -----------------------------
# PredictionRequest: alias support
# -----------------------------
# The external API uses the alias "int", while the internal schema field
# is named "intersection_type". This test ensures alias mapping works.
def test_prediction_request_accepts_int_alias(valid_prediction_payload):
    payload = PredictionRequest(**valid_prediction_payload)

    assert payload.intersection_type == 1


# -----------------------------
# PredictionResponse: valid output
# -----------------------------
# This test verifies that a properly structured prediction response is
# accepted by the response schema.
def test_prediction_response_accepts_valid_payload():
    response = PredictionResponse(
        prediction=2,
        severity="Serious injury",
        description="Predicted as an accident with serious injuries.",
        confidence=0.82,
        probabilities={
            "no_injury_minor": 0.05,
            "slight_injury": 0.10,
            "serious_injury": 0.82,
            "fatal": 0.03,
        },
    )

    assert response.prediction == 2
    assert response.severity == "Serious injury"
    assert response.confidence == 0.82


# -----------------------------
# PredictionResponse: invalid confidence
# -----------------------------
# Confidence should stay within [0.0, 1.0]. This test ensures invalid
# confidence values are rejected by the schema.
def test_prediction_response_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        PredictionResponse(
            prediction=2,
            severity="Serious injury",
            description="Predicted as an accident with serious injuries.",
            confidence=1.5,
            probabilities={
                "no_injury_minor": 0.05,
                "slight_injury": 0.10,
                "serious_injury": 0.82,
                "fatal": 0.03,
            },
        )