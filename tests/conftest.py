import pytest


@pytest.fixture
def valid_prediction_payload():
    return {
        "mois": 5,
        "jour": 12,
        "hour": 14,
        "lum": 1,
        "int": 1,
        "atm": 1,
        "col": 3,
        "catr": 4,
        "circ": 2,
        "nbv": 2,
        "vosp": 0,
        "surf": 1,
        "infra": 0,
        "situ": 1,
        "lat": 48.8566,
        "long": 2.3522,
        "place": 1,
        "catu": 1,
        "sexe": 1,
        "locp": 0,
        "actp": 0,
        "etatp": 1,
        "catv": 7,
        "victim_age": 35,
    }