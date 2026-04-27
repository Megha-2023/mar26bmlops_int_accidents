import os
from pathlib import Path

import pandas as pd


DEFAULT_REFERENCE_DATA_PATH = os.getenv(
    "MONITORING_REFERENCE_DATA_PATH",
    "data/preprocessed/X_train.csv",
)
DEFAULT_CURRENT_DATA_PATH = os.getenv(
    "MONITORING_CURRENT_DATA_PATH",
    "data/processed/accidents_2016_2018.csv",
)
MODEL_FEATURES = [
    "mois", "jour", "hour", "lum", "int", "atm", "col", "catr", "circ", "nbv",
    "vosp", "surf", "infra", "situ", "lat", "long", "place", "catu", "sexe",
    "locp", "actp", "etatp", "catv", "victim_age"
]


def load_dataset(dataset_path: str, dataset_label: str) -> pd.DataFrame:
    """Load a monitoring dataset and fail with a clear error if it is missing."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_label} monitoring dataset not found: {path}"
        )
    return pd.read_csv(path)


def load_reference_data(
    data_path: str = DEFAULT_REFERENCE_DATA_PATH,
) -> pd.DataFrame:
    """Load the reference dataset used as monitoring baseline."""
    data = load_dataset(data_path, "Reference")
    return data[MODEL_FEATURES]


def load_current_data(
    data_path: str = DEFAULT_CURRENT_DATA_PATH,
) -> pd.DataFrame:
    """Load the current dataset used for drift comparison."""
    data = load_dataset(data_path, "Current")

    if "hour" not in data.columns:
        if "hrmn" not in data.columns:
            raise ValueError("Current dataset must contain 'hrmn' to derive 'hour'.")
        data["hrmn"] = pd.to_numeric(data["hrmn"], errors="coerce")
        data["hour"] = (data["hrmn"] // 100).fillna(0).astype(int)

    if "victim_age" not in data.columns:
        required_columns = {"an", "an_nais"}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise ValueError(
                "Current dataset must contain 'an' and 'an_nais' to derive "
                f"'victim_age'. Missing: {sorted(missing_columns)}"
            )
        data["an"] = pd.to_numeric(data["an"], errors="coerce")
        data["an"] = data["an"].where(data["an"] >= 100, data["an"] + 2000)
        data["an_nais"] = pd.to_numeric(data["an_nais"], errors="coerce")
        data["victim_age"] = data["an"] - data["an_nais"]

    missing_features = [column for column in MODEL_FEATURES if column not in data.columns]
    if missing_features:
        raise ValueError(
            f"Current dataset is missing required monitoring features: {missing_features}"
        )

    return data[MODEL_FEATURES]
