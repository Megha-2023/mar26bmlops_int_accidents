from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/preprocessed")
REFERENCE_DATA_PATH = DATA_DIR / "X_train.csv"
CURRENT_DATA_PATH = DATA_DIR / "X_current_2016_2018.csv"


def load_reference_data() -> pd.DataFrame:
    """Load the reference dataset used as monitoring baseline."""
    return pd.read_csv(REFERENCE_DATA_PATH)


def load_current_data() -> pd.DataFrame:
    """Load the current dataset used for drift comparison."""
    if not CURRENT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Current monitoring dataset not found: {CURRENT_DATA_PATH}"
        )
    return pd.read_csv(CURRENT_DATA_PATH)