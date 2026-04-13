from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/preprocessed")


def load_reference_data() -> pd.DataFrame:
    """Load the reference dataset used as monitoring baseline."""
    return pd.read_csv(DATA_DIR / "X_train.csv")


def load_current_data() -> pd.DataFrame:
    """Load the current dataset used for drift comparison."""
    return pd.read_csv(DATA_DIR / "X_test.csv")