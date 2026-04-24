import pandas as pd
import os
from pathlib import Path


# ── Validate raw data before preprocessing ──
def validate_raw(data_path="data/accidents_full.csv"):
    print(f"Validating raw data at: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file missing: {data_path}")

    df = pd.read_csv(data_path, low_memory=False, nrows=1000)  # sample only, file is large

    required_cols = ["an", "grav", "hrmn", "an_nais"]
    missing_cols  = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Raw data missing required columns: {missing_cols}")

    print(f"Raw file OK — columns present: {list(df.columns[:10])} ...")
    print("Raw data validation passed.")
    return {"status": "ok", "columns": len(df.columns)}


# ── Validate preprocessed splits before training ─
def validate_preprocessed(data_path="data/preprocessed"):
    print(f"Validating preprocessed data at: {data_path}")
    data_path = Path(data_path)

    # 1. All 4 files must exist
    required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    missing = [f for f in required_files if not (data_path / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing preprocessed files: {missing}")

    # 2. Load all splits
    X_train = pd.read_csv(data_path / "X_train.csv")
    X_test  = pd.read_csv(data_path / "X_test.csv")
    y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
    y_test  = pd.read_csv(data_path / "y_test.csv").squeeze()

    # 3. Shapes must be consistent
    if len(X_train) != len(y_train):
        raise ValueError(f"X_train rows ({len(X_train)}) != y_train rows ({len(y_train)})")
    if len(X_test) != len(y_test):
        raise ValueError(f"X_test rows ({len(X_test)}) != y_test rows ({len(y_test)})")

    # 4. Feature columns must match between train and test
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("X_train and X_test have different columns")

    # 5. No nulls allowed after preprocessing
    null_train = X_train.isnull().sum().sum()
    null_test  = X_test.isnull().sum().sum()
    if null_train > 0:
        raise ValueError(f"X_train has {null_train} null values after preprocessing")
    if null_test > 0:
        raise ValueError(f"X_test has {null_test} null values after preprocessing")

    # 6. Class balance check — warn if severely imbalanced
    class_counts = y_train.value_counts(normalize=True)
    for cls, pct in class_counts.items():
        if pct < 0.02:
            print(f"WARNING: class {cls} has only {pct:.1%} of training samples")

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train classes: {dict(y_train.value_counts())}")
    print("Preprocessed data validation passed.")

    return {
        "train_rows":  len(X_train),
        "test_rows":   len(X_test),
        "features":    len(X_train.columns),
        "train_classes": y_train.nunique()
    }


# ── Run both when called directly
if __name__ == "__main__":
    validate_raw()
    validate_preprocessed()