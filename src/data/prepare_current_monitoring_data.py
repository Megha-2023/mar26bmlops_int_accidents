from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from src.data.make_dataset import (
    add_victim_age,
    clean_time_column,
    fix_year_column,
    select_features,
)


TRAINING_DATA_PATH = Path("data/accidents_full.csv")
CURRENT_DATA_PATH = Path("data/processed/accidents_2016_2018.csv")
OUTPUT_DIR = Path("data/preprocessed")

X_CURRENT_OUTPUT = OUTPUT_DIR / "X_current_2016_2018.csv"
Y_CURRENT_OUTPUT = OUTPUT_DIR / "y_current_2016_2018.csv"

MISSING_CATEGORY_TOKEN = "__MISSING__"


def build_reference_training_features() -> pd.DataFrame:
    """
    Rebuild the raw training dataset (before encoding/imputation) from the
    original accidents_full.csv file, using the same preprocessing logic
    as the training pipeline.
    """
    df = pd.read_csv(TRAINING_DATA_PATH, low_memory=False)

    df = fix_year_column(df)
    df = df[df["an"].between(2010, 2016)].copy()
    df = clean_time_column(df)
    df = add_victim_age(df)
    df = select_features(df)

    train_df = df[df["an"].between(2010, 2015)].copy()
    train_df = train_df.drop(columns=["an"])

    return train_df


def build_current_raw_features() -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the raw current dataset (before encoding/imputation) from the merged
    2016-2018 BAAC file, using the same preprocessing logic as training.
    """
    df = pd.read_csv(CURRENT_DATA_PATH, low_memory=False)

    df = fix_year_column(df)
    df = clean_time_column(df)
    df = add_victim_age(df)
    df = select_features(df)

    df = df.drop(columns=["an"])

    X_current = df.drop(columns=["grav"]).copy()
    y_current = (df["grav"] - 1).copy()

    return X_current, y_current


def validate_columns(reference_train_df: pd.DataFrame, current_raw: pd.DataFrame) -> None:
    """
    Ensure the current raw feature columns match the training raw feature columns exactly.
    """
    reference_cols = reference_train_df.drop(columns=["grav"]).columns.tolist()
    current_cols = current_raw.columns.tolist()

    if reference_cols != current_cols:
        raise ValueError(
            "Current dataset columns do not match training feature columns.\n"
            f"Training columns: {reference_cols}\n"
            f"Current columns: {current_cols}"
        )


def normalize_categorical_columns(
    df: pd.DataFrame,
    categorical_columns: list[str],
) -> pd.DataFrame:
    """
    Make categorical columns safe for OrdinalEncoder by replacing missing values
    and converting all values to strings.
    """
    df = df.copy()

    for col in categorical_columns:
        df[col] = df[col].fillna(MISSING_CATEGORY_TOKEN).astype(str)

    return df


def fit_preprocessors(
    X_train_raw: pd.DataFrame,
) -> tuple[OrdinalEncoder, SimpleImputer, list[str]]:
    """
    Fit the categorical encoder and imputer on the raw training data only.
    """
    X_train = X_train_raw.copy()

    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    X_train = normalize_categorical_columns(X_train, cat_cols)

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    if cat_cols:
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])

    imputer = SimpleImputer(strategy="most_frequent")
    imputer.fit(X_train)

    return encoder, imputer, cat_cols


def transform_current_data(
    X_current_raw: pd.DataFrame,
    encoder: OrdinalEncoder,
    imputer: SimpleImputer,
    cat_cols: list[str],
) -> pd.DataFrame:
    """
    Apply the fitted encoder and imputer to the current dataset.
    """
    X_current = X_current_raw.copy()
    X_current = normalize_categorical_columns(X_current, cat_cols)

    if cat_cols:
        X_current[cat_cols] = encoder.transform(X_current[cat_cols])

    feature_names = X_current.columns.tolist()
    X_current_imputed = imputer.transform(X_current)
    X_current_imputed = pd.DataFrame(
        X_current_imputed,
        columns=feature_names,
        index=X_current.index,
    )

    return X_current_imputed


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Rebuilding raw training features from accidents_full.csv...")
    train_df = build_reference_training_features()
    X_train_raw = train_df.drop(columns=["grav"]).copy()

    print("Building raw current features from merged 2016-2018 dataset...")
    X_current_raw, y_current = build_current_raw_features()

    print("Validating feature columns...")
    validate_columns(train_df, X_current_raw)

    print("Fitting encoder and imputer on training data...")
    encoder, imputer, cat_cols = fit_preprocessors(X_train_raw)

    print("Transforming current dataset...")
    X_current = transform_current_data(X_current_raw, encoder, imputer, cat_cols)

    X_current.to_csv(X_CURRENT_OUTPUT, index=False)
    pd.DataFrame({"grav": y_current}).to_csv(Y_CURRENT_OUTPUT, index=False)

    print("\nDone.")
    print(f"Saved X_current to: {X_CURRENT_OUTPUT}")
    print(f"Saved y_current to: {Y_CURRENT_OUTPUT}")
    print(f"X_current shape: {X_current.shape}")
    print(f"y_current shape: {y_current.shape}")
    print(
        "Columns match training features: "
        f"{X_current.columns.tolist() == X_train_raw.columns.tolist()}"
    )


if __name__ == "__main__":
    main()