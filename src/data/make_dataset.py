import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from pathlib import Path

DATA_PATH = "data/accidents_full.csv"
OUTPUT_DIR = Path("data/preprocessed")

FEATURES = [
    "an",
    "mois",
    "jour",
    "hour",
    "lum",
    "int",
    "atm",
    "col",
    "catr",
    "circ",
    "nbv",
    "vosp",
    "surf",
    "infra",
    "situ",
    "lat",
    "long",
    "place",
    "catu",
    "sexe",
    "locp",
    "actp",
    "etatp",
    "catv",
    "victim_age",
    "grav",
]


# -----------------------------
# Preprocessing helpers
# -----------------------------
def fix_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert two-digit years like 16 into 2016."""
    df = df.copy()
    df["an"] = df["an"].apply(lambda x: x + 2000 if x < 100 else x)
    return df


def filter_years(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only accidents from 2010 to 2016 inclusive."""
    return df[df["an"].between(2010, 2016)].copy()


def clean_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert hrmn to numeric hour values."""
    df = df.copy()
    df["hrmn"] = pd.to_numeric(df["hrmn"], errors="coerce")
    df["hour"] = (df["hrmn"] // 100).fillna(0).astype(int)
    return df


def add_victim_age(df: pd.DataFrame) -> pd.DataFrame:
    """Create victim_age and keep only realistic ages."""
    df = df.copy()
    df["victim_age"] = df["an"] - df["an_nais"]
    df = df[df["victim_age"].between(0, 100)]
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only expected feature columns that exist in the dataset."""
    return df[[col for col in FEATURES if col in df.columns]].copy()


def split_train_test(df: pd.DataFrame):
    """
    Split data by year:
    - train: 2010 to 2015
    - test: 2016
    Also remove 'an' and separate target 'grav'.
    """
    train = df[df["an"].between(2010, 2015)].copy()
    test = df[df["an"] == 2016].copy()

    train = train.drop(columns=["an"])
    test = test.drop(columns=["an"])

    X_train = train.drop(columns=["grav"])
    y_train = train["grav"] - 1
    X_test = test.drop(columns=["grav"])
    y_test = test["grav"] - 1

    if X_train.shape[0] == 0:
        raise ValueError("X_train is empty. Check filtering or data quality.")

    return X_train, X_test, y_train, y_test


def encode_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Ordinal-encode object columns."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    cat_cols = X_train.select_dtypes(include="object").columns
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])

    return X_train, X_test


def impute_missing_values(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Impute missing values with the most frequent value."""
    imputer = SimpleImputer(strategy="most_frequent")

    feature_names = X_train.columns.tolist()
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    X_train_imputed = pd.DataFrame(X_train_imputed, columns=feature_names)
    X_test_imputed = pd.DataFrame(X_test_imputed, columns=feature_names)

    return X_train_imputed, X_test_imputed


def save_outputs(X_train, X_test, y_train, y_test, output_dir=OUTPUT_DIR):
    """Save processed train/test datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv(output_dir / "y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv(output_dir / "y_test.csv", index=False)


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    df = fix_year_column(df)
    df = filter_years(df)
    df = clean_time_column(df)
    df = add_victim_age(df)
    df = select_features(df)

    X_train, X_test, y_train, y_test = split_train_test(df)
    X_train, X_test = encode_categoricals(X_train, X_test)
    X_train, X_test = impute_missing_values(X_train, X_test)

    save_outputs(X_train, X_test, y_train, y_test)
    print("Dataset prepared successfully")


if __name__ == "__main__":
    main()