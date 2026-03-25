import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from pathlib import Path

DATA_PATH = "data/accidents_full.csv"

def main():

    # -----------------------------
    # 0. Load data
    # -----------------------------
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # -----------------------------
    # 1. FIX YEAR COLUMN (mixed encoding)
    # -----------------------------
    df["an"] = df["an"].apply(lambda x: x + 2000 if x < 100 else x)

    # -----------------------------
    # 2. FILTER YEARS
    # -----------------------------
    df = df[df["an"].between(2010, 2016)]

    # -----------------------------
    # 3. CLEAN TIME COLUMN (IMPORTANT FIX)
    # -----------------------------
    df["hrmn"] = pd.to_numeric(df["hrmn"], errors="coerce")
    df["hour"] = (df["hrmn"] // 100).fillna(0).astype(int)

    # -----------------------------
    # 4. FEATURE ENGINEERING
    # -----------------------------
    df["victim_age"] = df["an"] - df["an_nais"]
    df = df[df["victim_age"].between(0, 100)]

    # -----------------------------
    # 5. FINAL FEATURE SET (REAL DATA ONLY)
    # -----------------------------
    features = [
        # time / characteristics
        "an", "mois", "jour", "hour",
        "lum", "int", "atm", "col",
        "catr", "circ", "nbv", "vosp",
        "surf", "infra", "situ",
        "lat", "long",

        # user features
        "place", "catu", "sexe",
        "locp", "actp", "etatp",

        # vehicle
        "catv",

        # engineered
        "victim_age",

        # target
        "grav"
    ]

    df = df[[col for col in features if col in df.columns]]

    # -----------------------------
    # 6. TRAIN / TEST SPLIT
    # -----------------------------
    train = df[df["an"].between(2010, 2015)]
    test  = df[df["an"] == 2016]

    train = train.drop(columns=["an"])
    test = test.drop(columns=["an"])

    X_train = train.drop(columns=["grav"])
    y_train = train["grav"]-1

    X_test = test.drop(columns=["grav"])
    y_test = test["grav"]-1

    # -----------------------------
    # 7. SAFETY CHECK (IMPORTANT)
    # -----------------------------
    if X_train.shape[0] == 0:
        raise ValueError("X_train is empty. Check filtering or data quality.")

    # -----------------------------
    # 8. ENCODING
    # -----------------------------
    cat_cols = X_train.select_dtypes(include="object").columns

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])

    # -----------------------------
    # 9. MISSING VALUES
    # -----------------------------
    imputer = SimpleImputer(strategy="most_frequent")

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # -----------------------------
    # 10. SAVE OUTPUTS
    # -----------------------------
    OUTPUT_DIR = Path("data/preprocessed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train).to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    print("Dataset prepared successfully ")


if __name__ == "__main__":
    main()
