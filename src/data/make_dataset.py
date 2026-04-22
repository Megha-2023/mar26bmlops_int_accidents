import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from pathlib import Path

DEFAULT_DATA_PATH = "data/accidents_full.csv"
DEFAULT_OUTPUT_DIR = Path("data/preprocessed")
CHUNK_SIZE = 100_000

FEATURES = [
    "an","mois","jour","hour","lum","int","atm","col","catr","circ","nbv","vosp",
    "surf","infra","situ","lat","long","place","catu","sexe","locp","actp",
    "etatp","catv","victim_age","grav"
]

dtype_dict = {
    "lum":"str","int":"str","atm":"str","col":"str","catr":"str","circ":"str",
    "vosp":"str","surf":"str","infra":"str","situ":"str","place":"str","catu":"str",
    "sexe":"str","locp":"str","actp":"str","etatp":"str","catv":"str"
}

# -----------------------------
# Preprocessing helpers
# -----------------------------

def fix_year(x):
    return x + 2000 if x < 100 else x

def process_chunk(chunk):
    # Fix year
    chunk.loc[:, "an"] = chunk["an"].apply(fix_year)
    
    # Filter years
    chunk = chunk.loc[chunk["an"].between(2010, 2016)].copy()
    
    # Clean time
    chunk["hrmn"] = pd.to_numeric(chunk["hrmn"], errors="coerce")
    chunk["hour"] = (chunk["hrmn"] // 100).fillna(0).astype(int)
    
    # Add victime age
    chunk["victim_age"] = chunk["an"] - chunk["an_nais"]
    chunk = chunk.loc[chunk["victim_age"].between(0, 100)].copy()
    
    # Select features
    return chunk[[col for col in FEATURES if col in chunk.columns]].copy()


# -----------------------------
# Main processing
# -----------------------------
def make_dataset(
        data_path=DEFAULT_DATA_PATH,
        output_path=DEFAULT_OUTPUT_DIR
):
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and processing data in chunks...")
    chunks = []
    for chunk in pd.read_csv(data_path, chunksize=CHUNK_SIZE, dtype=dtype_dict, low_memory=False):
        chunks.append(process_chunk(chunk))

    df = pd.concat(chunks, ignore_index=True)

    train = df[df["an"].between(2010, 2015)]
    test = df[df["an"] == 2016]

    y_train = train["grav"] - 1
    y_test = test["grav"] - 1
    X_train = train.drop(columns=["an", "grav"])
    X_test = test.drop(columns=["an", "grav"])

    # Ensure all categorical columns are strings
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    print("Encoding......")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])

    imputer = SimpleImputer(strategy="most_frequent")
    X_train[:] = imputer.fit_transform(X_train)
    X_test[:] = imputer.transform(X_test)

    print("Saving processed Data to files.......")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print(f"Dataset saved to {output_dir} - train: {X_train.shape}, test: {X_test.shape}")

    # Return summary so DAG can push to XCom
    return {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": len(X_train.columns)
    }

if __name__ == "__main__":
    make_dataset()
