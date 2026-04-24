import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from pathlib import Path

DEFAULT_DATA_PATH = "data/accidents_full.csv"
DEFAULT_OUTPUT_DIR = Path("data/preprocessed")
CHUNK_SIZE = 100_000
REQUIRED_SOURCE_COLUMNS = [
    "an", "mois", "jour", "hrmn", "lum", "int", "atm", "col", "catr", "circ",
    "nbv", "vosp", "surf", "infra", "situ", "lat", "long", "place", "catu",
    "sexe", "locp", "actp", "etatp", "catv", "an_nais", "grav"
]

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


def count_csv_rows(path):
    with open(path, "r", encoding="utf-8") as file:
        return max(sum(1 for _ in file) - 1, 0)


def outputs_are_up_to_date(data_path, output_dir):
    expected_files = [
        output_dir / "X_train.csv",
        output_dir / "X_test.csv",
        output_dir / "y_train.csv",
        output_dir / "y_test.csv",
    ]
    if not all(path.exists() for path in expected_files):
        return False

    source_mtime = Path(data_path).stat().st_mtime
    return all(path.stat().st_mtime >= source_mtime for path in expected_files)

def process_chunk(chunk):
    # Fix year
    chunk["an"] = pd.to_numeric(chunk["an"], errors="coerce")
    chunk["an"] = chunk["an"].where(chunk["an"] >= 100, chunk["an"] + 2000)
    
    # Filter years
    chunk = chunk.loc[chunk["an"].between(2010, 2016)].copy()
    
    # Clean time
    chunk["hrmn"] = pd.to_numeric(chunk["hrmn"], errors="coerce")
    chunk["hour"] = (chunk["hrmn"] // 100).fillna(0).astype(int)
    
    # Add victime age
    chunk["an_nais"] = pd.to_numeric(chunk["an_nais"], errors="coerce")
    chunk["victim_age"] = chunk["an"] - chunk["an_nais"]
    chunk = chunk.loc[chunk["victim_age"].between(0, 100)].copy()
    
    # Select features
    return chunk[[col for col in FEATURES if col in chunk.columns]].copy()


# -----------------------------
# Main processing
# -----------------------------
def make_dataset(
        data_path=DEFAULT_DATA_PATH,
        output_path=DEFAULT_OUTPUT_DIR,
        force=False
):
    data_path = Path(data_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not force and outputs_are_up_to_date(data_path, output_dir):
        print(f"Processed dataset already exists in {output_dir}. Skipping preprocessing.")
        X_train_sample = pd.read_csv(output_dir / "X_train.csv", nrows=1)
        return {
            "train_rows": count_csv_rows(output_dir / "X_train.csv"),
            "test_rows": count_csv_rows(output_dir / "X_test.csv"),
            "features": len(X_train_sample.columns),
            "skipped": True,
        }

    print("Loading and processing data in chunks...")
    train_chunks = []
    test_chunks = []
    for chunk in pd.read_csv(
        data_path,
        usecols=REQUIRED_SOURCE_COLUMNS,
        chunksize=CHUNK_SIZE,
        dtype=dtype_dict,
        low_memory=False
    ):
        processed = process_chunk(chunk)
        train_chunks.append(processed.loc[processed["an"].between(2010, 2015)])
        test_chunks.append(processed.loc[processed["an"] == 2016])

    train = pd.concat(train_chunks, ignore_index=True)
    test = pd.concat(test_chunks, ignore_index=True)

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
        "features": len(X_train.columns),
        "skipped": False,
    }

if __name__ == "__main__":
    make_dataset()
