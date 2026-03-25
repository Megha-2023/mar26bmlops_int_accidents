import pandas as pd
import os

DATA_FILE = "data/accidents_full.csv"

def main():

    print("Checking dataset...")

    # -----------------------------
    # Check merged dataset exists
    # -----------------------------
    if not os.path.exists(DATA_FILE):
        raise Exception(f"Missing file: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, low_memory=False)

    print("Dataset loaded successfully ")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # -----------------------------
    # Basic validation checks
    # -----------------------------
    required_cols = ["an", "grav"]

    for col in required_cols:
        if col not in df.columns:
            raise Exception(f"Missing required column: {col}")

    print("Required columns exist ")
    print("Data validation passed successfully ")


if __name__ == "__main__":
    main()
