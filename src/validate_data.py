import pandas as pd
import os

DATA_PATH = "data"

files = [
    "caracteristiques.csv",
    "lieux.csv",
    "vehicules.csv",
    "usagers.csv"
]

print("Checking datasets...")

for f in files:
    path = os.path.join(DATA_PATH, f)

    if not os.path.exists(path):
        raise Exception(f"Missing file: {f}")

    df = pd.read_csv(path)

    print(f"{f} loaded successfully")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("All datasets validated successfully.")
