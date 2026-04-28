import pandas as pd
from xgboost import XGBClassifier
import joblib
import json
import os
from pathlib import Path


# DATA_DIR = Path("/opt/airflow")

DEFAULT_DATA_PATH = "data/preprocessed"
DEFAULT_MODEL_PATH = "models/xgb_model.pkl"
DEFAULT_PARAM_PATH = "params.json"


def model_is_up_to_date(model_path, training_files):
    if not model_path.exists():
        return False

    model_mtime = model_path.stat().st_mtime
    return all(path.exists() and model_mtime >= path.stat().st_mtime for path in training_files)


def count_csv_rows(path):
    with open(path, "r", encoding="utf-8") as file:
        return max(sum(1 for _ in file) - 1, 0)


def train_model(
        data_path=DEFAULT_DATA_PATH,
        model_path=DEFAULT_MODEL_PATH,
        params_path=DEFAULT_PARAM_PATH,
        force=False
):

    data_path = Path(data_path)
    model_path = Path(model_path)
    params_path = Path(params_path)
    training_files = [data_path / "X_train.csv", data_path / "y_train.csv"]

    if not force and model_is_up_to_date(model_path, training_files):
        print(f"Model already exists at {model_path}. Skipping training.")
        X_train_sample = pd.read_csv(data_path / "X_train.csv", nrows=1)
        return {
            "model_path": str(model_path),
            "params_path": str(params_path),
            "train_rows": count_csv_rows(data_path / "X_train.csv"),
            "features": len(X_train_sample.columns),
            "skipped": True,
        }

    # -----------------------------
    # Load data
    # -----------------------------
    print("Loading processsed data.....")
    X_train = pd.read_csv(data_path / "X_train.csv")

    y_train = pd.read_csv(data_path / "y_train.csv").squeeze()

    # -----------------------------
    # Load parameters from file(if exists) and Define Model
    # -----------------------------
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8-sig") as f:
            content = f.read().strip()
        # override = yaml.safe_load(f) or {} 
        override = json.loads(content) if content else {}
    else:
        override = {}

    default_params = {
        "n_estimators":    100,
        "learning_rate":   0.1,
        "max_depth":       4,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "eval_metric":     "logloss",
        "random_state":    42,
        "tree_method": "hist",
        "n_jobs": -1,
    }

    # override defaults with anything in params.json
    final_params = {**default_params, **override}
    model = XGBClassifier(**final_params)

    # -----------------------------
    # Train
    # -----------------------------
    print("Training Model......")
    model.fit(X_train, y_train)

    # -------------------------------
    # Save parameters to params.json
    # -------------------------------
    params = model.get_params()
    
    with open(params_path, "w") as file:
        json.dump(params, file, indent=4)

    print("Parameters saved to params.json Successfully !")
    
    # -----------------------------
    # Save model
    # -----------------------------
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print("\nModel saved successfully ")

    return {
        "model_path":  str(model_path),
        "params_path": str(params_path),
        "train_rows":  len(X_train),
        "features":    len(X_train.columns),
        "skipped": False,
    }


if __name__ == "__main__":
    train_model()
