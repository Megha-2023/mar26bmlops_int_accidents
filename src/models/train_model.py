import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os
from pathlib import Path

def train_model(
        data_path="data/preprocessed",
        model_path="models/xgb_model.pkl",
        params_path="params.json"
):

    data_path = Path(data_path)
    model_path = Path(model_path)
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
        "n_estimators":    200,
        "learning_rate":   0.05,
        "max_depth":       6,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "eval_metric":     "logloss",
        "random_state":    42,
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
    }


if __name__ == "__main__":
    train_model()

