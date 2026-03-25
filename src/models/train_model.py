import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

def main():

    # -----------------------------
    # Load data
    # -----------------------------
    X_train = pd.read_csv("data/preprocessed/X_train.csv")
    X_test  = pd.read_csv("data/preprocessed/X_test.csv")

    y_train = pd.read_csv("data/preprocessed/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/preprocessed/y_test.csv").squeeze()

    # -----------------------------
    # Model (best choice for your dataset)
    # -----------------------------
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    # -----------------------------
    # Train
    # -----------------------------
    model.fit(X_train, y_train)

    # -----------------------------
    # Predict
    # -----------------------------
    y_pred = model.predict(X_test)

    # -----------------------------
    # Evaluate (same logic as evaluate_model.py)
    # -----------------------------
    acc = accuracy_score(y_test, y_pred)

    print("\n Training Evaluation")
    print("----------------------")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # -----------------------------
    # Save model
    # -----------------------------
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "xgb_model.pkl")

    print("\nModel saved successfully ")


if __name__ == "__main__":
    main()
