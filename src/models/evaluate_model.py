import pandas as pd
import joblib
import json
import os
from sklearn.metrics import f1_score, accuracy_score, classification_report

def evaluate_model():

    # -----------------------------
    # Load test data
    # -----------------------------
    X_test = pd.read_csv("data/preprocessed/X_test.csv")
    y_test = pd.read_csv("data/preprocessed/y_test.csv").squeeze()

    # -----------------------------
    # Load correct model
    # -----------------------------
    model = joblib.load("models/xgb_model.pkl")

    # -----------------------------
    # Predictions
    # -----------------------------
    preds = model.predict(X_test)

    # -----------------------------
    # Evaluation
    # -----------------------------
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    metrics_path = "metrics"
    os.makedirs(metrics_path, exist_ok=True)

    metrics = {
        "Accuracy": f"{acc:.4f}",
        "F1 Score": f"{f1:.4f}"
    }

    with open(os.path.join(metrics_path, "metrics.json"), "w") as file:
        json.dump(metrics, file, indent=4)
    print("Metrics stored Successfully in a file 'metrics.json' !")

    print("\n Evaluation Results")
    print("----------------------")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))
    
    return model


if __name__ == "__main__":
    model = evaluate_model()
