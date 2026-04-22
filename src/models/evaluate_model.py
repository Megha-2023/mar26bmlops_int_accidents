import pandas as pd
import joblib
import json
import os
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, classification_report

def evaluate_model(
        data_path="data/preprocessed",
        model_path="models/xgb_model.pkl",
        metrics_path="metrics"
):
    data_path = Path(data_path)
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)

    # -----------------------------
    # Load test data
    # -----------------------------
    X_test = pd.read_csv(data_path / "X_test.csv")
    y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

    # -----------------------------
    # Load correct model
    # -----------------------------
    model = joblib.load(model_path)

    # -----------------------------
    # Predictions
    # -----------------------------
    preds = model.predict(X_test)

    # -----------------------------
    # Evaluation
    # -----------------------------
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    metrics_path.mkdir(parents=True, exist_ok=True)

    metrics = {
        "Accuracy": acc,
        "F1 Score": f1
    }

    class_report = classification_report(y_test, preds)

    print("\n Evaluation Results")
    print("----------------------")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:\n")
    print(class_report)
    
    # Save metrics to its path
    with open(metrics_path / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    with open(metrics_path / "classification_report.txt", "w") as f:
        f.write(class_report)
    
    print(f"Metrics and Classification Report stored Successfully in {metrics_path}!")

    return {
        "model_path":  str(model_path),
        "metrics_path": str(metrics_path),
        "test_rows":  len(X_test),
        "features":    len(X_test.columns),
    }


if __name__ == "__main__":
    evaluate_model()

