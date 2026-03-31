import pandas as pd
import joblib
from sklearn.metrics import f1_score, accuracy_score, classification_report

def main():

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

    print("\n Evaluation Results")
    print("----------------------")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()
