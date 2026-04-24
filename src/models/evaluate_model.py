import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    ConfusionMatrixDisplay, roc_auc_score,
    roc_curve, auc
)

def plot_roc_curve(y_test, probs, plots_path):
    """ Function defined to plot roc curve for multiclass use one-vs-test """
    plots_path = Path(plots_path)
    plots_path.mkdir(parents=True, exist_ok=True)

    classes = sorted(y_test.unique())
    y_bin = label_binarize(y_test, classes=classes)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"Class {cls} (AUC = {auc_score:.2f})")
    
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Postive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (onv-vs-rest)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plots_path / "roc_curve.png")
    plt.close()
    print("ROC curve saved.")



def evaluate_model(
        data_path="data/preprocessed",
        model_path="models/xgb_model.pkl",
        metrics_path="metrics",
        plots_path="metrics/plots"
):
    data_path = Path(data_path)
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    plots_path = Path(plots_path)

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
    probs = model.predict_proba(X_test)
    # -----------------------------
    # Evaluation
    # -----------------------------
    # Scalar Metrics: Accuracy, F1 score, roc_auc
    metrics_path.mkdir(parents=True, exist_ok=True)
    acc = round(accuracy_score(y_test, preds), 4)
    f1 = round(f1_score(y_test, preds, average="weighted"), 4)

    
    metrics = {
        "Accuracy": acc,
        "F1 Score": f1,
    }
    # ROC-AUC — handle binary and multiclass automatically
    n_classes = len(y_test.unique())
    if n_classes == 2:
        metrics["roc_auc"] = round(roc_auc_score(y_test, probs[:, 1]), 4)
    else:
        metrics["roc_auc"] = round(
            roc_auc_score(y_test, probs, multi_class="ovr", average="weighted"), 4
        )
    # Classification Report
    class_report = classification_report(y_test, preds)

    # Confusion matrix → image artifact
    plots_path.mkdir(parents=True, exist_ok=True)

    cm_path = plots_path / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print("Confusion matrix plot saved.")

    # ROC curve → image artifact 
    plot_roc_curve(y_test, probs, plots_path)
    
    # Save Scalar metrics and Classification report to its path
    with open(metrics_path / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    with open(metrics_path / "classification_report.txt", "w") as f:
        f.write(class_report)
    
    print(f"Metrics and Classification Report stored Successfully in {metrics_path}!")

    print("\n Evaluation Results")
    print("----------------------")
    print(f"Accuracy: {acc}")
    print(f"F1 Score:  {f1}")

    print("\nClassification Report:\n")
    print(class_report)
    return metrics


if __name__ == "__main__":
    evaluate_model()
