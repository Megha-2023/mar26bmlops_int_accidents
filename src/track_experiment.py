import mlflow 
import os
import json
import joblib
from datetime import datetime


def track_experiment():

    # Connect to the MLflow container
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow.set_experiment("Accident_Prediction_Project")
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        # Load model
        trained_model = joblib.load("models/xgb_model.pkl")

        # Load parameters and log
        if os.path.exists("params.json"):
            with open("params.json", "r") as file:
                params = json.load(file)
            mlflow.log_params(params)
            print("Parameters logged to MLflow Successfully !")
        else:
            print("Parameters file not found !")

        # Log metrics from the metrics.json file
        metrics_path = os.path.join(os.getcwd(), "metrics/metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as file:
                metrics = json.load(file)
            mlflow.log_metrics(metrics)
            print("Metrics Logged to MLflow Successfully !")
        else:
            print(f"{metrics_path} not found !")

        # Log model 
        mlflow.sklearn.log_model(
            sk_model=trained_model,
            name="model"
        )
        
        # Register Model to Mlflow Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(
            model_uri=model_uri,
            name="accident_prediction"
        )


if __name__ == "__main__":
    track_experiment()
