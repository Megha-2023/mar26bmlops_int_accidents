import mlflow
from mlflow import MlflowClient
import os
import json
import joblib
from pathlib import Path
from datetime import datetime


DEFAULT_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def get_or_create_experiment(client, experiment_name):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location="mlflow-artifacts:/",
        )
        print(f"Created MLflow experiment '{experiment_name}' with proxied artifacts.")
        return experiment_id

    if experiment.artifact_location.startswith("/"):
        raise RuntimeError(
            f"Experiment '{experiment_name}' uses local artifact path "
            f"'{experiment.artifact_location}', which is not writable from Airflow. "
            "Delete and recreate this experiment, or use a new experiment name."
        )

    return experiment.experiment_id


def track_experiment(
        model_path="models/xgb_model.pkl",
        params_path="params.json",
        metrics_path="metrics/metrics.json",
        plots_path="metrics/plots",
        mlflow_uri=DEFAULT_MLFLOW_URI,
        experiment_name="Accident_Prediction_Project",
):

    model_path = Path(model_path)
    metrics_path = Path(metrics_path)

    # Connect to the MLflow container
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    experiment_id = get_or_create_experiment(client, experiment_name)
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Log params, metrics and model artifact
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        run_id = run.info.run_id

        # Load parameters and log
        # params_path = os.path.join(os.getcwd(), "params.json")
        print(f"Parameters file {params_path} !")
        if os.path.exists(params_path):
            with open(params_path, "r", encoding='utf-8-sig') as f:
                content = f.read().strip()

                if not content:
                    print("WARNING: params.json file is empty - skipping")
                else:
                    params = json.loads(content)
                    if params:
                        mlflow.log_params(params)
                        print(f"{len(params)} Parameters logged to MLflow Successfully !")
        else:
            print("Parameter file params.json not found !")

        # Log Scalar Metrics from the metrics.json file
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding='utf-8-sig') as file:
                metrics = json.load(file)
            mlflow.log_metrics(metrics)
            print("Metrics logged to MLflow Successfully !")
        else:
            print(f"{metrics_path} not found !")

        # Log Classification report
        report_path = metrics_path.parent / "classification_report.txt"
        if report_path.exists():
            mlflow.log_artifact(str(report_path), artifact_path="reports")
            print("Classification Report logged to MLflow Successfully!")
        
        # Log plots as artifacts
        for plot_file in ["confusion_matrix.png", "roc_curve.png"]:
            plot_full_path = os.path.join(plots_path, plot_file)
            if os.path.exists(plot_full_path):
                mlflow.log_artifact(plot_full_path, artifact_path="plots")
                print(f"Plots logged Successfully!")
        
        # Log model artifact
        trained_model = joblib.load(model_path)
        mlflow.xgboost.log_model(
            xgb_model=trained_model,
            artifact_path="model",
        )

    # Post-run block: register + tag + alias
    print(f" Run {run_id} completed. Registering model.....")

    # Register Model to Mlflow Model Registry
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(
                model_uri=model_uri,
                name="accident_prediction"
            )
    version = result.version
    print(f"Model registered as version {version}")

    # Tag the version
    client.set_model_version_tag(
        name="accident_prediction",
        version=version,
        key="stage",
        value="candidate"
    )

    # Set alias
    client.set_registered_model_alias(
        name="accident_prediction",
        alias="challenger",
        version=version
    )
    print(f"Alias '@challenger' -> version {version} set.")
    return version

if __name__ == "__main__":
    track_experiment()
