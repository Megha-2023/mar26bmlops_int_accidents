from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

# Make src scripts importable inside Airflow container
sys.path.insert(0, "/opt/airflow/src")

from src.data.make_dataset import make_dataset
from src.data.validate_data import validate_raw, validate_preprocessed
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model
from src.track_experiment import track_experiment

RAW_DATA_PATH = "/opt/airflow/data/accidents_full.csv"
PREPROCESSED_DATA_PATH = "/opt/airflow/data/preprocessed"
MODEL_PATH = "/opt/airflow/models/xgb_model.pkl"
PARAMS_PATH = "/opt/airflow/params.json"
METRICS_PATH = "/opt/airflow/metrics"
PLOTS_PATH = "/opt/airflow/metrics/plots"
MLFLOW_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "Accident_Prediction_Project_v2"

# Step-1 : Validate RaW data
def run_validate_raw_data(**context):
    validate_raw(data_path=RAW_DATA_PATH)

# Step-2 : Make Dataset
def run_make_dataset(**context):
    result = make_dataset(
        data_path=RAW_DATA_PATH,
        output_path=PREPROCESSED_DATA_PATH,
        force=False,
    )
    print(f"make_dataset summary: {result}")
    return result

# Step-3 : Validate Processed data
def run_validate_processed_data(**context):
    import gc
    validate_preprocessed(data_path=PREPROCESSED_DATA_PATH)
    gc.collect()

# Step-4 : Train Model
def run_train_model(**context):
    result = train_model(
        data_path=PREPROCESSED_DATA_PATH,
        model_path=MODEL_PATH,
        params_path=PARAMS_PATH,
        force=False,
    )
    print(f"train_model summary: {result}")
    return result

# Step-4 : Evaluate Model
def run_evaluate_model(**context):
    metrics = evaluate_model(
        data_path=PREPROCESSED_DATA_PATH,
        model_path=MODEL_PATH,
        metrics_path=METRICS_PATH,
        plots_path=PLOTS_PATH
    )
    # Push AUC to XCom for model promotion
    context["ti"].xcom_push(key="roc_auc", value=metrics["roc_auc"])

# Step-5 : Track Experiments to MLflow
def run_track_experiment(**context):
    version = track_experiment(
        model_path=MODEL_PATH,
        params_path=PARAMS_PATH,
        metrics_path=f"{METRICS_PATH}/metrics.json",
        plots_path=PLOTS_PATH,
        mlflow_uri=MLFLOW_URI,
        experiment_name=EXPERIMENT_NAME
    )
    context["ti"].xcom_push(key="model_version", value=version)


# Step-6 : Promote Model, if better
def run_promote_model(**context):
    import mlflow
    from mlflow import MlflowClient

    ti = context["ti"]
    new_auc = ti.xcom_pull(task_ids="evaluate_model", key="roc_auc")
    new_version = ti.xcom_pull(task_ids="track_experiment", key="model_version")

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    should_promote = False
    try:
        champion_model = client.get_model_version_by_alias("accident_prediction", "champion")
        champ_run = client.get_run(champion_model.run_id)
        champion_auc = float(champ_run.data.metrics.get("roc_auc", 0))
        # Compare auc to promote model or not
        if new_auc > champion_auc:
            should_promote = True
        print(f"Champion AUC: {champion_auc} | Challenger AUC: {new_auc}")
    except Exception:
        should_promote = True
        print("No Champion Model found - auto promoting.")
    
    if should_promote:
        client.set_registered_model_alias(
            name="accident_prediction", alias="champion", version=new_version
        )
        print(f"Model version {new_version} promoted to @champion")
    else:
        print(f"New model version did not beat champion - skipping promotion.")


# CREATE DAG
with DAG(
    dag_id="accident_pipeline_dag",
    start_date=datetime(2026, 4, 1),
    schedule=None,
    catchup=False,
    tags=["accident", "Liora"],
) as dag:
    
    t1 = PythonOperator(task_id="validate_raw_data", python_callable=run_validate_raw_data, provide_context=True)
    t2 = PythonOperator(task_id="make_dataset", python_callable=run_make_dataset, provide_context=True)
    t3 = PythonOperator(task_id="validate_processed_data", python_callable=run_validate_processed_data, provide_context=True)
    t4 = PythonOperator(task_id="train_model", python_callable=run_train_model, provide_context=True)
    
    
    t5 = PythonOperator(task_id="evaluate_model", python_callable=run_evaluate_model, provide_context=True)
    t6 = PythonOperator(task_id="track_experiment", python_callable=run_track_experiment, provide_context=True)
    t7 = PythonOperator(task_id="promote_model", python_callable=run_promote_model, provide_context=True)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7
