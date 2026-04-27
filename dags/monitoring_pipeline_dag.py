from datetime import datetime
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

# Make src scripts importable inside Airflow container
sys.path.insert(0, "/opt/airflow/src")

from src.monitoring.data_loader import load_current_data, load_reference_data
from src.monitoring.evidently_report import (
    generate_data_drift_report,
    generate_prediction_drift_report,
)
from src.monitoring.model_loader import generate_predictions, load_model


REFERENCE_DATA_PATH = "/opt/airflow/data/preprocessed/X_train.csv"
CURRENT_DATA_PATH = "/opt/airflow/data/processed/accidents_2016_2018.csv"
REPORTS_DIR = "/opt/airflow/metrics/reports"


def attach_prediction_column(model, data):
    data_with_predictions = data.copy()
    data_with_predictions["prediction"] = generate_predictions(model, data)
    return data_with_predictions


def run_generate_data_drift_report(**context):
    reference_data = load_reference_data(data_path=REFERENCE_DATA_PATH)
    current_data = load_current_data(data_path=CURRENT_DATA_PATH)

    report_path = generate_data_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        report_name="xtrain_vs_current_data_drift_report.html",
        reports_dir=REPORTS_DIR,
    )

    result = {"report_path": str(report_path), "report_type": "data_drift"}
    print(f"Generated data drift report: {result}")
    return result


def run_generate_prediction_drift_report(**context):
    reference_data = load_reference_data(data_path=REFERENCE_DATA_PATH)
    current_data = load_current_data(data_path=CURRENT_DATA_PATH)
    model = load_model()

    reference_with_predictions = attach_prediction_column(model, reference_data)
    current_with_predictions = attach_prediction_column(model, current_data)

    report_path = generate_prediction_drift_report(
        reference_data=reference_with_predictions,
        current_data=current_with_predictions,
        prediction_column="prediction",
        report_name="xtrain_vs_current_prediction_drift_report.html",
        reports_dir=REPORTS_DIR,
    )

    result = {"report_path": str(report_path), "report_type": "prediction_drift"}
    print(f"Generated prediction drift report: {result}")
    return result


with DAG(
    dag_id="monitoring_pipeline_dag",
    start_date=datetime(2026, 4, 1),
    schedule=None,
    catchup=False,
    tags=["monitoring", "evidently", "accident"],
) as dag:

    t1 = PythonOperator(
        task_id="generate_data_drift_report",
        python_callable=run_generate_data_drift_report,
        provide_context=True,
    )
    t2 = PythonOperator(
        task_id="generate_prediction_drift_report",
        python_callable=run_generate_prediction_drift_report,
        provide_context=True,
    )

    t1 >> t2

