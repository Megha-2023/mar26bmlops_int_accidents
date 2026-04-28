from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="accident_pipeline",
    start_date=datetime(2026, 4, 1),
    schedule=None,
    catchup=False,
) as dag:

    t1 = BashOperator(
        task_id="make_dataset",
        bash_command="python /opt/airflow/src/data/make_dataset.py"
    )

    t2 = BashOperator(
        task_id="validate_data",
        bash_command="python /opt/airflow/src/data/validate_data.py"
    )

    

    t1 >> t2