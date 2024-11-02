from datetime import timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime
from batch_ingest import ingest_data
from transform import transform_data
from train_model import train_model
from visualize import visualize_data

default_args = {
    'owner': 'pisanopaige',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 1),
    'email': ['pisanopaige@vt.edu'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'fraud_detection_dag',
    default_args=default_args,
    description='A pipeline for credit card fraud detection that includes batch ingestion, data transformation, model training, and visualization',
    schedule_interval=timedelta(days=1),
)

ingest_task = PythonOperator(
    task_id='ingest_dataset',
    python_callable=ingest_data,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_dataset',
    python_callable=transform_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

visualize_task = PythonOperator(
    task_id='visualize_data',
    python_callable=visualize_data,
    dag=dag,
)

# Set task dependencies
ingest_task >> transform_task >> train_task >> visualize_task
