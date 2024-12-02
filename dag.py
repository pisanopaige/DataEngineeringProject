from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime
from batch_ingest import ingest_data
from transform import transform_data
from train_model import train_and_save_model
from feature_extract import feature_extract
from sql_load import upload_data_to_sql

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
    description='A pipeline for credit card fraud detection that includes batch ingestion, feature extraction, data transformation, model training, and SQL data upload',
    schedule_interval=timedelta(days=1),
    catchup=False
)

# Define tasks
ingest_etl = PythonOperator(
    task_id='ingest_dataset',
    python_callable=ingest_data,
    dag=dag,
)

transform_etl = PythonOperator(
    task_id='transform_dataset',
    python_callable=transform_data,
    dag=dag,
)

feature_extract_etl = PythonOperator(
    task_id='feature_extract',
    python_callable=feature_extract,
    dag=dag,
)

train_etl = PythonOperator(
    task_id='train_model',
    python_callable=train_and_save_model,
    dag=dag,
)

sql_load_etl = PythonOperator(
    task_id='sql_load',
    python_callable=upload_data_to_sql,
    dag=dag,
)

# Set task dependencies
ingest_etl >> transform_etl >> feature_extract_etl >> train_etl >> sql_load_etl
