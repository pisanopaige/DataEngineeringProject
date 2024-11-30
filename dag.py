from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime
from batch_ingest import ingest_data
from transform import transform_data
from train_model import train_and_save_random_forest
from feature_extraction import feature_extract  # Added the feature extraction import
# from visualize import visualize_data  # Commented out visualize task

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
    description='A pipeline for credit card fraud detection that includes batch ingestion, feature extraction, data transformation, model training, and visualization',
    schedule_interval=timedelta(days=1),
    catchup=False  # Ensures the DAG doesn't run for previous days
)

# Define tasks
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

feature_extract_task = PythonOperator(
    task_id='feature_extract',
    python_callable=feature_extract,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_and_save_random_forest,
    dag=dag,
)

# Commented out the visualize task for now
# visualize_task = PythonOperator(
#     task_id='visualize_data',
#     python_callable=visualize_data,  # Assuming you have this function defined
#     dag=dag,
# )

# Set task dependencies
ingest_task >> transform_task >> feature_extract_task >> train_task
