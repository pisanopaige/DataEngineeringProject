import os
import pickle
import pandas as pd
from s3fs import S3FileSystem
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def ingest_data():
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Define dataset and download path
    dataset = "kartik2112/fraud-detection"
    download_path = '/tmp/fraud-detection'

    # Download dataset from Kaggle
    api.dataset_download_files(dataset, path=download_path, unzip=True)

    # Define each path for the extracted datasets
    train_dataset_path = os.path.join(download_path, 'fraudTrain.csv')
    test_dataset_path = os.path.join(download_path, 'fraudTest.csv')

    # Load datasets into pandas dataframes
    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)

    # Initialize S3
    s3 = S3FileSystem()

    # Define S3 directory
    DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/batch_ingest/'

    # Push train and test data to S3 as pickle files
    with s3.open('{}/{}'.format(DIR, 'train_data.pkl'), 'wb') as f_train:
        f_train.write(pickle.dumps(train_data))

    with s3.open('{}/{}'.format(DIR, 'test_data.pkl'), 'wb') as f_test:
        f_test.write(pickle.dumps(test_data))

if __name__ == "__main__":
    ingest_data()