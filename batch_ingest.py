import kagglehub
import pandas as pd
from s3fs import S3FileSystem
import pickle
import zipfile
import os

def ingest_data():
    # Load fraud dataset from Kaggle
    path = kagglehub.dataset_download("kartik2112/fraud-detection")

    # Unzip the downloaded zip file
    zip_file_path = 'archive.zip'
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        zip_file.extractall()

    # Load datasets into pandas dataframes
    train_dataset_path = os.path.join(path, 'archive', 'fraudTrain.csv')
    test_dataset_path = os.path.join(path, 'archive', 'fraudTest.csv')
    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)

    s3 = S3FileSystem()
    # S3 bucket directory
    DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/'

    # Push train and test data to S3 bucket as pickle files
    with s3.open('{}/{}'.format(DIR, 'train_data.pkl'), 'wb') as f_train:
        f_train.write(pickle.dumps(train_data))
    with s3.open('{}/{}'.format(DIR, 'test_data.pkl'), 'wb') as f_test:
        f_test.write(pickle.dumps(test_data))

if __name__ == "__main__":
    ingest_data()