import pandas as pd
from s3fs import S3FileSystem
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


def transform_data():
    s3 = S3FileSystem()
    # S3 bucket directory
    DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/batch_ingest'

    # Load train and test data from S3 bucket
    with s3.open('{}/{}'.format(DIR, 'train_data.pkl'), 'rb') as f_train:
        train_data = pickle.load(f_train)

    with s3.open('{}/{}'.format(DIR, 'test_data.pkl'), 'rb') as f_test:
        test_data = pickle.load(f_test)

    # Remove any null values
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    # Remove any duplicate entries
    train_data.drop_duplicates(inplace=True)
    test_data.drop_duplicates(inplace=True)

    # Separate features and target variable
    X_train = train_data.drop('is_fraud', axis=1)
    y_train = train_data['is_fraud']

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Oversample using SMOTE
    smote = SMOTE()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    # Return the transformed datasets
    return X_train_balanced, y_train_balanced, test_data


if __name__ == "__main__":
    X_train_balanced, y_train_balanced, test_data = transform_data()