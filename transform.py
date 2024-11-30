import pandas as pd
from s3fs import S3FileSystem
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def transform_data():
    s3 = S3FileSystem()
    # S3 bucket directory
    DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/batch_ingest/'

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

    # Convert datetime columns to numeric (if applicable)
    # For example, you can convert datetime to timestamp
    datetime_columns = train_data.select_dtypes(include=['datetime']).columns
    for col in datetime_columns:
        train_data[col] = train_data[col].astype('int64') // 10**9  # Convert to seconds since epoch
        test_data[col] = test_data[col].astype('int64') // 10**9  # Convert to seconds since epoch

    # Separate features and target variable
    X_train = train_data.drop('is_fraud', axis=1)
    y_train = train_data['is_fraud']

    # Standardize (ensure the data is numeric)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Oversample using SMOTE
    smote = SMOTE()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    # S3 bucket directory for transformed data
    DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/transformed_data/'

    # Push train and test transformed data to S3 bucket as pickle files
    with s3.open('{}/{}'.format(DIR, 'X_train_transformed.pkl'), 'wb') as f_X:
        pickle.dump(X_train_balanced, f_X)
    with s3.open('{}/{}'.format(DIR, 'y_train_transformed.pkl'), 'wb') as f_y:
        pickle.dump(y_train_balanced, f_y)
    with s3.open('{}/{}'.format(DIR, 'test_data_transformed.pkl'), 'wb') as f_test_out:
        pickle.dump(test_data, f_test_out)

    # Return the transformed datasets
    return X_train_balanced, y_train_balanced, test_data

if __name__ == "__main__":
    X_train_balanced, y_train_balanced, test_data = transform_data()
