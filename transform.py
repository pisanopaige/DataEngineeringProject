import pandas as pd
from s3fs import S3FileSystem
import pickle
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

def transform_data():
    s3 = S3FileSystem()
    # S3 bucket directory
    DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/batch_ingest/'

    # Load train and test data from S3 bucket
    train_data = np.load(s3.open('{}/{}'.format(DIR, 'train_data.pkl')), allow_pickle=True)  # insert here
    test_data = np.load(s3.open('{}/{}'.format(DIR, 'test_data.pkl')), allow_pickle=True)  # insert here

    # Remove any null values
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    # Remove any duplicate entries
    train_data.drop_duplicates(inplace=True)
    test_data.drop_duplicates(inplace=True)

    # Convert 'trans_date_trans_time' to datetime and extract useful time features
    train_data['trans_date_trans_time'] = pd.to_datetime(train_data['trans_date_trans_time'])
    train_data['hour'] = train_data['trans_date_trans_time'].dt.hour
    train_data['day_of_week'] = train_data['trans_date_trans_time'].dt.dayofweek
    train_data['day_of_year'] = train_data['trans_date_trans_time'].dt.dayofyear
    train_data['year'] = train_data['trans_date_trans_time'].dt.year  # Useful if trends change across years

    test_data['trans_date_trans_time'] = pd.to_datetime(test_data['trans_date_trans_time'])
    test_data['hour'] = test_data['trans_date_trans_time'].dt.hour
    test_data['day_of_week'] = test_data['trans_date_trans_time'].dt.dayofweek
    test_data['day_of_year'] = test_data['trans_date_trans_time'].dt.dayofyear
    test_data['year'] = test_data['trans_date_trans_time'].dt.year  # Useful if trends change across years

    # Convert 'dob' to 'age' by subtracting from current year
    train_data['dob'] = pd.to_datetime(train_data['dob'])
    train_data['age'] = (datetime.now() - train_data['dob']).dt.days // 365

    test_data['dob'] = pd.to_datetime(test_data['dob'])
    test_data['age'] = (datetime.now() - test_data['dob']).dt.days // 365

    # Drop 'dob' and 'trans_date_trans_time' as we now have 'age' and time features
    train_data.drop(columns=['dob', 'trans_date_trans_time'], inplace=True)
    test_data.drop(columns=['dob', 'trans_date_trans_time'], inplace=True)

    # Encode categorical features (merchant, category, gender, job, state, city)
    categorical_cols = ['merchant', 'category', 'gender', 'job', 'state', 'city']
    encoder = LabelEncoder()
    for col in categorical_cols:
        train_data[col] = encoder.fit_transform(train_data[col])
        test_data[col] = encoder.transform(test_data[col])  # Ensure same encoding for test data

    # Gender binary encoding (0 for female, 1 for male)
    train_data['gender'] = train_data['gender'].apply(lambda x: 1 if x == 'M' else 0)
    test_data['gender'] = test_data['gender'].apply(lambda x: 1 if x == 'M' else 0)

    # Drop columns that may not add predictive value (e.g., cc_num, first, last)
    train_data.drop(columns=['cc_num', 'first', 'last'], inplace=True)
    test_data.drop(columns=['cc_num', 'first', 'last'], inplace=True)

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