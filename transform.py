import pandas as pd
from s3fs import S3FileSystem
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from datetime import datetime

def transform_data():
    # Initialize S3
    s3 = S3FileSystem()

    # Define S3 directory
    DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/batch_ingest//'

    # Load train and test data from S3 bucket
    train_data = pd.read_pickle(s3.open(f'{DIR}train_data.pkl', 'rb'))
    test_data = pd.read_pickle(s3.open(f'{DIR}test_data.pkl', 'rb'))

    # Perform data cleaning
    # Drop empty cells
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    # Drop duplicates
    train_data.drop_duplicates(inplace=True)
    test_data.drop_duplicates(inplace=True)

    # Convert date and time based columns to numerical data
    for df in [train_data, test_data]:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
        df['day_of_year'] = df['trans_date_trans_time'].dt.dayofyear
        df['year'] = df['trans_date_trans_time'].dt.year
        df['dob'] = pd.to_datetime(df['dob'])
        df['age'] = (datetime.now() - df['dob']).dt.days // 365

    # Drop the original data and time based columns
    train_data.drop(columns=['dob', 'trans_date_trans_time'], inplace=True)
    test_data.drop(columns=['dob', 'trans_date_trans_time'], inplace=True)

    # Encode labels for categorical data
    categorical_cols = ['merchant', 'category', 'gender', 'job', 'state', 'city']
    encoder = LabelEncoder()
    combined_data = pd.concat([train_data[categorical_cols], test_data[categorical_cols]])
    for col in categorical_cols:
        encoder.fit(combined_data[col])
        train_data[col] = encoder.transform(train_data[col])
        test_data[col] = encoder.transform(test_data[col])

    # Drop unnecessary columns
    train_data.drop(columns=['cc_num', 'first', 'last'], inplace=True)
    test_data.drop(columns=['cc_num', 'first', 'last'], inplace=True)

    # Separate features and target
    X_train = train_data.drop('is_fraud', axis=1)
    y_train = train_data['is_fraud']
    X_test = test_data.drop('is_fraud', axis=1)
    y_test = test_data['is_fraud']

    # Scale features
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    X_train_numeric = X_train[numeric_cols]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)

    X_test_numeric = X_test[numeric_cols]
    X_test_scaled = scaler.transform(X_test_numeric)

    # Save transformed datasets
    transformed_dir = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/transformed_data//'
    with s3.open(f'{transformed_dir}X_train_transformed.pkl', 'wb') as f_transformed:
        pickle.dump(X_train_scaled, f_transformed)
    with s3.open(f'{transformed_dir}y_train_transformed.pkl', 'wb') as f_y_transformed:
        pickle.dump(y_train, f_y_transformed)
    with s3.open(f'{transformed_dir}X_test_transformed.pkl', 'wb') as f_test_transformed:
        pickle.dump(X_test_scaled, f_test_transformed)
    with s3.open(f'{transformed_dir}y_test.pkl', 'wb') as f_y_test:
        pickle.dump(y_test, f_y_test)

    # Apply SMOTE to balance training data
    smote = SMOTE()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    with s3.open(f'{transformed_dir}X_train_balanced.pkl', 'wb') as f_balanced:
        pickle.dump(X_train_balanced, f_balanced)
    with s3.open(f'{transformed_dir}y_train_balanced.pkl', 'wb') as f_y_balanced:
        pickle.dump(y_train_balanced, f_y_balanced)

    return f'{transformed_dir}X_train_transformed.pkl', f'{transformed_dir}y_train_transformed.pkl', f'{transformed_dir}X_train_balanced.pkl', f'{transformed_dir}y_train_balanced.pkl', f'{transformed_dir}X_test_transformed.pkl', f'{transformed_dir}y_test.pkl'

if __name__ == "__main__":
    X_train_transformed_path, y_train_transformed_path, X_train_balanced_path, y_train_balanced_path, X_test_transformed_path, y_test_path = transform_data()
