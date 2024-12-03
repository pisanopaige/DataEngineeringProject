import pandas as pd
import pickle
from s3fs import S3FileSystem
from sklearn.preprocessing import MinMaxScaler

def feature_extract():
    # Initialize S3
    s3 = S3FileSystem()

    # Define S3 directory
    DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/transformed_data/'

    # Load balanced training features
    with s3.open(f'{DIR}/X_train_balanced.pkl', 'rb') as f_X:
        X_train_balanced = pickle.load(f_X)

    # Scale the features using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)

    # Convert back to DataFrame and retain feature names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_balanced.columns)

    # Save the scaled features to S3
    feature_dir = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/feature_extraction/'
    with s3.open(f'{feature_dir}/X_train_features.pkl', 'wb') as f_X_features:
        pickle.dump(X_train_scaled_df, f_X_features)

    return f'{feature_dir}/X_train_features.pkl'

if __name__ == "__main__":
    X_train_features_path = feature_extract()

