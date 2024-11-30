import pandas as pd
from s3fs import S3FileSystem
import pickle
from sklearn.preprocessing import MinMaxScaler

def feature_extract():
    s3 = S3FileSystem()
    DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/transformed_data/'
    with s3.open(f'{DIR}/X_train_transformed.pkl', 'rb') as f_X:
        X_train_balanced = pickle.load(f_X)
    with s3.open(f'{DIR}/y_train_transformed.pkl', 'rb') as f_y:
        y_train_balanced = pickle.load(f_y)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)

    DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/feature_extraction/'
    with s3.open(f'{DIR}/X_train_features.pkl', 'wb') as f_X_features:
        pickle.dump(X_train_scaled, f_X_features)

    return X_train_scaled

if __name__ == "__main__":
    feature_extract()
