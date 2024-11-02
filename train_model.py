import pandas as pd
import pickle
import logging
from s3fs import S3FileSystem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set up logging
log_filename = 'model_metrics.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def upload_log_to_s3():
    s3 = S3FileSystem()
    # S3 bucket directory for logs
    DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/logs'

    # Upload log file to S3
    with s3.open(f'{DIR}/{log_filename}', 'wb') as f:
        with open(log_filename, 'rb') as local_file:
            f.write(local_file.read())

def log_model_metrics(metrics):
    logging.info(f'Model Metrics: {metrics}')
    upload_log_to_s3()

def train_model():
    s3 = S3FileSystem()
    # S3 bucket directory for data
    DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/transformed_data'

    # Load transformed train data
    with s3.open('{}/{}'.format(DIR, 'X_train_transformed.pkl'), 'rb') as f_X:
        X_train_balanced = pickle.load(f_X)
    with s3.open('{}/{}'.format(DIR, 'y_train_transformed.pkl'), 'rb') as f_y:
        y_train_balanced = pickle.load(f_y)

    # Load transformed test data
    with s3.open('{}/{}'.format(DIR, 'test_data_transformed.pkl'), 'rb') as f_test:
        test_data = pickle.load(f_test)

    # Separate features and target variable for test data
    X_test = test_data.drop('is_fraud', axis=1)
    y_test = test_data['is_fraud']

    # Initialize the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train_balanced, y_train_balanced)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Log metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    log_model_metrics(metrics)

    # Initialize S3 file system and specify the S3 bucket directory for model outputs
    DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/model_outputs'

    # Save predictions and model directly to S3
    with s3.open('{}/{}'.format(DIR, 'predictions.pkl'), 'wb') as f_pred:
        f_pred.write(pickle.dumps(predictions))

    with s3.open('{}/{}'.format(DIR, 'random_forest_model.pkl'), 'wb') as f_model:
        f_model.write(pickle.dumps(model))

    return model

if __name__ == "__main__":
    # Train the model and log metrics
    model = train_model()

