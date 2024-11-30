import pickle
import logging
from s3fs import S3FileSystem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tempfile

log_filename = 'model_metrics.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def upload_to_s3(local_file, s3_path):
    s3 = S3FileSystem()
    with open(local_file, 'rb') as f_local:
        with s3.open(s3_path, 'wb') as f_s3:
            f_s3.write(f_local.read())

def train_and_save_random_forest():
    s3 = S3FileSystem()
    DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/feature_extraction'
    with s3.open(f'{DIR}/X_train_features.pkl', 'rb') as f_X:
        X_train_features = pickle.load(f_X)

    DIR_transformed = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/transformed_data'
    with s3.open(f'{DIR_transformed}/y_train_transformed.pkl', 'rb') as f_y:
        y_train_balanced = pickle.load(f_y)
    with s3.open(f'{DIR_transformed}/test_data_transformed.pkl', 'rb') as f_test:
        test_data = pickle.load(f_test)

    X_test = test_data.drop('is_fraud', axis=1)
    y_test = test_data['is_fraud']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_features, y_train_balanced)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
    logging.info(f"Model Metrics: {metrics}")

    with tempfile.TemporaryDirectory() as tempdir:
        model_path = f"{tempdir}/random_forest_model.pkl"
        with open(model_path, 'wb') as f_model:
            pickle.dump(model, f_model)

        s3_dir = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/model_outputs'
        upload_to_s3(model_path, f'{s3_dir}/random_forest_model.pkl')

    return model

if __name__ == "__main__":
    train_and_save_random_forest()


