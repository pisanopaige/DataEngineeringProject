import pickle
import logging
from s3fs import S3FileSystem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tempfile

# Configure logging
log_filename = 'model_metrics.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def upload_to_s3(local_file, s3_path):
    # Initialize S3
    s3 = S3FileSystem()

    with open(local_file, 'rb') as f_local:
        with s3.open(s3_path, 'wb') as f_s3:
            f_s3.write(f_local.read())

def train_and_save_model():
    # Initialize S3
    s3 = S3FileSystem()

    # Define S3 directories
    DIR_features = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/feature_extraction/'
    DIR_transformed = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/transformed_data/'

    # Load transformed training features and target labels
    with s3.open(f'{DIR_features}/X_train_features.pkl', 'rb') as f_X:
        X_train_features = pickle.load(f_X)
    with s3.open(f'{DIR_transformed}/y_train_balanced.pkl', 'rb') as f_y:
        y_train_balanced = pickle.load(f_y)

    # Load transformed test features and target labels
    with s3.open(f'{DIR_transformed}/X_test_transformed.pkl', 'rb') as f_X_test:
        X_test = pickle.load(f_X_test)
    with s3.open(f'{DIR_transformed}/y_test.pkl', 'rb') as f_y_test:
        y_test = pickle.load(f_y_test)

    # Save feature names to S3
    with s3.open(f'{DIR_features}/feature_names.pkl', 'wb') as f_names:
        pickle.dump(X_train_features.columns.tolist(), f_names)

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_features, y_train_balanced)

    # Evaluate the model on the test data
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Log the metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    logging.info(f"Model Metrics: {metrics}")

    # Save model to a local file
    with tempfile.TemporaryDirectory() as tempdir:
        model_path = f"{tempdir}/random_forest_model.pkl"
        with open(model_path, 'wb') as f_model:
            pickle.dump(model, f_model)

        # Upload model to S3
        s3_dir = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/model_outputs/'
        upload_to_s3(model_path, f'{s3_dir}/random_forest_model.pkl')

    return metrics

if __name__ == "__main__":
    metrics = train_and_save_model()
    print("Metrics:", metrics)

