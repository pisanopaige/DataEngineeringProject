import pickle
import logging
from s3fs import S3FileSystem
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
from sqlalchemy import create_engine

# Configure logging
log_filename = 'model_metrics.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


def upload_data_to_sql():
    # Initialize S3
    s3 = S3FileSystem()

    # Define S3 directories
    MODEL_DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/model_outputs/'
    DATA_DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/transformed_data/'
    FEATURE_DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/feature_extraction/'

    # Load trained model
    with s3.open(f'{MODEL_DIR}/random_forest_model.pkl', 'rb') as f_model:
        model = pickle.load(f_model)

    # Load test and training data
    with s3.open(f'{DATA_DIR}/X_train_balanced.pkl', 'rb') as f_X_train:
        X_train_balanced = pickle.load(f_X_train)
    with s3.open(f'{DATA_DIR}/y_train_balanced.pkl', 'rb') as f_y_train:
        y_train_balanced = pickle.load(f_y_train)
    with s3.open(f'{DATA_DIR}/X_test_transformed.pkl', 'rb') as f_X_test:
        X_test = pickle.load(f_X_test)
    with s3.open(f'{DATA_DIR}/y_test.pkl', 'rb') as f_y_test:
        y_test = pickle.load(f_y_test)

    # Load feature names
    with s3.open(f'{FEATURE_DIR}/feature_names.pkl', 'rb') as f_names:
        feature_names = pickle.load(f_names)

    # Get predictions and evaluation metrics
    predictions = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, y_scores)
    cm = confusion_matrix(y_test, predictions)

    # Save confusion matrix and feature importances
    cm_df = pd.DataFrame(cm, columns=['Pred_Legitimate', 'Pred_Fraudulent'],
                         index=['Actual_Legitimate', 'Actual_Fraudulent'])

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Store precision, recall, and F1 scores
    precision_recall_df = pd.DataFrame({
        'Precision': [precision],
        'Recall': [recall],
        'F1_Score': [f1]
    })

    # Detailed confusion matrix breakdown
    cm_values = cm.flatten()
    cm_detailed_df = pd.DataFrame({
        'True Positive': [cm_values[3]],
        'False Positive': [cm_values[1]],
        'False Negative': [cm_values[2]],
        'True Negative': [cm_values[0]]
    })

    # Store model metrics over time
    metrics_history_df = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1_Score': [f1],
        'ROC_AUC': [roc_auc]
    })

    # Cross-validation metrics
    cross_val_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
    cross_val_df = pd.DataFrame({
        'Cross-Val Accuracy': cross_val_scores,
        'Mean Accuracy': np.mean(cross_val_scores),
        'Std Accuracy': np.std(cross_val_scores)
    })

    # Store hyperparameters
    hyperparameters_df = pd.DataFrame({
        'Hyperparameter': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
        'Value': [model.n_estimators, model.max_depth, model.min_samples_split, model.min_samples_leaf]
    })

    # Correlation matrix
    corr_matrix = pd.DataFrame(X_train_balanced, columns=feature_names).corr()

    # Create sqlalchemy engine to connect to MySQL
    user = "admin"
    pw = "C|r(kf!$]8c.1smc.HxN<D$v_S(Z"  # Update with actual password
    endpoint = "data-eng-db.cluster-cwgvgleixj0c.us-east-1.rds.amazonaws.com"
    db_name = "pisanopaige"

    # Create engine and connect to MySQL
    engine = create_engine(f"mysql+pymysql://{user}:{pw}@{endpoint}/{db_name}")

    # Insert DataFrames into MySQL DB
    with engine.connect() as connection:
        # Insert confusion matrix and other tables
        cm_df.to_sql('confusion_matrix', con=engine, if_exists='replace', index=False)
        feature_importance_df.to_sql('feature_importances', con=engine, if_exists='replace', index=False)
        precision_recall_df.to_sql('precision_recall_metrics', con=engine, if_exists='replace', index=False)
        cm_detailed_df.to_sql('confusion_matrix_detailed', con=engine, if_exists='replace', index=False)
        metrics_history_df.to_sql('model_metrics_history', con=engine, if_exists='replace', index=False)
        cross_val_df.to_sql('cross_validation_metrics', con=engine, if_exists='replace', index=False)
        hyperparameters_df.to_sql('model_hyperparameters', con=engine, if_exists='replace', index=False)
        corr_matrix.to_sql('feature_correlations', con=engine, if_exists='replace', index=False)

        # Insert transformed data
        X_train_df = pd.DataFrame(X_train_balanced, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        y_train_df = pd.DataFrame(y_train_balanced, columns=["is_fraud"])
        y_test_df = pd.DataFrame(y_test, columns=["is_fraud"])

        # Upload to SQL
        X_train_df.to_sql('X_train', con=engine, if_exists='replace', index=False)
        X_test_df.to_sql('X_test', con=engine, if_exists='replace', index=False)
        y_train_df.to_sql('y_train', con=engine, if_exists='replace', index=False)
        y_test_df.to_sql('y_test', con=engine, if_exists='replace', index=False)


if __name__ == "__main__":
    upload_data_to_sql()

