import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, roc_auc_score
from sqlalchemy import create_engine
from s3fs import S3FileSystem

def upload_data_to_sql():
    # Initialize S3
    s3 = S3FileSystem()

    # Define S3 directories
    MODEL_DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/model_outputs/'
    DATA_DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/transformed_data/'

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

    # Get predictions and evaluation
    predictions = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, predictions)
    roc_auc = roc_auc_score(y_test, y_scores)

    # Save confusion matrix and feature importances
    cm_df = pd.DataFrame(cm, columns=['Pred_Legitimate', 'Pred_Fraudulent'],
                         index=['Actual_Legitimate', 'Actual_Fraudulent'])
    importances = model.feature_importances_
    feature_names = X_test.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Create sqlalchemy engine to connect to MySQL
    user = "admin"
    pw = "C|r(kf!$]8c.1smc.HxN<D$v_S(Z"  # Update with actual password
    endpoint = "data-eng-db.cluster-cwgvgleixj0c.us-east-1.rds.amazonaws.com"
    db_name = "pisanopaige"

    # Create engine and connect to MySQL
    engine = create_engine(f"mysql+pymysql://{user}:{pw}@{endpoint}/{db_name}")

    # Insert DataFrames into MySQL DB
    with engine.connect() as connection:
        # Insert confusion matrix as table
        cm_df.to_sql('confusion_matrix', con=engine, if_exists='replace', index=False)

        # Insert feature importances as table
        feature_importance_df.to_sql('feature_importances', con=engine, if_exists='replace', index=False)

        # Insert ROC AUC score as table
        auc_df = pd.DataFrame({'Metric': ['ROC AUC'], 'Score': [roc_auc]})
        auc_df.to_sql('roc_auc_score', con=engine, if_exists='replace', index=False)

        # Insert predictions as table
        predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
        predictions_df.to_sql('predictions', con=engine, if_exists='replace', index=False)

        # Insert transformed data
        X_train_df = pd.DataFrame(X_train_balanced, columns=[f"feature_{i}" for i in range(X_train_balanced.shape[1])])
        X_test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
        y_train_df = pd.DataFrame(y_train_balanced, columns=["is_fraud"])
        y_test_df = pd.DataFrame(y_test, columns=["is_fraud"])

        # Upload to SQL
        X_train_df.to_sql('X_train', con=engine, if_exists='replace', index=False)
        X_test_df.to_sql('X_test', con=engine, if_exists='replace', index=False)
        y_train_df.to_sql('y_train', con=engine, if_exists='replace', index=False)
        y_test_df.to_sql('y_test', con=engine, if_exists='replace', index=False)


if __name__ == "__main__":
    upload_data_to_sql()
