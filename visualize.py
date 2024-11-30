import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from s3fs import S3FileSystem


def visualize_data():
    s3 = S3FileSystem()
    # S3 bucket directories
    MODEL_DIR= 's3://ece5984-s3-pisanopaige/DataEngineeringProject/model_outputs/'
    DATA_DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/transformed_data/'

    # Load the trained model
    with s3.open('{}/{}'.format(MODEL_DIR, 'random_forest_model.pkl'), 'rb') as f_model:
        model = pickle.load(f_model)

    with s3.open('{}/{}'.format(DATA_DIR, 'test_data_transformed.pkl'), 'rb') as f_test:
        test_data = pickle.load(f_test)

    # Separate features and target variable for test data
    X_test = test_data.drop('is_fraud', axis=1)
    y_test = test_data['is_fraud']

    # Set up the matplotlib figure
    plt.figure(figsize=(15, 10))

    # Bar plot for the number of faulty vs legitimate transactions
    plt.subplot(2, 2, 1)
    sns.countplot(x=y_test)
    plt.title('Count of Faulty vs Legitimate Transactions')
    plt.xlabel('Transaction Type')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Legitimate', 'Fraudulent'])

    #Confusion Matrix
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)

    plt.subplot(2, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(ticks=[0.5, 1.5], labels=['Legitimate', 'Fraudulent'])
    plt.yticks(ticks=[0.5, 1.5], labels=['Legitimate', 'Fraudulent'])

    # ROC Curve
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)

    plt.subplot(2, 2, 3)
    plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    # Feature Importance Plot
    importances = model.feature_importances_
    feature_names = X_test.columns
    indices = importances.argsort()[::-1]

    plt.subplot(2, 2, 4)
    plt.title('Feature Importances')
    plt.barh(range(len(importances)), importances[indices], align='center')
    plt.yticks(range(len(importances)), feature_names[indices])
    plt.xlabel('Relative Importance')

    # Adjust layout
    plt.tight_layout()

    # Initialize S3 file system and specify the S3 bucket directory for model outputs
    VISUALIZATION_DIR = 's3://ece5984-s3-pisanopaige/DataEngineeringProject/visualizations/'

    # Push plots to S3 bucket
    plt.savefig('fraud_visualization_plots.png')

    with s3.open('{}/{}'.format(VISUALIZATION_DIR, 'fraud_visualization_plots.png'), 'wb') as f:
        with open('fraud_visualization_plots.png', 'rb') as local_file:
            f.write(local_file.read())

if __name__ == "__main__":
    # Call the visualize_data function
    visualize_data()
