# Step 2 - train the 3 models and log everything to MLflow on Azure ML
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from config import *

# connect mlflow to the azure workspace
ml_client = MLClient(DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)
tracking_uri = ml_client.workspaces.get(WORKSPACE_NAME).mlflow_tracking_uri
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(EXPERIMENT_NAME)
print("Tracking to:", tracking_uri)

# same data prep as train.py
fraud_df = pd.read_csv("./data/onlinefraud.csv")
fraud_df.drop('isFlaggedFraud', axis=1, inplace=True)
fraud_df['hour'] = fraud_df['step'] % 24
fraud_df['errorBalanceOrig'] = fraud_df['newbalanceOrig'] + fraud_df['amount'] - fraud_df['oldbalanceOrg']
fraud_df['errorBalanceDest'] = fraud_df['oldbalanceDest'] + fraud_df['amount'] - fraud_df['newbalanceDest']
fraud_df['accountEmptied'] = ((fraud_df['newbalanceOrig'] == 0) & (fraud_df['oldbalanceOrg'] > 0)).astype(int)
fraud_df['amountRatio'] = fraud_df['amount'] / (fraud_df['oldbalanceOrg'] + 1)
fraud_df['isNight'] = ((fraud_df['hour'] >= 0) & (fraud_df['hour'] <= 5)).astype(int)
fraud_df['destIsMerchant'] = fraud_df['nameDest'].str.startswith('M').astype(int)

X = fraud_df.drop(['isFraud', 'nameOrig', 'nameDest', 'step'], axis=1)
Y = fraud_df['isFraud']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# using a sample for training here to keep the runs quick, keeping all fraud rows
sample_idx = Y_train[Y_train == 1].index.tolist() + Y_train[Y_train == 0].sample(1000000, random_state=42).index.tolist()
X_train = X_train.loc[sample_idx]
Y_train = Y_train.loc[sample_idx]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['type']),
    ('num', StandardScaler(), [c for c in X.columns if c != 'type'])
])

# best params from the grid search in train.py
models = {
    'logistic_regression': LogisticRegression(max_iter=1000, C=10, class_weight='balanced'),
    'decision_tree': DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight='balanced', random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5,
                                            class_weight='balanced', n_jobs=-1, random_state=42),
}

os.makedirs("./plots/mlflow", exist_ok=True)

for name, clf in models.items():
    with mlflow.start_run(run_name=name):
        print(f"\n=== {name} ===")
        pipe = Pipeline([('prep', preprocessor), ('model', clf)])
        pipe.fit(X_train, Y_train)

        Y_pred = pipe.predict(X_test)
        Y_prob = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(Y_test, Y_pred),
            'precision': precision_score(Y_test, Y_pred),
            'recall': recall_score(Y_test, Y_pred),
            'f1': f1_score(Y_test, Y_pred),
            'roc_auc': roc_auc_score(Y_test, Y_prob),
            'pr_auc': average_precision_score(Y_test, Y_prob),
        }
        print(metrics)

        # log params
        mlflow.log_param("model", name)
        mlflow.log_params({k: v for k, v in clf.get_params().items()
                           if k in ['C', 'max_depth', 'min_samples_leaf', 'n_estimators', 'class_weight']})
        mlflow.log_param("train_rows", len(X_train))

        # log metrics
        mlflow.log_metrics(metrics)

        # log plots as artifacts
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ConfusionMatrixDisplay(confusion_matrix(Y_test, Y_pred)).plot(ax=ax[0])
        ax[0].set_title('Confusion Matrix')
        RocCurveDisplay.from_predictions(Y_test, Y_prob, ax=ax[1])
        ax[1].set_title('ROC Curve')
        PrecisionRecallDisplay.from_predictions(Y_test, Y_prob, ax=ax[2])
        ax[2].set_title('PR Curve')
        plt.tight_layout()
        plot_path = f"./plots/mlflow/{name}_eval.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)

        if hasattr(clf, 'feature_importances_'):
            feat_names = pipe.named_steps['prep'].get_feature_names_out()
            imp = pd.Series(clf.feature_importances_, index=feat_names).sort_values()
            plt.figure(figsize=(8, 6))
            imp.plot(kind='barh')
            plt.title('Feature Importance')
            plt.tight_layout()
            imp_path = f"./plots/mlflow/{name}_importance.png"
            plt.savefig(imp_path)
            plt.close()
            mlflow.log_artifact(imp_path)

        # log the model itself
        mlflow.sklearn.log_model(pipe, artifact_path="model", input_example=X_test.head(2))
        print("Run id:", mlflow.active_run().info.run_id)

print("\nAll runs logged. Check the Jobs page in Azure ML studio.")
