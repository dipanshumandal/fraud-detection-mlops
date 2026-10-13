# Step 3 - compare runs, pick the best one and register it in the model registry
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import mlflow
from mlflow.tracking import MlflowClient
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from config import *

ml_client = MLClient(DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)
mlflow.set_tracking_uri(ml_client.workspaces.get(WORKSPACE_NAME).mlflow_tracking_uri)

# all runs of the experiment as a dataframe
runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
runs = runs[runs['status'] == 'FINISHED']
runs = runs.sort_values('metrics.f1', ascending=False)

print("Runs ranked by F1:")
print(runs[['run_id', 'tags.mlflow.runName', 'metrics.f1', 'metrics.recall',
            'metrics.precision', 'metrics.pr_auc']].to_string(index=False))

best = runs.iloc[0]
print("\nBest run:", best['tags.mlflow.runName'], best['run_id'])

# register
model_uri = f"runs:/{best['run_id']}/model"
registered = mlflow.register_model(model_uri, MODEL_NAME)
print(f"Registered {MODEL_NAME} version {registered.version}")

# tag it as staging
client = MlflowClient()
client.set_model_version_tag(MODEL_NAME, registered.version, "stage", "Staging")
client.set_model_version_tag(MODEL_NAME, registered.version, "f1", f"{best['metrics.f1']:.4f}")
print("Tagged as Staging")
