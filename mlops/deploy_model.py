# Step 4 - deploy the registered model as a managed online endpoint
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Environment, CodeConfiguration
from azure.identity import DefaultAzureCredential
from config import *

ml_client = MLClient(DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)

# endpoint
print("Creating endpoint...")
endpoint = ManagedOnlineEndpoint(
    name=ENDPOINT_NAME,
    description="Online fraud detection - real time scoring",
    auth_mode="key"
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Endpoint ready:", ENDPOINT_NAME)

# latest registered model version
versions = [int(m.version) for m in ml_client.models.list(name=MODEL_NAME)]
model = ml_client.models.get(name=MODEL_NAME, version=str(max(versions)))
print("Deploying model version", model.version)

# own environment + scoring script for the container
env = Environment(
    name="fraud-endpoint-env",
    conda_file=os.path.join(os.path.dirname(__file__), "conda.yaml"),
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
)

deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=ENDPOINT_NAME,
    model=model,
    environment=env,
    code_configuration=CodeConfiguration(
        code=os.path.dirname(__file__),
        scoring_script="score.py"
    ),
    instance_type=ENDPOINT_VM_SIZE,
    instance_count=1,
    app_insights_enabled=True
)
ml_client.online_deployments.begin_create_or_update(deployment).result()

# send all traffic to blue
endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print("Deployment done")
print("Scoring URI:", endpoint.scoring_uri)
