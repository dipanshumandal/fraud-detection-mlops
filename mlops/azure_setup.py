# Step 1 - create the Azure ML workspace, compute instance and compute cluster
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace, ComputeInstance, AmlCompute
from azure.identity import DefaultAzureCredential
from config import *

credential = DefaultAzureCredential()
ml_client = MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP)

# workspace - this also creates storage account, key vault, app insights and container registry
print("Creating workspace...")
ws = Workspace(
    name=WORKSPACE_NAME,
    location=LOCATION,
    display_name="Fraud Detection MLOps",
    description="Capstone project - online fraud detection"
)
ws = ml_client.workspaces.begin_create(ws).result()
print("Workspace ready:", ws.name)

# now connect to the workspace
ml_client = MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)

# compute instance (for notebooks / dev)
print("Creating compute instance...")
ci = ComputeInstance(name=COMPUTE_INSTANCE, size=VM_SIZE)
ml_client.compute.begin_create_or_update(ci).result()
print("Compute instance ready:", COMPUTE_INSTANCE)

# compute cluster (for training jobs), scales down to 0 when idle so no cost
print("Creating compute cluster...")
cluster = AmlCompute(
    name=COMPUTE_CLUSTER,
    size=VM_SIZE,
    min_instances=0,
    max_instances=2,
    idle_time_before_scale_down=120
)
ml_client.compute.begin_create_or_update(cluster).result()
print("Compute cluster ready:", COMPUTE_CLUSTER)

print("\nMLflow tracking URI:", ml_client.workspaces.get(WORKSPACE_NAME).mlflow_tracking_uri)
