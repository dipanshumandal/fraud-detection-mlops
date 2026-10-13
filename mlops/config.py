# azure settings - fill subscription id from `az account show`
SUBSCRIPTION_ID = "88782a35-2f03-4c87-ad6d-e339e2982983"
RESOURCE_GROUP = "rg-fraud-mlops"
WORKSPACE_NAME = "fraud-mlops-ws"
LOCATION = "eastus"

COMPUTE_INSTANCE = "fraud-ci"
COMPUTE_CLUSTER = "cpu-cluster"
VM_SIZE = "Standard_DS3_v2"
# smaller size for the endpoint - subscription quota is only 6 cores and endpoints reserve 2x
ENDPOINT_VM_SIZE = "Standard_DS2_v2"

EXPERIMENT_NAME = "fraud-detection"
MODEL_NAME = "fraud-detection-model"
ENDPOINT_NAME = "fraud-detect-endpoint"
