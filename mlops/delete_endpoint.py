# delete the endpoint when done - it costs money while it is running
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from config import *

ml_client = MLClient(DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)
ml_client.online_endpoints.begin_delete(ENDPOINT_NAME).result()
print("Endpoint deleted")
