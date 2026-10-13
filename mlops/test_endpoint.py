# Step 5 - call the deployed endpoint with the REST key
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import requests
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from config import *

ml_client = MLClient(DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)

endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
key = ml_client.online_endpoints.get_keys(ENDPOINT_NAME).primary_key

# same test rows saved by train.py
x_test = pd.read_csv("./data/x_test_sample.csv").head(5)
y_test = pd.read_csv("./data/y_test_sample.csv").head(5)

payload = {
    "input_data": {
        "columns": x_test.columns.tolist(),
        "data": x_test.values.tolist()
    }
}

headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
response = requests.post(endpoint.scoring_uri, headers=headers, data=json.dumps(payload))

print("Status:", response.status_code)
print("Predictions:", response.json())
print("Actual:     ", y_test['isFraud'].tolist())
print("\nScoring URI:", endpoint.scoring_uri)
print("(key printed below for Postman, do not commit it)")
print("Key:", key)
