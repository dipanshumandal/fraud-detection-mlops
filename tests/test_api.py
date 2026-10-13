import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

normal_txn = {
    "step": 1,
    "type": "PAYMENT",
    "amount": 9839.64,
    "nameOrig": "C1231006815",
    "oldbalanceOrg": 170136.0,
    "newbalanceOrig": 160296.36,
    "nameDest": "M1979787155",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
}

fraud_txn = {
    "step": 212,
    "type": "TRANSFER",
    "amount": 1250000.0,
    "nameOrig": "C1305486145",
    "oldbalanceOrg": 1250000.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C553264065",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
}


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Online Fraud Detection API"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] == True


def test_predict_normal():
    response = client.post("/predict", json=normal_txn)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert 0 <= data["fraud_probability"] <= 1


def test_predict_fraud():
    response = client.post("/predict", json=fraud_txn)
    assert response.status_code == 200
    assert response.json()["is_fraud"] == True


def test_fraud_scores_higher_than_normal():
    p_normal = client.post("/predict", json=normal_txn).json()["fraud_probability"]
    p_fraud = client.post("/predict", json=fraud_txn).json()["fraud_probability"]
    assert p_fraud > p_normal


def test_missing_field():
    txn = normal_txn.copy()
    del txn["amount"]
    response = client.post("/predict", json=txn)
    assert response.status_code == 422


def test_wrong_data_type():
    txn = normal_txn.copy()
    txn["amount"] = "abc"
    response = client.post("/predict", json=txn)
    assert response.status_code == 422


def test_negative_amount():
    txn = normal_txn.copy()
    txn["amount"] = -100
    response = client.post("/predict", json=txn)
    assert response.status_code == 422


def test_unknown_transaction_type():
    txn = normal_txn.copy()
    txn["type"] = "CRYPTO"
    response = client.post("/predict", json=txn)
    assert response.status_code == 422
