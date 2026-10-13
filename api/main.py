import os
import requests
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI(
    title="Online Fraud Detection API",
    version="1.0"
)

model = joblib.load("./model/fraud_model.pkl")


# input is the raw transaction, same columns as the dataset
class Transaction(BaseModel):
    step: int = Field(..., ge=1)
    type: Literal['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    amount: float = Field(..., ge=0)
    nameOrig: str
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    nameDest: str
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)


# same feature engineering as train.py
def make_features(df):
    df['hour'] = df['step'] % 24
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['accountEmptied'] = ((df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)).astype(int)
    df['amountRatio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['isNight'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    df['destIsMerchant'] = df['nameDest'].str.startswith('M').astype(int)
    df = df.drop(['step', 'nameOrig', 'nameDest'], axis=1)
    return df


@app.get("/")
def home():
    return {
        "message": "Online Fraud Detection API"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(data: Transaction):
    input_data = pd.DataFrame([data.model_dump()])
    input_data = make_features(input_data)

    # make sure columns are in the same order as training
    input_data = input_data[model.feature_names_in_]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "is_fraud": bool(prediction == 1),
        "fraud_probability": round(float(probability), 4)
    }


# calls the model deployed on Azure ML instead of the local pkl
# set AZURE_ENDPOINT_URL and AZURE_ENDPOINT_KEY as environment variables before starting uvicorn
@app.post("/predict-azure")
def predict_azure(data: Transaction):
    url = os.environ.get("AZURE_ENDPOINT_URL")
    key = os.environ.get("AZURE_ENDPOINT_KEY")
    if not url or not key:
        return {"error": "AZURE_ENDPOINT_URL / AZURE_ENDPOINT_KEY not set"}

    input_data = pd.DataFrame([data.model_dump()])
    input_data = make_features(input_data)
    input_data = input_data[model.feature_names_in_]

    payload = {
        "input_data": {
            "columns": input_data.columns.tolist(),
            "data": input_data.values.tolist()
        }
    }
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()

    return {
        "prediction": int(result["predictions"][0]),
        "is_fraud": bool(result["predictions"][0] == 1),
        "fraud_probability": result["fraud_probability"][0],
        "source": "azure-ml-endpoint"
    }
