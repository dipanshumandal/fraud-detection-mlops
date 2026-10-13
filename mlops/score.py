# scoring script for the azure online endpoint
import os
import json
import mlflow.sklearn
import pandas as pd

model = None


def init():
    global model
    # azure puts the registered model under AZUREML_MODEL_DIR
    model_dir = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")
    model = mlflow.sklearn.load_model(model_dir)
    print("Model loaded from", model_dir)


def run(raw_data):
    data = json.loads(raw_data)["input_data"]
    df = pd.DataFrame(data["data"], columns=data["columns"])

    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1]

    return {
        "predictions": prediction.tolist(),
        "fraud_probability": [round(float(p), 4) for p in probability]
    }
