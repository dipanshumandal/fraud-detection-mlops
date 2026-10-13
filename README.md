# Online Fraud Transaction Detection - MLOps

Capstone Project 1 (Option A). Fraud detection model on the PaySim online transactions dataset
(6.3M transactions, 0.13% fraud), deployed with FastAPI, tested with GitHub Actions CI and
tracked / registered / deployed on Azure ML with MLflow.

```
Capstone-1 MLOps/
│
├── data
|    ├── onlinefraud.csv        # dataset (not pushed to git, too big)
|    ├── x_test_sample.csv      # sample rows for testing
|    ├── y_test_sample.csv
├── model
|    ├── fraud_model.pkl        # best model (Random Forest)
|    ├── model_comparison.csv
├── plots                       # EDA and evaluation plots
├── src
|    ├── train.py               # EDA, feature engineering, 3 models, grid search, evaluation
|    ├── prediction.py          # loads the pkl and predicts on sample rows
├── api
|    ├── main.py                # FastAPI - /predict (local model), /predict-azure (azure endpoint)
├── tests
|    ├── test_api.py            # pytest cases for the API
├── mlops
|    ├── config.py              # azure workspace settings
|    ├── azure_setup.py         # workspace + compute
|    ├── mlflow_tracking.py     # log runs to MLflow on Azure
|    ├── register_model.py      # register best run
|    ├── deploy_model.py        # managed online endpoint
|    ├── score.py               # scoring script for the endpoint
|    ├── conda.yaml             # endpoint environment
|    ├── test_endpoint.py       # call the endpoint with the key
|    ├── delete_endpoint.py
├── .github/workflows
|    ├── ci.yml                 # flake8 + pytest on every push / PR
├── requirements.txt
├── README.md
```

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Section 1 - Train

```
python src/train.py
python src/prediction.py
```

## Section 2 - API and CI

```
uvicorn api.main:app --reload
```
Open http://127.0.0.1:8000/docs

```
pytest -v tests/
```

CI runs flake8, pytest and the prediction script on every push and pull request to main.

## Section 3 - Azure ML + MLflow

```
az login
az group create --name rg-fraud-mlops --location eastus
python mlops/azure_setup.py        # workspace, compute instance, compute cluster
python mlops/mlflow_tracking.py    # 3 runs with params, metrics, plots, model
python mlops/register_model.py     # best run -> model registry (Staging tag)
python mlops/deploy_model.py       # managed online endpoint
python mlops/test_endpoint.py      # REST call with key
python mlops/delete_endpoint.py    # stop billing when done
```

To call the Azure endpoint through the FastAPI service:

```
AZURE_ENDPOINT_URL="<scoring uri>" AZURE_ENDPOINT_KEY="<key>" uvicorn api.main:app
```
then use POST /predict-azure.

### Monitoring and retraining

- Application Insights is enabled on the endpoint and logs every request (count, latency, errors).
- Data drift: compare live feature distributions with the training data stats every week,
  retrain if the drift on key features (errorBalanceOrig, amount, accountEmptied) crosses a threshold.
- Scheduled retraining: monthly job on the compute cluster with the latest data, register the new version.
- Blue/green rollout: deploy the new version as "green" with 0% traffic, test it, move traffic
  10% -> 50% -> 100%, keep "blue" around to roll back if errors go up.
