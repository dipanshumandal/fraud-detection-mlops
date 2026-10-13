# Online Fraud Transaction Detection - MLOps

Capstone Project 1 (Option A). Fraud detection model on the PaySim online transactions dataset,
deployed with FastAPI, tested with GitHub Actions CI and tracked/deployed on Azure ML with MLflow.

```
Capstone-1 MLOps/
│
├── data
|    ├── onlinefraud.csv        # dataset (not pushed to git, too big)
|    ├── x_test_sample.csv      # sample rows for testing
├── model
|    ├── fraud_model.pkl
├── plots                       # EDA and evaluation plots
├── src
|    ├── train.py
|    ├── prediction.py
├── api
|    ├── main.py
├── tests
|    ├── test_api.py
├── .github/workflows
|    ├── ci.yml
├── requirements.txt
├── README.md
```

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train

```
python src/train.py
```

## Run API

```
uvicorn api.main:app --reload
```
Open http://127.0.0.1:8000/docs
