import joblib
import pandas as pd

model = joblib.load("./model/fraud_model.pkl")

# load the sample rows saved by train.py
x_test = pd.read_csv("./data/x_test_sample.csv")
y_test = pd.read_csv("./data/y_test_sample.csv")

print(x_test.columns.tolist())
print(x_test.head())

yhat = model.predict(x_test)
prob = model.predict_proba(x_test)[:, 1]

result = pd.DataFrame({
    'type': x_test['type'],
    'amount': x_test['amount'],
    'actual': y_test['isFraud'],
    'predicted': yhat,
    'fraud_probability': prob.round(4)
})
print(result)

print(f"\nCorrect: {(result['actual'] == result['predicted']).sum()} / {len(result)}")

# single random prediction like the loan project
sample = x_test.sample(1)
print("\nRandom sample prediction:", model.predict(sample))
