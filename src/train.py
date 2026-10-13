# 1. Import Packages
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             classification_report, RocCurveDisplay, PrecisionRecallDisplay,
                             ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 2. Read Data
fraud_df = pd.read_csv("./data/onlinefraud.csv")

print(fraud_df.head())
print(fraud_df.shape)
print(fraud_df.info())
print(fraud_df.describe())
print("\nMissing values:\n", fraud_df.isnull().sum())
print("\nDuplicate rows:", fraud_df.duplicated().sum())

# 3. Data Cleaning
fraud_df.drop_duplicates(inplace=True)

# isFlaggedFraud is just a business rule (transfer > 200000), only 16 rows have it set
# so it is not useful for the model
fraud_df.drop('isFlaggedFraud', axis=1, inplace=True)

# NOTE: not removing outliers in amount. Fraud transactions have much bigger amounts
# so the outliers are actually the fraud signal
print("\nFraud amount stats:\n", fraud_df[fraud_df['isFraud'] == 1]['amount'].describe())
print("\nNormal amount stats:\n", fraud_df[fraud_df['isFraud'] == 0]['amount'].describe())

# 4. EDA
print("\nClass balance:\n", fraud_df['isFraud'].value_counts())
print("Fraud %:", round(fraud_df['isFraud'].mean() * 100, 3))
print("\nFraud by transaction type:\n", pd.crosstab(fraud_df['type'], fraud_df['isFraud']))

plt.figure(figsize=(6, 4))
sns.countplot(x='isFraud', data=fraud_df)
plt.yscale('log')
plt.title('Class Balance (log scale)')
plt.savefig('./plots/class_balance.png')
plt.close()

plt.figure(figsize=(8, 4))
sns.countplot(x='type', hue='isFraud', data=fraud_df)
plt.yscale('log')
plt.title('Transaction Type vs Fraud')
plt.savefig('./plots/type_vs_fraud.png')
plt.close()

fraud_df['hour'] = fraud_df['step'] % 24
hourly = fraud_df.groupby('hour')['isFraud'].mean()
plt.figure(figsize=(8, 4))
hourly.plot()
plt.title('Fraud rate by hour of day')
plt.savefig('./plots/fraud_by_hour.png')
plt.close()

plt.figure(figsize=(8, 4))
sns.histplot(np.log1p(fraud_df[fraud_df['isFraud'] == 0]['amount']), color='blue', label='Normal', stat='density', bins=50)
sns.histplot(np.log1p(fraud_df[fraud_df['isFraud'] == 1]['amount']), color='red', label='Fraud', stat='density', bins=50)
plt.legend()
plt.title('log(amount) distribution')
plt.savefig('./plots/amount_dist.png')
plt.close()

# 5. Feature Engineering
# balance difference - in fraud the balances dont add up
fraud_df['errorBalanceOrig'] = fraud_df['newbalanceOrig'] + fraud_df['amount'] - fraud_df['oldbalanceOrg']
fraud_df['errorBalanceDest'] = fraud_df['oldbalanceDest'] + fraud_df['amount'] - fraud_df['newbalanceDest']
# most frauds empty the account completely
fraud_df['accountEmptied'] = ((fraud_df['newbalanceOrig'] == 0) & (fraud_df['oldbalanceOrg'] > 0)).astype(int)
fraud_df['amountRatio'] = fraud_df['amount'] / (fraud_df['oldbalanceOrg'] + 1)
fraud_df['isNight'] = ((fraud_df['hour'] >= 0) & (fraud_df['hour'] <= 5)).astype(int)
# destination starting with M is a merchant
fraud_df['destIsMerchant'] = fraud_df['nameDest'].str.startswith('M').astype(int)

print("\nCorrelation with isFraud:")
num_cols = fraud_df.select_dtypes(include='number').columns
corr = fraud_df[num_cols].corr()
print(corr['isFraud'].sort_values(ascending=False))

plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('./plots/correlation.png')
plt.close()

# 6. Features & Target
# dropping account ids - almost every nameOrig is unique so it cant be learned
X = fraud_df.drop(['isFraud', 'nameOrig', 'nameDest', 'step'], axis=1)
Y = fraud_df['isFraud']

print(X.head())
print(X.columns.tolist())

# 7. Split Data into Train & Test (stratified because of imbalance)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
print("Train:", X_train.shape, "Test:", X_test.shape)

# preprocessing - one hot for type, scaling for numbers
cat_cols = ['type']
numeric_cols = [c for c in X.columns if c != 'type']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), numeric_cols)
])

# 8. Handling Imbalanced Data - comparing different methods
# using a smaller sample here because SMOTE on 5 million rows needs too much memory
sample_idx = Y_train[Y_train == 1].index.tolist() + Y_train[Y_train == 0].sample(500000, random_state=42).index.tolist()
X_sample = X_train.loc[sample_idx]
Y_sample = Y_train.loc[sample_idx]

X_sample_prep = preprocessor.fit_transform(X_sample)
X_test_prep = preprocessor.transform(X_test)

print("\n--- Imbalance handling comparison (Logistic Regression) ---")
methods = {
    'No resampling': (X_sample_prep, Y_sample),
    'SMOTE': SMOTE(random_state=42).fit_resample(X_sample_prep, Y_sample),
    'Under sampling': RandomUnderSampler(random_state=42).fit_resample(X_sample_prep, Y_sample),
}
for name, (xs, ys) in methods.items():
    lr = LogisticRegression(max_iter=1000)
    lr.fit(xs, ys)
    pred = lr.predict(X_test_prep)
    print(f"{name:16s} recall={recall_score(Y_test, pred):.3f} precision={precision_score(Y_test, pred):.3f} f1={f1_score(Y_test, pred):.3f}")

lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_sample_prep, Y_sample)
pred = lr.predict(X_test_prep)
print(f"{'Class weight':16s} recall={recall_score(Y_test, pred):.3f} precision={precision_score(Y_test, pred):.3f} f1={f1_score(Y_test, pred):.3f}")

# 9. Model Training + Hyperparameter Tuning
# GridSearch on the sample (full data takes too long), then refit best params on full train set
models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, class_weight='balanced'),
                            {'model__C': [0.1, 1, 10]}),
    'Decision Tree': (DecisionTreeClassifier(class_weight='balanced', random_state=42),
                      {'model__max_depth': [6, 10, 15], 'model__min_samples_leaf': [10, 50]}),
    'Random Forest': (RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42),
                      {'model__n_estimators': [100], 'model__max_depth': [10, 20], 'model__min_samples_leaf': [5]}),
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = []
trained = {}

for name, (clf, params) in models.items():
    print(f"\n=== {name} ===")
    pipe = Pipeline([('prep', preprocessor), ('model', clf)])
    grid = GridSearchCV(pipe, params, cv=cv, scoring='average_precision', n_jobs=1)
    grid.fit(X_sample, Y_sample)
    print("Best params:", grid.best_params_)
    print("Best CV PR-AUC:", round(grid.best_score_, 4))

    best = grid.best_estimator_
    best.fit(X_train, Y_train)
    trained[name] = best

    # 10. Evaluation
    Y_pred = best.predict(X_test)
    Y_prob = best.predict_proba(X_test)[:, 1]

    acc = accuracy_score(Y_test, Y_pred)
    prec = precision_score(Y_test, Y_pred)
    rec = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    roc = roc_auc_score(Y_test, Y_prob)
    pr_auc = average_precision_score(Y_test, Y_prob)
    results.append([name, acc, prec, rec, f1, roc, pr_auc])

    print(classification_report(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ConfusionMatrixDisplay(confusion_matrix(Y_test, Y_pred)).plot(ax=ax[0])
    ax[0].set_title(name + ' - Confusion Matrix')
    RocCurveDisplay.from_predictions(Y_test, Y_prob, ax=ax[1])
    ax[1].set_title('ROC Curve')
    PrecisionRecallDisplay.from_predictions(Y_test, Y_prob, ax=ax[2])
    ax[2].set_title('Precision-Recall Curve')
    plt.tight_layout()
    plt.savefig('./plots/' + name.lower().replace(' ', '_') + '_eval.png')
    plt.close()

# 11. Model Comparison
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC'])
print("\n", results_df.round(4))
results_df.to_csv('./model/model_comparison.csv', index=False)

# picking by F1 - accuracy is useless here, 99.87% accuracy just by predicting 0 always
best_name = results_df.sort_values('F1', ascending=False).iloc[0]['Model']
best_model = trained[best_name]
print("\nBest model:", best_name)

# feature importance
if hasattr(best_model.named_steps['model'], 'feature_importances_'):
    feat_names = best_model.named_steps['prep'].get_feature_names_out()
    importance = pd.Series(best_model.named_steps['model'].feature_importances_, index=feat_names)
    importance = importance.sort_values(ascending=False)
    print("\nFeature importance:\n", importance)
    plt.figure(figsize=(8, 6))
    importance.plot(kind='barh')
    plt.title('Feature Importance - ' + best_name)
    plt.tight_layout()
    plt.savefig('./plots/feature_importance.png')
    plt.close()

# 12. Save Model
joblib.dump(best_model, "./model/fraud_model.pkl")
print("\nModel Saved Successfully!")

# save some test rows for prediction.py and the api tests
X_test.head(20).to_csv("./data/x_test_sample.csv", index=False)
Y_test.head(20).to_csv("./data/y_test_sample.csv", index=False)

# check reload gives same answers
loaded = joblib.load("./model/fraud_model.pkl")
print("Reload check:", (loaded.predict(X_test.head(20)) == best_model.predict(X_test.head(20))).all())
