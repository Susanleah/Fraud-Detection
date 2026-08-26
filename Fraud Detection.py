"""
fraud_Detection.py
Fraud Detection in Card Transactions: A Machine Learning Approach
Using Linked Customer, Card, and Merchant Data

Step 1: Load & inspect raw data
Step 2: Merge into one dataset
Step 3: Feature engineering
Step 4: Encode categorical data + train/test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Modeling Libraries 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import joblib
import json


# ==========================================================
# STEP 1: Load raw data
# ==========================================================
transactions = pd.read_csv("/home/dennis/Documents/Fraud Detection/Transaction_Data_250k.csv")
customers = pd.read_csv("/home/dennis/Documents/Fraud Detection/Cusmtomer_data.csv")
cards = pd.read_csv("/home/dennis/Documents/Fraud Detection/Cards_Data.csv")
merchants = pd.read_csv("/home/dennis/Documents/Fraud Detection/merchant_table.csv")

print("Raw shapes:")
print("Transactions:", transactions.shape)
print("Customers:", customers.shape)
print("Cards:", cards.shape)
print("Merchants:", merchants.shape)

# ==========================================================
# STEP 2: Merge into one dataset
# ==========================================================
df = transactions.merge(
    customers.drop(columns=["Customer_Name", "State", "City"]),
    on="Customer_ID", how="left"
)
df = df.merge(
    cards.drop(columns=["Customer_ID"]),
    on="Card_ID", how="left"
)
df = df.merge(
    merchants.drop(columns=["Merchant_Name", "Merchant_Category", "State", "City", "Merchant_Risk_Level"]),
    on="Merchant_ID", how="left"
)

print("\nMerged shape:", df.shape)
df.to_csv("merged_data.csv", index=False)
print("Saved -> merged_data.csv")

# ==========================================================
# STEP 3: Feature engineering
# ==========================================================

# Fix data quality issue
df["Marital_Status"] = df["Marital_Status"].replace("Divorsed", "Divorced")

# Parse date/time columns
df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
df["Transaction_Hour"] = pd.to_datetime(df["Transaction_Time"], format="%H:%M:%S", errors="coerce").dt.hour
df["Transaction_DayOfWeek"] = df["Transaction_Date"].dt.dayofweek
df["Is_Weekend"] = df["Transaction_DayOfWeek"].isin([5, 6]).astype(int)
df["Is_Night_Transaction"] = df["Transaction_Hour"].apply(
    lambda h: 1 if pd.notnull(h) and (h >= 23 or h <= 5) else 0
)

# Card expiry check
df["Expiry_Date"] = pd.to_datetime(df["Expiry_Date"], errors="coerce")
df["Is_Card_Expired"] = (df["Transaction_Date"] > df["Expiry_Date"]).astype(int)

# Customer tenure at time of transaction
df["Customer_Since"] = pd.to_datetime(df["Customer_Since"], errors="coerce")
df["Customer_Tenure_Days"] = (df["Transaction_Date"] - df["Customer_Since"]).dt.days.clip(lower=0)

# Amount relative to credit limit
df["Amount_to_CreditLimit_Ratio"] = (
    df["Transaction_Amount"] / df["Credit_Limit"].replace(0, np.nan)
).fillna(0)

# Customer spending behavior (historical average & transaction count)
cust_stats = df.groupby("Customer_ID")["Transaction_Amount"].agg(
    Customer_Avg_Amount="mean", Customer_Txn_Count="count"
).reset_index()
df = df.merge(cust_stats, on="Customer_ID", how="left")

# Merchant historical fraud rate (risk signal)
merchant_fraud_rate = df.groupby("Merchant_ID")["Fraud_Flag"].mean().rename("Merchant_Fraud_Rate")
df = df.merge(merchant_fraud_rate, on="Merchant_ID", how="left")

print("\nEngineered feature check:")
print(df[["Transaction_Hour", "Is_Weekend", "Is_Night_Transaction",
          "Is_Card_Expired", "Customer_Tenure_Days", "Amount_to_CreditLimit_Ratio",
          "Customer_Avg_Amount", "Merchant_Fraud_Rate"]].describe())
print("\nMarital_Status values:", df["Marital_Status"].unique())

df.to_csv("featured_data.csv", index=False)
print("Saved -> featured_data.csv")

# ==========================================================
# STEP 4: Encode categorical data + train/test split
# ==========================================================

# Columns to exclude from modeling:
# - IDs: not predictive, just identifiers
# - Raw date/time columns: already captured by engineered features
# - Fraud_Reason: only filled in for fraud cases -> would leak the answer
# - Customer_City / Merchant_City / Merchant_Since: too many unique values
drop_cols = [
    "Transaction_ID", "Customer_ID", "Card_ID", "Merchant_ID",
    "Transaction_Date", "Transaction_Time", "Issue_Date", "Expiry_Date", "Customer_Since",
    "Fraud_Reason", "Customer_City", "Merchant_City", "Merchant_Since",
]

target = "Fraud_Flag"
X = df.drop(columns=drop_cols + [target])
Y = df[target]

print(X)
print(Y)

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()
print("\nCategorical columns:", categorical_cols)
print("\nNumeric columns:", numeric_cols)

X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print("\nEncoded shape:", X_encoded.shape)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_encoded, Y, test_size=0.2, random_state=42, stratify=Y
)

print("\nTrain shape:", X_train.shape, "| Test shape:", X_test.shape)
print("Train fraud rate:", round(Y_train.mean(), 4))
print("Test fraud rate:", round(Y_test.mean(), 4))

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
Y_train.to_csv("Y_train.csv", index=False)
Y_test.to_csv("Y_test.csv", index=False)
print("\nSaved train/test splits.")

# ==========================================================
# STEP 5: Train models, compare, and save best one
# ==========================================================

# Logistic Regression needs scaled features to converge properly and perform well
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train_scaled, Y_train)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                             random_state=42, n_jobs=-1)
rf.fit(X_train, Y_train)

# XGBOOST

scale_pos_weight = (Y_train == 0).sum() / (Y_train == 1).sum()
xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
                     scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                     random_state=42, n_jobs=-1)
xgb.fit(X_train, Y_train)

# Evaluate all 3 models
model_results = []
for name, proba in [("Logistic Regression", lr.predict_proba(X_test_scaled)[:, 1]),
                     ("Random Forest", rf.predict_proba(X_test)[:, 1]),
                     ("XGBoost", xgb.predict_proba(X_test)[:, 1])]:
    pred = (proba >= 0.5).astype(int)
    print(f"\n{name}:")
    print("ROC-AUC:", roc_auc_score(Y_test, proba))
    print("PR-AUC:", average_precision_score(Y_test, proba))
    print(classification_report(Y_test, pred))
    model_results.append({
        "name": name,
        "confusion_matrix": confusion_matrix(Y_test, pred).tolist(),
        "roc_auc": roc_auc_score(Y_test, proba),
        "pr_auc": average_precision_score(Y_test, proba),
    })

# Pick best model by PR-AUC 
best = max(model_results, key=lambda r: r["pr_auc"])
print(f"\nBest model: {best['name']} (PR-AUC={best['pr_auc']:.4f})")

# Save best model, scaler, and results 
best_models = {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb}
joblib.dump(best_models[best["name"]], "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

with open("model_results.json", "w") as f:
    json.dump({"results": model_results, "best_model": best["name"]}, f, indent=2)

print("Saved best_model.pkl, scaler.pkl, model_results.json")












