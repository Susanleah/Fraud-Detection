"""
fraud_Detection.py
Step 1: Load the 4 raw CSV files and inspect them before doing anything else.
"""

import pandas as pd

# Load each raw table
transactions = pd.read_csv("/home/dennis/Documents/Fraud Detection/Transaction_Data_250k.csv")
customers = pd.read_csv("/home/dennis/Documents/Fraud Detection/Cusmtomer_data.csv")
cards = pd.read_csv("/home/dennis/Documents/Fraud Detection/Cards_Data.csv")
merchants = pd.read_csv("/home/dennis/Documents/Fraud Detection/merchant_table.csv")

datasets = {
    "Transactions": transactions,
    "Customers": customers,
    "Cards": cards,
    "Merchants": merchants,
}

for name, df in datasets.items():
    print("=" * 60)
    print(f"{name}  ->  shape: {df.shape}")
    print("=" * 60)
    print(df.head(3))
    print("\n--- Column info ---")
    print(df.info())
    print("\n--- Missing values ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("\n")

# Step 2: Merge all 4 tables into a single dataset.

# --- Merge transactions -> customers (on Customer_ID) ---
df = transactions.merge(
    customers.drop(columns=["Customer_Name", "State", "City"]),  # drop duplicates/PII not needed for modeling
    on="Customer_ID", how="left"
)

