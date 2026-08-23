"""
fraud_Detection.py
Step 1: Load the 4 raw CSV files and inspect them before doing anything else.
"""
# Load the necessary libraries
import pandas as pd
import numpy as np


# Load the raw table files
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
# Inspect the datasets 
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

# Step 2 : Merge the 4 tables into a single table for analysis. The final table should have the following column

# Merge transactions -> customers (on Customer_ID)
df = transactions.merge(
    customers.drop(columns=["Customer_Name", "State", "City"]),  # drop duplicates/PII not needed for modeling
    on="Customer_ID", how="left"
)

# Merge -> cards (on Card_ID) 
df = df.merge(
    cards.drop(columns=["Customer_ID"]),  # already have Customer_ID from transactions
    on="Card_ID", how="left"
)

# Merge -> merchants (on Merchant_ID) 
df = df.merge(
    merchants.drop(columns=["Merchant_Name", "Merchant_Category", "State", "City", "Merchant_Risk_Level"]),
    on="Merchant_ID", how="left"
)

print("Merged shape:", df.shape)
print(df.columns.tolist())
print("\nMissing values after merge:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Save the merged dataset so we don't have to redo this every time
df.to_csv("/home/dennis/Documents/Fraud Detection/merged_data.csv", index=False)
print("\nSaved merged dataset -> /home/dennis/Documents/Fraud Detection/merged_data.csv")

# Step 3: Feature Engineering

#Fix data quality issue
df["Marital_Status"] = df["Marital_Status"].replace("Divorsed", "Divorced")  # Fix typo in Marital_Status

#Parse date/time columns 
df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")  # Convert to datetime, coerce errors to NaT
df["Transaction_hour"] = pd.to_datetime(df["Transaction_Date"] , format="%H:%M:%S", errors="coerce").dt.hour  # Extract hour from Transaction_Date
