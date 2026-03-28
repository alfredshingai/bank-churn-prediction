# BANK CHURN PROJECT — Step 1: Data Cleaning & Merging

# Dataset : Bank_Churn_Messy.xlsx (two sheets)
# Target  : Exited (1 = churned, 0 = stayed)


import pandas as pd

# 1. Load both sheets 
customer_df = pd.read_excel("Bank_Churn_Messy.xlsx", sheet_name="Customer_Info")
account_df  = pd.read_excel("Bank_Churn_Messy.xlsx", sheet_name="Account_Info")

print("Customer_Info shape:", customer_df.shape)   # (10001, 8)
print("Account_Info shape :", account_df.shape)    # (10002, 7)


# 2. Fix Account_Info 

# 2a. Remove fully duplicate rows (2 found during inspection)
account_df = account_df.drop_duplicates()

# 2b. If a CustomerId still appears twice keep the first occurrence
account_df = account_df.drop_duplicates(subset="CustomerId", keep="first")

# 2c. Strip the € symbol from Balance and cast to float
account_df["Balance"] = (
    account_df["Balance"]
    .str.replace("€", "", regex=False)
    .astype(float)
)

# 2d. Encode Yes/No columns as 0/1
account_df["IsActiveMember"] = account_df["IsActiveMember"].map({"Yes": 1, "No": 0})
account_df["HasCrCard"]      = account_df["HasCrCard"].map({"Yes": 1, "No": 0})

# 2e. Drop Tenure from Account_Info — it's already in Customer_Info
#     (same values; keeping one avoids a merge conflict)
account_df = account_df.drop(columns=["Tenure"])


# 3. Fix Customer_Info 

# 3a. Drop rows where Age is missing (only 3 rows — safe to remove)
customer_df = customer_df.dropna(subset=["Age"])

# 3b. Normalise Geography typos: 'FRA' and 'French' → 'France'
geo_map = {"FRA": "France", "French": "France"}
customer_df["Geography"] = customer_df["Geography"].replace(geo_map)

# 3c. Strip € from EstimatedSalary and cast to float
customer_df["EstimatedSalary"] = (
    customer_df["EstimatedSalary"]
    .str.replace("€", "", regex=False)
    .astype(float)
)

# 3d. Drop Surname — not useful for modelling
customer_df = customer_df.drop(columns=["Surname"])


# 4. Merge the two sheets on CustomerId 
df = pd.merge(customer_df, account_df, on="CustomerId", how="inner")

# CustomerId is no longer needed after the join
df = df.drop(columns=["CustomerId"])


# 5. Sanity checks
print("\n── Final dataset ──")
print("Shape   :", df.shape)          # Expected: (9998, 11)
print("Columns :", df.columns.tolist())
print("\nNull counts:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

# Target class balance
churn_rate = df["Exited"].mean() * 100
print(f"\nChurn rate: {churn_rate:.1f}%")

# 6. Save clean file for next steps
df.to_csv("bank_churn_clean.csv", index=False)
print("\nSaved → bank_churn_clean.csv")
