import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import re

print("--- Starting Phase 1 & 2: Data Preprocessing Pipeline ---")

# 1. Load the raw data
file_path = r"C:\Users\Satyartha Shukla\Desktop\home-credit-default-risk\application_train.csv"
print("Loading raw dataset...")
df = pd.read_csv(file_path)

# 2. Feature Engineering
print("Engineering financial features...")
df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

# 3. Drop missing columns
print("Dropping columns with >50% missing data...")
threshold = 50
missing_percentages = (df.isnull().sum() / len(df)) * 100
cols_to_drop = missing_percentages[missing_percentages > threshold].index
df_clean = df.drop(columns=cols_to_drop)

# 4. Impute missing values
print("Imputing remaining missing values...")
for col in df_clean.columns:
    if is_numeric_dtype(df_clean[col]):
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    else:
        df_clean[col] = df_clean[col].fillna('Unknown')

# 5. One-Hot Encoding
print("Encoding categorical variables...")
df_encoded = pd.get_dummies(df_clean, drop_first=True)

# 6. FIX THE LIGHTGBM ERROR: Clean the column names
print("Cleaning column names for LightGBM compatibility...")
# This replaces any special JSON/Regex characters with an underscore
df_encoded.columns = df_encoded.columns.str.replace(r'[^A-Za-z0-9_]', '_', regex=True)

# 7. Save the checkpoint file
output_file = "cleaned_banking_data.csv"
print(f"Saving cleaned dataset to {output_file} (This may take a minute)...")
df_encoded.to_csv(output_file, index=False)

print("\n--- Pipeline Complete! You can now run 02_model_training.py ---")