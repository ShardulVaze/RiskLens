import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from database.db_config import DATABASE_URL

print("START — Loading data from DB...")

engine = create_engine(DATABASE_URL)
df = pd.read_sql("SELECT * FROM vw_risk_model_dataset", engine)

print(f"Loaded {len(df)} rows and {len(df.columns)} columns")

df = df.drop(columns=["SK_ID_CURR"])

X = df.drop(columns=["TARGET"])
y = df["TARGET"]

print(f"Class distribution — Defaulters: {y.sum()}, Non-defaulters: {(y==0).sum()}")
print(f"Default rate: {round(y.mean() * 100, 2)}%")

print("Filling nulls with median...")
for col in X.columns:
    if X[col].isnull().sum() > 0:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        print(f"  {col}: filled {X[col].isnull().sum()} nulls with median {round(median_val, 4)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nApplying SMOTE")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print(f"Before SMOTE — Defaulters: {y_train.sum()}, Non-defaulters: {(y_train==0).sum()}")
print(f"After SMOTE  — Defaulters: {y_train_res.sum()}, Non-defaulters: {(y_train_res==0).sum()}")

os.makedirs("data/processed", exist_ok=True)
os.makedirs("ml_model", exist_ok=True)

joblib.dump((X_train_res, y_train_res, X_test_scaled, y_test),
            "data/processed/train_test_data.pkl")
joblib.dump(scaler, "ml_model/scaler.pkl")
joblib.dump(X.columns.tolist(), "ml_model/feature_names.pkl")

print(f"\nFeature names saved: {X.columns.tolist()}")
print("DONE — preprocessing complete")