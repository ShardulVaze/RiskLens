import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from database.db_config import DATABASE_URL

print("START")

engine = create_engine(DATABASE_URL)
df = pd.read_sql("SELECT * FROM vw_risk_model_dataset", engine)

print(len(df))

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

numeric_cols = [c for c in numeric_cols if c not in ["SK_ID_CURR", "TARGET"]]

X = df[numeric_cols]
y = df["TARGET"]

X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print(f"Before SMOTE Frauds: {sum(y_train)}")
print(f"After SMOTE Frauds:  {sum(y_train_res)}")

os.makedirs("data/processed", exist_ok=True)
os.makedirs("ml_model", exist_ok=True)

joblib.dump((X_train_res, y_train_res, X_test_scaled, y_test),
            "data/processed/train_test_data.pkl")

joblib.dump(scaler, "ml_model/scaler.pkl")

joblib.dump(X.columns.tolist(), "ml_model/feature_names.pkl")


print("DONE")
