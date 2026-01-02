import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_selection import SelectFromModel

print("Start")
X_train, y_train, X_test, y_test = joblib.load(
    "data/processed/train_test_data.pkl"
)

selector = SelectFromModel(XGBClassifier(
    n_estimators=300, random_state=42, n_jobs=-1
))
selector.fit(X_train, y_train)

X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)

print(f"Selected top {X_train_sel.shape[1]} features")

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])


model = XGBClassifier(
    n_estimators=900,
    max_depth=9,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=1.5,
    min_child_weight=2,
    reg_lambda=2.5,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_sel, y_train)

y_proba = model.predict_proba(X_test_sel)[:, 1]
y_pred = (y_proba > 0.20).astype(int)  

print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:", roc_auc)

joblib.dump(model, "ml_model/model_xgb_v4.pkl")
print("Done")
