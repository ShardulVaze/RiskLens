import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score


X_train, y_train, X_test, y_test = joblib.load(
    "data/processed/train_test_data.pkl"
)

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

model = XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Evaluating Model")
y_pred = (model.predict_proba(X_test)[:, 1] > 0.35).astype(int)

print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print("ROC-AUC:", roc_auc)

joblib.dump(model, "ml_model/model_xgb.pkl")
print("DONE")
