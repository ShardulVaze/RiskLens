import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

print("START")
X_train, y_train, X_test, y_test = joblib.load(
    "data/processed/train_test_data.pkl"
)

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])


model = XGBClassifier(
    n_estimators=1000,
    max_depth=10,
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

model.fit(X_train, y_train)


y_proba = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
best_threshold = thresholds[np.argmax(tpr - fpr)]

print("Best Threshold =", best_threshold)

y_pred = (y_proba > best_threshold).astype(int)

print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:", roc_auc)

joblib.dump(model, "ml_model/model.pkl")
joblib.dump(best_threshold, "ml_model/threshold.pkl")

print("DONE")
